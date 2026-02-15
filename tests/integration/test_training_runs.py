"""Tests for training run tracking."""

import pytest

from llm_learn import ConflictError, NotFoundError, ValidationError
from llm_learn.training import TrainingRunClient


@pytest.fixture
def runs_client(logger, database, test_context):
    """Create TrainingRunClient for testing."""
    return TrainingRunClient(
        lg=logger,
        session_factory=database.session,
        context_key=test_context,
    )


@pytest.fixture
def sample_preferences(learn_client, clean_tables):
    """Create sample preference pairs for testing."""
    ids = []
    for i in range(5):
        fact_id = learn_client.preferences.record(
            context=f"Context {i}",
            chosen=f"Good response {i}",
            rejected=f"Bad response {i}",
            margin=0.5 + (i * 0.1),  # 0.5, 0.6, 0.7, 0.8, 0.9
            category="test",
        )
        ids.append(fact_id)
    return ids


class TestTrainingRunCRUD:
    """Test basic CRUD operations for training runs."""

    def test_create_run(self, runs_client, clean_tables):
        """Test creating a training run."""
        run = runs_client.create(adapter_name="test-adapter")

        assert run.id > 0
        assert run.adapter_name == "test-adapter"
        assert run.status == "pending"
        assert run.config is None
        assert run.metrics is None
        assert run.created_at is not None
        assert run.started_at is None
        assert run.completed_at is None

    def test_create_run_with_config(self, runs_client, clean_tables):
        """Test creating a run with config."""
        config = {"learning_rate": 1e-4, "epochs": 3}
        run = runs_client.create(adapter_name="test", config=config)

        assert run.config == config

    def test_get_run(self, runs_client, clean_tables):
        """Test getting a run by ID."""
        created = runs_client.create(adapter_name="get-test")
        fetched = runs_client.get(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.adapter_name == "get-test"

    def test_get_nonexistent_run(self, runs_client, clean_tables):
        """Test getting a nonexistent run returns None."""
        result = runs_client.get(99999)
        assert result is None

    def test_list_runs(self, runs_client, clean_tables):
        """Test listing runs."""
        runs_client.create(adapter_name="run-1")
        runs_client.create(adapter_name="run-2")
        runs_client.create(adapter_name="run-3")

        runs = runs_client.list()
        assert len(runs) == 3

    def test_list_runs_with_status_filter(self, runs_client, clean_tables):
        """Test listing runs filtered by status."""
        runs_client.create(adapter_name="pending-run")
        started_run = runs_client.create(adapter_name="started-run")
        runs_client.start(started_run.id)

        pending = runs_client.list(status="pending")
        assert len(pending) == 1
        assert pending[0].adapter_name == "pending-run"

        running = runs_client.list(status="running")
        assert len(running) == 1
        assert running[0].adapter_name == "started-run"

    def test_list_runs_invalid_status(self, runs_client, clean_tables):
        """Test listing with invalid status raises ValidationError."""
        with pytest.raises(ValidationError, match="status must be one of"):
            runs_client.list(status="invalid")

    def test_list_runs_descending(self, runs_client, clean_tables):
        """Test listing runs in descending order (default)."""
        runs_client.create(adapter_name="first")
        runs_client.create(adapter_name="second")
        runs_client.create(adapter_name="third")

        runs = runs_client.list(descending=True)
        assert runs[0].adapter_name == "third"  # Most recent first

    def test_delete_run(self, runs_client, clean_tables):
        """Test deleting a run."""
        run = runs_client.create(adapter_name="to-delete")

        result = runs_client.delete(run.id)
        assert result is True

        # Verify deleted
        assert runs_client.get(run.id) is None

    def test_delete_nonexistent_run(self, runs_client, clean_tables):
        """Test deleting a nonexistent run returns False."""
        result = runs_client.delete(99999)
        assert result is False


class TestPairAssignment:
    """Test preference pair assignment."""

    def test_assign_pairs(self, runs_client, sample_preferences):
        """Test assigning pairs to a run."""
        run = runs_client.create(adapter_name="assign-test")
        count = runs_client.assign_pairs(run.id, sample_preferences[:3])

        assert count == 3

    def test_assign_empty_list(self, runs_client, clean_tables):
        """Test assigning empty list returns 0."""
        run = runs_client.create()
        count = runs_client.assign_pairs(run.id, [])
        assert count == 0

    def test_assign_to_nonexistent_run(self, runs_client, sample_preferences):
        """Test assigning to nonexistent run raises NotFoundError."""
        with pytest.raises(NotFoundError, match="not found"):
            runs_client.assign_pairs(99999, sample_preferences[:1])

    def test_exclusive_assignment_constraint(self, runs_client, sample_preferences):
        """Test that pairs can only be assigned to one run (exclusive)."""
        run1 = runs_client.create(adapter_name="run-1")
        run2 = runs_client.create(adapter_name="run-2")

        # Assign pairs to first run
        runs_client.assign_pairs(run1.id, sample_preferences[:3])

        # Try to assign same pairs to second run - should fail
        with pytest.raises(ConflictError, match="already assigned"):
            runs_client.assign_pairs(run2.id, sample_preferences[:3])

    def test_partial_overlap_fails(self, runs_client, sample_preferences):
        """Test that partial overlap in assignment fails."""
        run1 = runs_client.create(adapter_name="run-1")
        run2 = runs_client.create(adapter_name="run-2")

        # Assign first 3 to run1
        runs_client.assign_pairs(run1.id, sample_preferences[:3])

        # Try to assign pairs 2-4 to run2 (overlaps on 2,3)
        with pytest.raises(ConflictError, match="already assigned"):
            runs_client.assign_pairs(run2.id, sample_preferences[2:5])

    def test_get_pairs(self, runs_client, sample_preferences):
        """Test getting pairs assigned to a run."""
        run = runs_client.create(adapter_name="get-pairs-test")
        runs_client.assign_pairs(run.id, sample_preferences[:3])

        pairs = runs_client.get_pairs(run.id)
        assert len(pairs) == 3

        # Verify structure
        for fact, details in pairs:
            assert fact.type == "preference"
            assert details is not None
            assert details.context.startswith("Context")

    def test_get_pairs_nonexistent_run(self, runs_client, clean_tables):
        """Test getting pairs from nonexistent run raises NotFoundError."""
        with pytest.raises(NotFoundError, match="not found"):
            runs_client.get_pairs(99999)


class TestUntrainedPairs:
    """Test get_untrained_pairs functionality."""

    def test_get_untrained_pairs_all(self, runs_client, sample_preferences):
        """Test getting all untrained pairs (none assigned yet)."""
        pairs = runs_client.get_untrained_pairs()
        assert len(pairs) == 5

    def test_get_untrained_pairs_excludes_assigned(self, runs_client, sample_preferences):
        """Test that assigned pairs are excluded."""
        run = runs_client.create(adapter_name="test")
        runs_client.assign_pairs(run.id, sample_preferences[:3])

        untrained = runs_client.get_untrained_pairs()
        assert len(untrained) == 2  # 5 total - 3 assigned = 2 untrained

    def test_get_untrained_pairs_min_margin(self, runs_client, sample_preferences):
        """Test filtering by minimum margin."""
        # Pairs have margins: 0.5, 0.6, 0.7, 0.8, 0.9
        pairs = runs_client.get_untrained_pairs(min_margin=0.7)
        assert len(pairs) == 3  # 0.7, 0.8, 0.9

    def test_get_untrained_pairs_limit(self, runs_client, sample_preferences):
        """Test limiting results."""
        pairs = runs_client.get_untrained_pairs(limit=2)
        assert len(pairs) == 2

    def test_get_untrained_pairs_combined_filters(self, runs_client, sample_preferences):
        """Test combining min_margin and limit."""
        pairs = runs_client.get_untrained_pairs(min_margin=0.6, limit=2)
        assert len(pairs) == 2

    def test_pairs_freed_after_run_delete(self, runs_client, sample_preferences):
        """Test that deleting a run frees its pairs for reuse."""
        run = runs_client.create(adapter_name="to-delete")
        runs_client.assign_pairs(run.id, sample_preferences[:3])

        # Verify pairs are assigned
        assert len(runs_client.get_untrained_pairs()) == 2

        # Delete the run
        runs_client.delete(run.id)

        # All pairs should be untrained now
        assert len(runs_client.get_untrained_pairs()) == 5


class TestLifecycle:
    """Test training run lifecycle transitions."""

    def test_start_run(self, runs_client, clean_tables):
        """Test starting a run transitions to running."""
        run = runs_client.create()
        runs_client.start(run.id)

        updated = runs_client.get(run.id)
        assert updated.status == "running"
        assert updated.started_at is not None

    def test_complete_run(self, runs_client, clean_tables):
        """Test completing a run."""
        run = runs_client.create()
        runs_client.start(run.id)
        runs_client.complete(run.id, metrics={"loss": 0.01})

        updated = runs_client.get(run.id)
        assert updated.status == "completed"
        assert updated.completed_at is not None
        assert updated.metrics == {"loss": 0.01}

    def test_fail_run(self, runs_client, clean_tables):
        """Test failing a run."""
        run = runs_client.create()
        runs_client.start(run.id)
        runs_client.fail(run.id, "Out of memory")

        updated = runs_client.get(run.id)
        assert updated.status == "failed"
        assert updated.completed_at is not None
        assert updated.error_message == "Out of memory"

    def test_fail_from_pending(self, runs_client, clean_tables):
        """Test that pending runs can fail directly."""
        run = runs_client.create()
        runs_client.fail(run.id, "Cancelled")

        updated = runs_client.get(run.id)
        assert updated.status == "failed"

    def test_invalid_transition_pending_to_completed(self, runs_client, clean_tables):
        """Test that pending cannot go directly to completed."""
        run = runs_client.create()
        with pytest.raises(ValidationError, match="Cannot transition"):
            runs_client.complete(run.id)

    def test_invalid_transition_completed_to_failed(self, runs_client, clean_tables):
        """Test that completed runs cannot transition."""
        run = runs_client.create()
        runs_client.start(run.id)
        runs_client.complete(run.id)

        with pytest.raises(ValidationError, match="Cannot transition.*terminal"):
            runs_client.fail(run.id, "error")

    def test_invalid_transition_failed_to_running(self, runs_client, clean_tables):
        """Test that failed runs cannot restart."""
        run = runs_client.create()
        runs_client.start(run.id)
        runs_client.fail(run.id, "error")

        with pytest.raises(ValidationError, match="Cannot transition.*terminal"):
            runs_client.start(run.id)

    def test_start_nonexistent_run(self, runs_client, clean_tables):
        """Test starting nonexistent run raises NotFoundError."""
        with pytest.raises(NotFoundError, match="not found"):
            runs_client.start(99999)


class TestContextIsolation:
    """Test context-based isolation."""

    def test_runs_isolated_by_context(self, logger, database, clean_tables):
        """Test that runs are isolated by context_key."""
        client_a = TrainingRunClient(
            lg=logger, session_factory=database.session, context_key="context-a"
        )
        client_b = TrainingRunClient(
            lg=logger, session_factory=database.session, context_key="context-b"
        )

        # Create runs in each context
        run_a = client_a.create(adapter_name="run-a")
        run_b = client_b.create(adapter_name="run-b")

        # Each client should only see its own runs
        assert len(client_a.list()) == 1
        assert client_a.list()[0].adapter_name == "run-a"

        assert len(client_b.list()) == 1
        assert client_b.list()[0].adapter_name == "run-b"

        # Cross-context get should return None
        assert client_a.get(run_b.id) is None
        assert client_b.get(run_a.id) is None

    def test_glob_pattern_context(self, logger, database, clean_tables):
        """Test glob pattern matching for context."""
        # Create runs with hierarchical contexts
        client_a1 = TrainingRunClient(
            lg=logger, session_factory=database.session, context_key="tenant:a:agent1"
        )
        client_a2 = TrainingRunClient(
            lg=logger, session_factory=database.session, context_key="tenant:a:agent2"
        )
        client_b = TrainingRunClient(
            lg=logger, session_factory=database.session, context_key="tenant:b:agent1"
        )

        client_a1.create(adapter_name="a1-run")
        client_a2.create(adapter_name="a2-run")
        client_b.create(adapter_name="b-run")

        # Glob pattern to match all tenant:a:* runs
        client_a_all = TrainingRunClient(
            lg=logger, session_factory=database.session, context_key="tenant:a:*"
        )
        runs = client_a_all.list()
        assert len(runs) == 2
        assert {r.adapter_name for r in runs} == {"a1-run", "a2-run"}


class TestResetAll:
    """Test reset_all functionality."""

    def test_reset_all(self, runs_client, sample_preferences):
        """Test resetting all runs in context."""
        # Create runs and assign pairs
        run1 = runs_client.create(adapter_name="run-1")
        run2 = runs_client.create(adapter_name="run-2")
        runs_client.assign_pairs(run1.id, sample_preferences[:2])
        runs_client.assign_pairs(run2.id, sample_preferences[2:4])

        # Verify pairs are assigned
        assert len(runs_client.get_untrained_pairs()) == 1

        # Reset all
        count = runs_client.reset_all()
        assert count == 2

        # Verify runs are gone
        assert len(runs_client.list()) == 0

        # Verify all pairs are now untrained
        assert len(runs_client.get_untrained_pairs()) == 5

    def test_reset_all_empty(self, runs_client, clean_tables):
        """Test reset_all with no runs returns 0."""
        count = runs_client.reset_all()
        assert count == 0
