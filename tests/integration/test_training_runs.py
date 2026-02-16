"""Tests for DPO training client with new schema."""

import pytest

from llm_learn import NotFoundError, ValidationError
from llm_learn.training import DpoClient, PairTuple


@pytest.fixture
def dpo_client(logger, database, test_context):
    """Create DpoClient for testing."""
    return DpoClient(
        lg=logger,
        session_factory=database.session,
        context_key=test_context,
    )


@pytest.fixture
def sample_preferences(learn_client, clean_tables):
    """Create sample preference pairs for testing."""
    ids = []
    for i in range(5):
        fact_id = learn_client.atomic.preferences.record(
            context=f"Context {i}",
            chosen=f"Good response {i}",
            rejected=f"Bad response {i}",
            margin=0.5 + (i * 0.1),  # 0.5, 0.6, 0.7, 0.8, 0.9
            category="test",
        )
        ids.append(fact_id)
    return ids


def make_pairs(fact_ids: list[int]) -> list[PairTuple]:
    """Convert fact IDs to pair tuples for testing.

    Uses fact_id for both chosen and rejected (same preference fact).
    """
    return [(fid, fid, f"Context {i}") for i, fid in enumerate(fact_ids)]


class TestTrainingRunCRUD:
    """Test basic CRUD operations for training runs."""

    def test_create_run(self, dpo_client, clean_tables):
        """Test creating a training run."""
        run = dpo_client.create(adapter_name="test-adapter")

        assert run.id > 0
        assert run.method == "dpo"
        assert run.adapter_name == "test-adapter"
        assert run.adapter == {"name": "test-adapter"}
        assert run.status == "pending"
        assert run.config is None
        assert run.metrics is None
        assert run.based_on is None
        assert run.system_status is None
        assert run.is_deleted is False
        assert run.created_at is not None
        assert run.started_at is None
        assert run.completed_at is None

    def test_create_run_with_config(self, dpo_client, clean_tables):
        """Test creating a run with config."""
        config = {"learning_rate": 1e-4, "epochs": 3}
        run = dpo_client.create(adapter_name="test", config=config)

        assert run.config == config

    def test_create_run_with_based_on(self, dpo_client, clean_tables):
        """Test creating a run with lineage (based_on)."""
        parent = dpo_client.create(adapter_name="parent")
        child = dpo_client.create(adapter_name="child", based_on=parent.id)

        assert child.based_on == parent.id

    def test_create_run_with_invalid_based_on(self, dpo_client, clean_tables):
        """Test creating a run with non-existent parent raises NotFoundError."""
        with pytest.raises(NotFoundError, match="Parent run 99999 not found"):
            dpo_client.create(adapter_name="orphan", based_on=99999)

    def test_get_run(self, dpo_client, clean_tables):
        """Test getting a run by ID."""
        created = dpo_client.create(adapter_name="get-test")
        fetched = dpo_client.get(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.adapter_name == "get-test"

    def test_get_nonexistent_run(self, dpo_client, clean_tables):
        """Test getting a nonexistent run returns None."""
        result = dpo_client.get(99999)
        assert result is None

    def test_list_runs(self, dpo_client, clean_tables):
        """Test listing runs."""
        dpo_client.create(adapter_name="run-1")
        dpo_client.create(adapter_name="run-2")
        dpo_client.create(adapter_name="run-3")

        runs = dpo_client.list_runs()
        assert len(runs) == 3

    def test_list_runs_with_status_filter(self, dpo_client, clean_tables):
        """Test listing runs filtered by status."""
        dpo_client.create(adapter_name="pending-run")
        started_run = dpo_client.create(adapter_name="started-run")
        dpo_client.start(started_run.id)

        pending = dpo_client.list_runs(status="pending")
        assert len(pending) == 1
        assert pending[0].adapter_name == "pending-run"

        running = dpo_client.list_runs(status="running")
        assert len(running) == 1
        assert running[0].adapter_name == "started-run"

    def test_list_runs_invalid_status(self, dpo_client, clean_tables):
        """Test listing with invalid status raises ValidationError."""
        with pytest.raises(ValidationError, match="status must be one of"):
            dpo_client.list_runs(status="invalid")

    def test_list_runs_descending(self, dpo_client, clean_tables):
        """Test listing runs in descending order (default)."""
        dpo_client.create(adapter_name="first")
        dpo_client.create(adapter_name="second")
        dpo_client.create(adapter_name="third")

        runs = dpo_client.list_runs(descending=True)
        assert runs[0].adapter_name == "third"  # Most recent first

    def test_delete_run(self, dpo_client, clean_tables):
        """Test soft-deleting a run."""
        run = dpo_client.create(adapter_name="to-delete")

        result = dpo_client.delete(run.id)
        assert result is True

        # Soft-deleted - not visible by default
        assert dpo_client.get(run.id) is None

        # But visible with include_deleted
        all_runs = dpo_client.list_runs(include_deleted=True)
        deleted_run = next((r for r in all_runs if r.id == run.id), None)
        assert deleted_run is not None
        assert deleted_run.is_deleted is True
        assert deleted_run.system_status is not None
        assert deleted_run.system_status.get("deleted") is True

    def test_delete_nonexistent_run(self, dpo_client, clean_tables):
        """Test deleting a nonexistent run returns False."""
        result = dpo_client.delete(99999)
        assert result is False


class TestPairAssignment:
    """Test preference pair assignment with new schema."""

    def test_assign_pairs(self, dpo_client, sample_preferences):
        """Test assigning pairs to a run."""
        run = dpo_client.create(adapter_name="assign-test")
        pairs = make_pairs(sample_preferences[:3])
        count = dpo_client.assign_pairs(run.id, pairs)

        assert count == 3

    def test_assign_empty_list(self, dpo_client, clean_tables):
        """Test assigning empty list returns 0."""
        run = dpo_client.create()
        count = dpo_client.assign_pairs(run.id, [])
        assert count == 0

    def test_assign_to_nonexistent_run(self, dpo_client, sample_preferences):
        """Test assigning to nonexistent run raises NotFoundError."""
        pairs = make_pairs(sample_preferences[:1])
        with pytest.raises(NotFoundError, match="not found"):
            dpo_client.assign_pairs(99999, pairs)

    def test_get_pairs(self, dpo_client, sample_preferences):
        """Test getting pending pairs assigned to a run."""
        run = dpo_client.create(adapter_name="get-pairs-test")
        pairs = make_pairs(sample_preferences[:3])
        dpo_client.assign_pairs(run.id, pairs)

        retrieved = dpo_client.get_pairs(run.id)
        assert len(retrieved) == 3

        # Verify structure - pairs are (chosen_id, rejected_id, prompt)
        for chosen_id, rejected_id, prompt in retrieved:
            assert chosen_id in sample_preferences[:3]
            assert rejected_id == chosen_id  # Same fact for both
            assert prompt.startswith("Context")

    def test_get_pairs_nonexistent_run(self, dpo_client, clean_tables):
        """Test getting pairs from nonexistent run raises NotFoundError."""
        with pytest.raises(NotFoundError, match="not found"):
            dpo_client.get_pairs(99999)

    def test_count_pending_pairs(self, dpo_client, sample_preferences):
        """Test counting pending pairs."""
        run = dpo_client.create(adapter_name="count-test")
        pairs = make_pairs(sample_preferences[:3])
        dpo_client.assign_pairs(run.id, pairs)

        # Count for specific run
        assert dpo_client.count_pending_pairs(run.id) == 3

        # Count for all runs in context
        assert dpo_client.count_pending_pairs() == 3


class TestLineageFiltering:
    """Test lineage-based pair filtering."""

    def test_pairs_filtered_by_lineage(self, dpo_client, sample_preferences):
        """Test that pairs used in ancestor runs are filtered out."""
        pairs = make_pairs(sample_preferences)

        # Create parent run and assign first 3 pairs
        parent = dpo_client.create(adapter_name="parent")
        dpo_client.assign_pairs(parent.id, pairs[:3])
        dpo_client.start(parent.id)
        dpo_client.complete(parent.id)

        # Create child run based on parent
        child = dpo_client.create(adapter_name="child", based_on=parent.id)

        # Try to assign all 5 pairs - should only get 2 (not used in parent)
        assigned = dpo_client.assign_pairs(child.id, pairs)
        assert assigned == 2

        # Verify the assigned pairs are the last 2
        child_pairs = dpo_client.get_pairs(child.id)
        assert len(child_pairs) == 2

    def test_multi_level_lineage(self, dpo_client, sample_preferences):
        """Test lineage filtering works across multiple generations."""
        pairs = make_pairs(sample_preferences)

        # Grandparent uses pair 0
        grandparent = dpo_client.create(adapter_name="grandparent")
        dpo_client.assign_pairs(grandparent.id, [pairs[0]])
        dpo_client.start(grandparent.id)
        dpo_client.complete(grandparent.id)

        # Parent uses pairs 1-2
        parent = dpo_client.create(adapter_name="parent", based_on=grandparent.id)
        dpo_client.assign_pairs(parent.id, pairs[1:3])
        dpo_client.start(parent.id)
        dpo_client.complete(parent.id)

        # Child should only get pairs 3-4
        child = dpo_client.create(adapter_name="child", based_on=parent.id)
        assigned = dpo_client.assign_pairs(child.id, pairs)
        assert assigned == 2

    def test_no_lineage_allows_all_pairs(self, dpo_client, sample_preferences):
        """Test that runs without lineage can use any pairs."""
        pairs = make_pairs(sample_preferences)

        run1 = dpo_client.create(adapter_name="run1")
        dpo_client.assign_pairs(run1.id, pairs[:3])

        # Run2 has no lineage relationship to run1
        run2 = dpo_client.create(adapter_name="run2")
        # All pairs available since no lineage
        assigned = dpo_client.assign_pairs(run2.id, pairs)
        assert assigned == 5


class TestPairMovement:
    """Test pair movement from pending to trained on completion."""

    def test_pairs_move_to_trained_on_complete(self, dpo_client, sample_preferences):
        """Test that completing a run moves pairs to trained table."""
        run = dpo_client.create(adapter_name="complete-test")
        pairs = make_pairs(sample_preferences[:3])
        dpo_client.assign_pairs(run.id, pairs)

        # Before completion - pairs are pending
        assert len(dpo_client.get_pairs(run.id)) == 3
        assert len(dpo_client.get_trained_pairs(run.id)) == 0

        dpo_client.start(run.id)
        dpo_client.complete(run.id)

        # After completion - pairs moved to trained
        assert len(dpo_client.get_pairs(run.id)) == 0
        assert len(dpo_client.get_trained_pairs(run.id)) == 3

    def test_pairs_cleared_on_fail(self, dpo_client, sample_preferences):
        """Test that failing a run clears pending pairs (frees for reuse)."""
        run = dpo_client.create(adapter_name="fail-test")
        pairs = make_pairs(sample_preferences[:3])
        dpo_client.assign_pairs(run.id, pairs)

        dpo_client.start(run.id)
        dpo_client.fail(run.id, "Test failure")

        # Pairs cleared (freed for reuse)
        assert len(dpo_client.get_pairs(run.id)) == 0
        assert len(dpo_client.get_trained_pairs(run.id)) == 0

    def test_pairs_freed_after_run_delete(self, dpo_client, sample_preferences):
        """Test that soft-deleting a run clears pending pairs."""
        run = dpo_client.create(adapter_name="to-delete")
        pairs = make_pairs(sample_preferences[:3])
        dpo_client.assign_pairs(run.id, pairs)

        # Delete the run
        dpo_client.delete(run.id)

        # Pending pairs should be cleared
        assert dpo_client.count_pending_pairs() == 0


class TestLifecycle:
    """Test training run lifecycle transitions."""

    def test_start_run(self, dpo_client, clean_tables):
        """Test starting a run transitions to running."""
        run = dpo_client.create()
        dpo_client.start(run.id)

        updated = dpo_client.get(run.id)
        assert updated.status == "running"
        assert updated.started_at is not None

    def test_complete_run(self, dpo_client, clean_tables):
        """Test completing a run."""
        run = dpo_client.create()
        dpo_client.start(run.id)
        dpo_client.complete(run.id, metrics={"loss": 0.01})

        updated = dpo_client.get(run.id)
        assert updated.status == "completed"
        assert updated.completed_at is not None
        assert updated.metrics == {"loss": 0.01}

    def test_fail_run(self, dpo_client, clean_tables):
        """Test failing a run."""
        run = dpo_client.create()
        dpo_client.start(run.id)
        dpo_client.fail(run.id, "Out of memory")

        updated = dpo_client.get(run.id)
        assert updated.status == "failed"
        assert updated.completed_at is not None
        assert updated.error_message == "Out of memory"

    def test_fail_from_pending(self, dpo_client, clean_tables):
        """Test that pending runs can fail directly."""
        run = dpo_client.create()
        dpo_client.fail(run.id, "Cancelled")

        updated = dpo_client.get(run.id)
        assert updated.status == "failed"

    def test_invalid_transition_pending_to_completed(self, dpo_client, clean_tables):
        """Test that pending cannot go directly to completed."""
        run = dpo_client.create()
        with pytest.raises(ValidationError, match="Cannot transition"):
            dpo_client.complete(run.id)

    def test_invalid_transition_completed_to_failed(self, dpo_client, clean_tables):
        """Test that completed runs cannot transition (terminal state)."""
        run = dpo_client.create()
        dpo_client.start(run.id)
        dpo_client.complete(run.id)

        with pytest.raises(ValidationError, match="Cannot transition"):
            dpo_client.fail(run.id, "error")

    def test_invalid_transition_failed_to_running(self, dpo_client, clean_tables):
        """Test that failed runs cannot restart (terminal state)."""
        run = dpo_client.create()
        dpo_client.start(run.id)
        dpo_client.fail(run.id, "error")

        with pytest.raises(ValidationError, match="Cannot transition"):
            dpo_client.start(run.id)

    def test_start_nonexistent_run(self, dpo_client, clean_tables):
        """Test starting nonexistent run raises NotFoundError."""
        with pytest.raises(NotFoundError, match="not found"):
            dpo_client.start(99999)


class TestContextIsolation:
    """Test context-based isolation."""

    def test_runs_isolated_by_context(self, logger, database, clean_tables):
        """Test that runs are isolated by context_key."""
        client_a = DpoClient(lg=logger, session_factory=database.session, context_key="context-a")
        client_b = DpoClient(lg=logger, session_factory=database.session, context_key="context-b")

        # Create runs in each context
        run_a = client_a.create(adapter_name="run-a")
        run_b = client_b.create(adapter_name="run-b")

        # Each client should only see its own runs
        assert len(client_a.list_runs()) == 1
        assert client_a.list_runs()[0].adapter_name == "run-a"

        assert len(client_b.list_runs()) == 1
        assert client_b.list_runs()[0].adapter_name == "run-b"

        # Cross-context get should return None
        assert client_a.get(run_b.id) is None
        assert client_b.get(run_a.id) is None

    def test_glob_pattern_context(self, logger, database, clean_tables):
        """Test glob pattern matching for context."""
        # Create runs with hierarchical contexts
        client_a1 = DpoClient(
            lg=logger, session_factory=database.session, context_key="tenant:a:agent1"
        )
        client_a2 = DpoClient(
            lg=logger, session_factory=database.session, context_key="tenant:a:agent2"
        )
        client_b = DpoClient(
            lg=logger, session_factory=database.session, context_key="tenant:b:agent1"
        )

        client_a1.create(adapter_name="a1-run")
        client_a2.create(adapter_name="a2-run")
        client_b.create(adapter_name="b-run")

        # Glob pattern to match all tenant:a:* runs
        client_a_all = DpoClient(
            lg=logger, session_factory=database.session, context_key="tenant:a:*"
        )
        runs = client_a_all.list_runs()
        assert len(runs) == 2
        assert {r.adapter_name for r in runs} == {"a1-run", "a2-run"}

    def test_create_with_glob_pattern_raises(self, logger, database, clean_tables):
        """Test that creating a run with glob-pattern context raises ValidationError."""
        client = DpoClient(lg=logger, session_factory=database.session, context_key="tenant:*")

        # Listing works with glob pattern
        assert client.list_runs() == []

        # Creating should fail - glob patterns are read-only
        with pytest.raises(ValidationError, match="glob-pattern"):
            client.create(adapter_name="should-fail")


class TestResetAll:
    """Test reset_all functionality."""

    def test_reset_all(self, dpo_client, sample_preferences):
        """Test resetting all runs in context."""
        pairs = make_pairs(sample_preferences)

        # Create runs and assign pairs
        run1 = dpo_client.create(adapter_name="run-1")
        run2 = dpo_client.create(adapter_name="run-2")
        dpo_client.assign_pairs(run1.id, pairs[:2])
        dpo_client.assign_pairs(run2.id, pairs[2:4])

        # Verify pairs are assigned
        assert dpo_client.count_pending_pairs() == 4

        # Reset all
        count = dpo_client.reset_all()
        assert count == 2

        # Verify runs are soft-deleted (not visible by default)
        assert len(dpo_client.list_runs()) == 0

        # Verify pending pairs are cleared
        assert dpo_client.count_pending_pairs() == 0

    def test_reset_all_empty(self, dpo_client, clean_tables):
        """Test reset_all with no runs returns 0."""
        count = dpo_client.reset_all()
        assert count == 0

    def test_clear_pending_pairs_for_context(self, dpo_client, sample_preferences):
        """Test clearing pending pairs without deleting runs."""
        pairs = make_pairs(sample_preferences[:3])
        run = dpo_client.create(adapter_name="test-run")
        dpo_client.assign_pairs(run.id, pairs)

        assert dpo_client.count_pending_pairs() == 3

        # Clear pending pairs
        cleared = dpo_client.clear_pending_pairs_for_context()
        assert cleared == 3

        # Run still exists
        assert dpo_client.get(run.id) is not None
        # But pairs are gone
        assert dpo_client.count_pending_pairs() == 0
