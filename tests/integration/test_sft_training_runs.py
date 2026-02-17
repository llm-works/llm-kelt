"""Tests for SFT training client."""

import pytest

from llm_learn import NotFoundError, ValidationError
from llm_learn.training.sft import Client as SftClient


@pytest.fixture
def sft_client(logger, database, test_context):
    """Create SftClient for testing."""
    return SftClient(
        lg=logger,
        session_factory=database.session,
        context_key=test_context,
    )


@pytest.fixture
def sample_feedback_facts(learn_client, clean_tables):
    """Create sample feedback facts for SFT testing."""
    ids = []
    for i in range(5):
        # Create content first
        content_id = learn_client.content.create(
            content_text=f"Good response content {i}",
            source="test",
        )
        # Record feedback on the content
        fact_id = learn_client.atomic.feedback.record(
            signal="positive",
            content_id=content_id,
            strength=0.8,
            tags=["test"],
        )
        ids.append(fact_id)
    return ids


class TestSftRunCRUD:
    """Test basic CRUD operations for SFT training runs."""

    def test_create_run(self, sft_client, clean_tables):
        """Test creating an SFT training run."""
        run = sft_client.create(adapter_name="test-sft-adapter")

        assert run.id > 0
        assert run.method == "sft"
        assert run.adapter_name == "test-sft-adapter"
        assert run.adapter == {"name": "test-sft-adapter"}
        assert run.status == "pending"
        assert run.config is None
        assert run.metrics is None
        assert run.based_on is None
        assert run.is_deleted is False

    def test_create_run_with_config(self, sft_client, clean_tables):
        """Test creating a run with config."""
        config = {"learning_rate": 1e-4, "epochs": 3}
        run = sft_client.create(adapter_name="test", config=config)

        assert run.config == config

    def test_create_run_with_based_on(self, sft_client, clean_tables):
        """Test creating a run with lineage."""
        parent = sft_client.create(adapter_name="parent")
        child = sft_client.create(adapter_name="child", based_on=parent.id)

        assert child.based_on == parent.id

    def test_create_run_with_invalid_based_on(self, sft_client, clean_tables):
        """Test creating a run with non-existent parent raises NotFoundError."""
        with pytest.raises(NotFoundError, match="Parent run 99999 not found"):
            sft_client.create(adapter_name="orphan", based_on=99999)

    def test_replace_stale_reuses_pending_run(self, sft_client, clean_tables):
        """Test that replace_stale=True reuses existing pending run."""
        first = sft_client.create(adapter_name="first")
        second = sft_client.create(adapter_name="second")

        assert second.id == first.id
        assert len(sft_client.list_runs()) == 1

    def test_replace_stale_false_creates_new(self, sft_client, clean_tables):
        """Test that replace_stale=False always creates a new run."""
        first = sft_client.create(adapter_name="first")
        second = sft_client.create(adapter_name="second", replace_stale=False)

        assert second.id != first.id
        assert len(sft_client.list_runs()) == 2

    def test_get_run(self, sft_client, clean_tables):
        """Test getting a run by ID."""
        created = sft_client.create(adapter_name="get-test")
        fetched = sft_client.get(created.id)

        assert fetched is not None
        assert fetched.id == created.id
        assert fetched.adapter_name == "get-test"

    def test_get_nonexistent_run(self, sft_client, clean_tables):
        """Test getting a nonexistent run returns None."""
        result = sft_client.get(99999)
        assert result is None

    def test_list_runs(self, sft_client, clean_tables):
        """Test listing runs."""
        sft_client.create(adapter_name="run-1", replace_stale=False)
        sft_client.create(adapter_name="run-2", replace_stale=False)
        sft_client.create(adapter_name="run-3", replace_stale=False)

        runs = sft_client.list_runs()
        assert len(runs) == 3

    def test_list_runs_with_status_filter(self, sft_client, clean_tables):
        """Test listing runs filtered by status."""
        sft_client.create(adapter_name="pending-run", replace_stale=False)
        started_run = sft_client.create(adapter_name="started-run", replace_stale=False)
        sft_client.start(started_run.id)

        pending = sft_client.list_runs(status="pending")
        assert len(pending) == 1
        assert pending[0].adapter_name == "pending-run"

        running = sft_client.list_runs(status="running")
        assert len(running) == 1
        assert running[0].adapter_name == "started-run"

    def test_delete_pending_run_cancels(self, sft_client, clean_tables):
        """Test deleting a pending run transitions to cancelled status."""
        run = sft_client.create(adapter_name="to-cancel")
        result = sft_client.delete(run.id)
        assert result is True

        cancelled = sft_client.get(run.id)
        assert cancelled is not None
        assert cancelled.status == "cancelled"

    def test_delete_completed_run_soft_deletes(self, sft_client, clean_tables):
        """Test deleting a completed run soft-deletes it."""
        run = sft_client.create(adapter_name="to-soft-delete")
        sft_client.start(run.id)
        sft_client.complete(run.id, metrics={"loss": 0.1})

        result = sft_client.delete(run.id)
        assert result is True

        # Soft-deleted - not visible by default
        assert sft_client.get(run.id) is None


class TestExampleAssignment:
    """Test example assignment for SFT runs."""

    def test_assign_examples(self, sft_client, sample_feedback_facts):
        """Test assigning examples to a run."""
        run = sft_client.create(adapter_name="assign-test")
        count = sft_client.assign_examples(run.id, sample_feedback_facts[:3])

        assert count == 3

    def test_assign_empty_list(self, sft_client, clean_tables):
        """Test assigning empty list returns 0."""
        run = sft_client.create()
        count = sft_client.assign_examples(run.id, [])
        assert count == 0

    def test_assign_to_nonexistent_run(self, sft_client, sample_feedback_facts):
        """Test assigning to nonexistent run raises NotFoundError."""
        with pytest.raises(NotFoundError, match="not found"):
            sft_client.assign_examples(99999, sample_feedback_facts[:1])

    def test_assign_to_non_pending_run(self, sft_client, sample_feedback_facts):
        """Test assigning to non-pending run raises ValidationError."""
        run = sft_client.create(adapter_name="status-test")
        sft_client.start(run.id)

        with pytest.raises(ValidationError, match="must be 'pending'"):
            sft_client.assign_examples(run.id, sample_feedback_facts[:1])

    def test_get_examples(self, sft_client, sample_feedback_facts):
        """Test getting pending examples assigned to a run."""
        run = sft_client.create(adapter_name="get-examples-test")
        sft_client.assign_examples(run.id, sample_feedback_facts[:3])

        retrieved = sft_client.get_examples(run.id)
        assert len(retrieved) == 3
        assert set(retrieved) == set(sample_feedback_facts[:3])

    def test_count_pending_examples(self, sft_client, sample_feedback_facts):
        """Test counting pending examples."""
        run = sft_client.create(adapter_name="count-test")
        sft_client.assign_examples(run.id, sample_feedback_facts[:3])

        assert sft_client.count_pending_examples(run.id) == 3
        assert sft_client.count_pending_examples() == 3


class TestSftLineageFiltering:
    """Test lineage-based example filtering for SFT."""

    def test_examples_filtered_by_lineage(self, sft_client, sample_feedback_facts):
        """Test that examples used in ancestor runs are filtered out."""
        # Create parent run and assign first 3 examples
        parent = sft_client.create(adapter_name="parent")
        sft_client.assign_examples(parent.id, sample_feedback_facts[:3])
        sft_client.start(parent.id)
        sft_client.complete(parent.id)

        # Create child run based on parent
        child = sft_client.create(adapter_name="child", based_on=parent.id)

        # Try to assign all 5 examples - should only get 2 (not used in parent)
        assigned = sft_client.assign_examples(child.id, sample_feedback_facts)
        assert assigned == 2

        # Verify the assigned examples are the last 2
        child_examples = sft_client.get_examples(child.id)
        assert len(child_examples) == 2
        assert set(child_examples) == set(sample_feedback_facts[3:])

    def test_no_lineage_allows_all_examples(self, sft_client, sample_feedback_facts):
        """Test that runs without lineage can use any examples."""
        run1 = sft_client.create(adapter_name="run1")
        sft_client.assign_examples(run1.id, sample_feedback_facts[:3])

        # Run2 has no lineage relationship to run1
        run2 = sft_client.create(adapter_name="run2", replace_stale=False)
        assert run2.id != run1.id

        # All examples available since no lineage
        assigned = sft_client.assign_examples(run2.id, sample_feedback_facts)
        assert assigned == 5


class TestSftExampleMovement:
    """Test example movement from pending to trained on completion."""

    def test_examples_move_to_trained_on_complete(self, sft_client, sample_feedback_facts):
        """Test that completing a run moves examples to trained table."""
        run = sft_client.create(adapter_name="complete-test")
        sft_client.assign_examples(run.id, sample_feedback_facts[:3])

        # Before completion - examples are pending
        assert len(sft_client.get_examples(run.id)) == 3
        assert len(sft_client.get_trained_examples(run.id)) == 0

        sft_client.start(run.id)
        sft_client.complete(run.id)

        # After completion - examples moved to trained
        assert len(sft_client.get_examples(run.id)) == 0
        assert len(sft_client.get_trained_examples(run.id)) == 3

    def test_examples_cleared_on_fail(self, sft_client, sample_feedback_facts):
        """Test that failing a run clears pending examples."""
        run = sft_client.create(adapter_name="fail-test")
        sft_client.assign_examples(run.id, sample_feedback_facts[:3])

        sft_client.start(run.id)
        sft_client.fail(run.id, "Test failure")

        # Examples cleared (freed for reuse)
        assert len(sft_client.get_examples(run.id)) == 0
        assert len(sft_client.get_trained_examples(run.id)) == 0


class TestSftLifecycle:
    """Test SFT training run lifecycle transitions."""

    def test_start_run(self, sft_client, clean_tables):
        """Test starting a run transitions to running."""
        run = sft_client.create()
        sft_client.start(run.id)

        updated = sft_client.get(run.id)
        assert updated.status == "running"
        assert updated.started_at is not None

    def test_complete_run(self, sft_client, clean_tables):
        """Test completing a run."""
        run = sft_client.create()
        sft_client.start(run.id)
        sft_client.complete(run.id, metrics={"loss": 0.01})

        updated = sft_client.get(run.id)
        assert updated.status == "completed"
        assert updated.completed_at is not None
        assert updated.metrics == {"loss": 0.01}

    def test_fail_run(self, sft_client, clean_tables):
        """Test failing a run."""
        run = sft_client.create()
        sft_client.start(run.id)
        sft_client.fail(run.id, "Out of memory")

        updated = sft_client.get(run.id)
        assert updated.status == "failed"
        assert updated.error_message == "Out of memory"

    def test_reset_run(self, sft_client, clean_tables):
        """Test resetting a running run back to pending."""
        run = sft_client.create()
        sft_client.start(run.id)
        sft_client.reset(run.id)

        updated = sft_client.get(run.id)
        assert updated.status == "pending"
        assert updated.started_at is None

    def test_invalid_transition_pending_to_completed(self, sft_client, clean_tables):
        """Test that pending cannot go directly to completed."""
        run = sft_client.create()
        with pytest.raises(ValidationError, match="Cannot transition"):
            sft_client.complete(run.id)


class TestSftContextIsolation:
    """Test context-based isolation for SFT."""

    def test_runs_isolated_by_context(self, logger, database, clean_tables):
        """Test that SFT runs are isolated by context_key."""
        client_a = SftClient(lg=logger, session_factory=database.session, context_key="sft-ctx-a")
        client_b = SftClient(lg=logger, session_factory=database.session, context_key="sft-ctx-b")

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


class TestSftResetAll:
    """Test reset_all functionality for SFT."""

    def test_reset_all(self, sft_client, sample_feedback_facts):
        """Test resetting all SFT runs in context."""
        run1 = sft_client.create(adapter_name="run-1", replace_stale=False)
        run2 = sft_client.create(adapter_name="run-2", replace_stale=False)
        sft_client.assign_examples(run1.id, sample_feedback_facts[:2])
        sft_client.assign_examples(run2.id, sample_feedback_facts[2:4])

        assert sft_client.count_pending_examples() == 4

        count = sft_client.reset_all()
        assert count == 2

        # Verify runs are soft-deleted
        assert len(sft_client.list_runs()) == 0
        # Verify pending examples are cleared
        assert sft_client.count_pending_examples() == 0

    def test_clear_pending_examples_for_context(self, sft_client, sample_feedback_facts):
        """Test clearing pending examples without deleting runs."""
        run = sft_client.create(adapter_name="test-run")
        sft_client.assign_examples(run.id, sample_feedback_facts[:3])

        assert sft_client.count_pending_examples() == 3

        cleared = sft_client.clear_pending_examples_for_context()
        assert cleared == 3

        # Run still exists
        assert sft_client.get(run.id) is not None
        # But examples are gone
        assert sft_client.count_pending_examples() == 0
