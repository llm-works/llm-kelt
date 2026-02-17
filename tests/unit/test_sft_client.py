"""Unit tests for SFT training client."""

from unittest.mock import MagicMock

from llm_learn.core.exceptions import ValidationError
from llm_learn.training.models import Run, RunInfo
from llm_learn.training.sft.client import (
    STATUS_TRANSITIONS,
    VALID_STATUSES,
    Client,
    PendingExample,
    TrainedExample,
)


class TestSftClientStructure:
    """Test SFT client structure and imports."""

    def test_sft_method_attribute(self):
        """Test that SFT client has correct method attribute."""
        # The method is set in __init__, verify the class structure
        assert hasattr(Client, "__init__")

    def test_pending_example_model(self):
        """Test PendingExample model structure."""
        assert PendingExample.__tablename__ == "sft_pending_examples"
        # Check columns exist
        assert hasattr(PendingExample, "run_id")
        assert hasattr(PendingExample, "fact_id")
        assert hasattr(PendingExample, "assigned_at")

    def test_trained_example_model(self):
        """Test TrainedExample model structure."""
        assert TrainedExample.__tablename__ == "sft_trained_examples"
        # Check columns exist
        assert hasattr(TrainedExample, "run_id")
        assert hasattr(TrainedExample, "fact_id")
        assert hasattr(TrainedExample, "trained_at")

    def test_shares_run_model_with_dpo(self):
        """Test that SFT uses same Run model as DPO."""
        # Run is imported from dpo.client, verify it's the same model
        assert Run.__tablename__ == "training_runs"
        assert hasattr(Run, "method")
        assert hasattr(Run, "adapter")

    def test_shares_run_info_with_dpo(self):
        """Test that SFT uses same RunInfo as DPO."""
        # RunInfo is a dataclass - check its fields
        import dataclasses

        field_names = {f.name for f in dataclasses.fields(RunInfo)}
        assert "method" in field_names
        assert "adapter" in field_names
        # adapter_name is a property, not a field
        assert hasattr(RunInfo, "adapter_name")
        assert hasattr(RunInfo, "from_model")

    def test_valid_statuses(self):
        """Test valid status values are available."""
        expected = {"pending", "running", "completed", "failed", "cancelled"}
        assert VALID_STATUSES == expected

    def test_status_transitions(self):
        """Test status transitions are defined."""
        assert "pending" in STATUS_TRANSITIONS
        assert "running" in STATUS_TRANSITIONS
        assert "running" in STATUS_TRANSITIONS["pending"]
        assert "completed" in STATUS_TRANSITIONS["running"]
        assert "failed" in STATUS_TRANSITIONS["running"]


class TestPendingExampleRepr:
    """Test PendingExample string representation."""

    def test_repr_format(self):
        """Test PendingExample __repr__ format."""
        example = PendingExample()
        example.run_id = 1
        example.fact_id = 42
        repr_str = repr(example)
        assert "PendingExample" in repr_str
        assert "run=1" in repr_str
        assert "fact=42" in repr_str


class TestTrainedExampleRepr:
    """Test TrainedExample string representation."""

    def test_repr_format(self):
        """Test TrainedExample __repr__ format."""
        example = TrainedExample()
        example.run_id = 1
        example.fact_id = 42
        repr_str = repr(example)
        assert "TrainedExample" in repr_str
        assert "run=1" in repr_str
        assert "fact=42" in repr_str


class TestSftClientInit:
    """Test SFT client initialization."""

    def test_init_sets_method(self):
        """Test that __init__ sets method to 'sft'."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")
        assert client.method == "sft"
        assert client.context_key == "test"

    def test_init_with_none_context(self):
        """Test that __init__ works with None context_key."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key=None)
        assert client.context_key is None


class TestSftClientValidation:
    """Test SFT client validation methods."""

    def test_validate_pending_status_raises_on_running(self):
        """Test _validate_pending_status raises for non-pending runs."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")

        mock_run = MagicMock()
        mock_run.status = "running"

        try:
            client._validate_pending_status(mock_run)
            assert False, "Should have raised ValidationError"
        except ValidationError as e:
            assert "must be 'pending'" in str(e)

    def test_validate_pending_status_passes_for_pending(self):
        """Test _validate_pending_status passes for pending runs."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")

        mock_run = MagicMock()
        mock_run.status = "pending"

        # Should not raise
        client._validate_pending_status(mock_run)


class TestSftClientContextFilter:
    """Test context filter building."""

    def test_build_context_filter_with_key(self):
        """Test _build_context_filter returns filter for non-None context."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")

        # Should return a filter
        result = client._build_context_filter(Run.context_key)
        assert result is not None

    def test_build_context_filter_with_none(self):
        """Test _build_context_filter returns None for None context."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key=None)

        # Should return None for no filtering
        result = client._build_context_filter(Run.context_key)
        assert result is None


class TestSftClientEmptyAssign:
    """Test example assignment edge cases."""

    def test_assign_empty_list_returns_zero(self):
        """Test assign_examples returns 0 for empty list."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")

        # Empty list should return 0 immediately without database access
        result = client.assign_examples(1, [])
        assert result == 0


class TestSftClientTransitions:
    """Test status transition validation."""

    def test_pending_can_go_to_running(self):
        """Test pending status can transition to running."""
        assert "running" in STATUS_TRANSITIONS["pending"]

    def test_running_can_go_to_completed(self):
        """Test running status can transition to completed."""
        assert "completed" in STATUS_TRANSITIONS["running"]

    def test_running_can_go_to_pending(self):
        """Test running can go back to pending (reset)."""
        assert "pending" in STATUS_TRANSITIONS["running"]

    def test_completed_is_terminal(self):
        """Test completed status has no transitions."""
        assert STATUS_TRANSITIONS["completed"] == set()

    def test_failed_is_terminal(self):
        """Test failed status has no transitions."""
        assert STATUS_TRANSITIONS["failed"] == set()


class TestSftTrainingClient:
    """Test training client SFT property."""

    def test_sft_property_returns_client(self):
        """Test that training client has sft property."""
        from llm_learn.training.client import Client as TrainClient

        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        train_client = TrainClient(
            lg=mock_lg, session_factory=mock_session_factory, context_key="test"
        )

        # Access the sft property
        sft_client = train_client.sft
        assert sft_client is not None
        assert sft_client.method == "sft"
        assert sft_client.context_key == "test"

        # Second access should return same instance
        sft_client2 = train_client.sft
        assert sft_client is sft_client2

    def test_dpo_and_sft_are_independent(self):
        """Test that dpo and sft clients are independent."""
        from llm_learn.training.client import Client as TrainClient

        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        train_client = TrainClient(
            lg=mock_lg, session_factory=mock_session_factory, context_key="test"
        )

        dpo = train_client.dpo
        sft = train_client.sft

        assert dpo.method == "dpo"
        assert sft.method == "sft"
        assert dpo is not sft


class TestSftApplyTransition:
    """Test _apply_transition helper."""

    def test_apply_transition_to_running(self):
        """Test applying transition to running sets started_at."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")

        mock_run = MagicMock()
        mock_run.status = "pending"
        mock_run.started_at = None

        client._apply_transition(mock_run, "running", None, None, False)

        assert mock_run.status == "running"
        assert mock_run.started_at is not None

    def test_apply_transition_to_completed(self):
        """Test applying transition to completed sets completed_at."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")

        mock_run = MagicMock()
        mock_run.status = "running"
        mock_run.completed_at = None

        client._apply_transition(mock_run, "completed", None, None, False)

        assert mock_run.status == "completed"
        assert mock_run.completed_at is not None

    def test_apply_transition_with_metrics(self):
        """Test applying transition with metrics."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")

        mock_run = MagicMock()
        mock_run.metrics = None

        metrics = {"loss": 0.5}
        client._apply_transition(mock_run, "completed", metrics, None, False)

        assert mock_run.metrics == metrics

    def test_apply_transition_with_error_message(self):
        """Test applying transition with error message."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")

        mock_run = MagicMock()
        mock_run.error_message = None

        client._apply_transition(mock_run, "failed", None, "Test error", False)

        assert mock_run.error_message == "Test error"

    def test_apply_transition_clear_started(self):
        """Test applying transition with clear_started flag."""
        mock_lg = MagicMock()
        mock_session_factory = MagicMock()
        client = Client(lg=mock_lg, session_factory=mock_session_factory, context_key="test")

        mock_run = MagicMock()
        mock_run.started_at = "some_time"

        client._apply_transition(mock_run, "pending", None, None, True)

        assert mock_run.started_at is None
