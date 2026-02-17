"""Unit tests for training models module."""

import pytest

from llm_learn.training.models import (
    STATUS_TRANSITIONS,
    VALID_STATUSES,
    RunInfo,
    _is_pattern,
)


class TestIsPattern:
    """Test glob pattern detection."""

    def test_none_is_not_pattern(self):
        assert _is_pattern(None) is False

    def test_plain_string_is_not_pattern(self):
        assert _is_pattern("context-a") is False
        assert _is_pattern("tenant:a:agent1") is False

    def test_asterisk_is_pattern(self):
        assert _is_pattern("tenant:*") is True
        assert _is_pattern("*") is True
        assert _is_pattern("tenant:a:*") is True

    def test_question_mark_is_pattern(self):
        assert _is_pattern("tenant:?") is True
        assert _is_pattern("agent?") is True


class TestStatusConstants:
    """Test status constants are valid."""

    def test_valid_statuses(self):
        expected = {"pending", "running", "completed", "failed", "cancelled"}
        assert VALID_STATUSES == expected

    def test_pending_transitions(self):
        assert STATUS_TRANSITIONS["pending"] == {"running", "failed", "cancelled"}

    def test_running_transitions(self):
        assert STATUS_TRANSITIONS["running"] == {"pending", "completed", "failed"}

    def test_terminal_states_no_transitions(self):
        assert STATUS_TRANSITIONS["completed"] == set()
        assert STATUS_TRANSITIONS["failed"] == set()
        assert STATUS_TRANSITIONS["cancelled"] == set()


class TestRunInfo:
    """Test RunInfo dataclass."""

    def test_adapter_name_with_adapter(self):
        info = RunInfo(
            id=1,
            method="dpo",
            context_key="test",
            adapter={"name": "test-adapter"},
            based_on=None,
            status="pending",
            config=None,
            metrics=None,
            error_message=None,
            system_status=None,
            created_at=None,
            started_at=None,
            completed_at=None,
        )
        assert info.adapter_name == "test-adapter"

    def test_adapter_name_without_adapter(self):
        info = RunInfo(
            id=1,
            method="dpo",
            context_key="test",
            adapter=None,
            based_on=None,
            status="pending",
            config=None,
            metrics=None,
            error_message=None,
            system_status=None,
            created_at=None,
            started_at=None,
            completed_at=None,
        )
        assert info.adapter_name is None

    def test_is_deleted_false_when_no_system_status(self):
        info = RunInfo(
            id=1,
            method="dpo",
            context_key="test",
            adapter=None,
            based_on=None,
            status="pending",
            config=None,
            metrics=None,
            error_message=None,
            system_status=None,
            created_at=None,
            started_at=None,
            completed_at=None,
        )
        assert info.is_deleted is False

    def test_is_deleted_true_when_marked(self):
        info = RunInfo(
            id=1,
            method="dpo",
            context_key="test",
            adapter=None,
            based_on=None,
            status="completed",
            config=None,
            metrics=None,
            error_message=None,
            system_status={"deleted": True},
            created_at=None,
            started_at=None,
            completed_at=None,
        )
        assert info.is_deleted is True

    def test_is_deleted_false_when_not_marked(self):
        info = RunInfo(
            id=1,
            method="dpo",
            context_key="test",
            adapter=None,
            based_on=None,
            status="completed",
            config=None,
            metrics=None,
            error_message=None,
            system_status={"some_other_key": "value"},
            created_at=None,
            started_at=None,
            completed_at=None,
        )
        assert info.is_deleted is False


class TestTrainingModuleLazyImports:
    """Test lazy imports in training module."""

    def test_lazy_import_train_lora(self):
        """Test that train_lora can be imported lazily."""
        from llm_learn.training import train_lora

        assert callable(train_lora)

    def test_lazy_import_train_dpo(self):
        """Test that train_dpo can be imported lazily."""
        from llm_learn.training import train_dpo

        assert callable(train_dpo)

    def test_invalid_attribute_raises(self):
        """Test that invalid attribute raises AttributeError."""
        import llm_learn.training as training

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = training.nonexistent_function
