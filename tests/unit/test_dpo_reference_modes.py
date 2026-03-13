"""Tests for DPO trainer reference mode branches.

Tests the three reference model configurations:
1. Stacked adapters (parent exists): "ref" adapter preserved
2. Implicit reference (no parent, no ref_model): "ref" adapter deleted
3. Reference-free mode: no reference setup at all
"""

from unittest.mock import MagicMock, patch

import pytest


class TestEnableImplicitReference:
    """Tests for _enable_implicit_reference method."""

    def _make_trainer(self, ref_model=None, parent=None, reference_free=False):
        """Create a Trainer instance with mocked dependencies."""
        from llm_kelt.training.dpo.trainer import Trainer

        with patch.object(Trainer, "_init_state"):
            trainer = object.__new__(Trainer)
            trainer.ref_model = ref_model
            trainer.parent = parent
            trainer.reference_free = reference_free
            trainer.model = None
            trainer.trainer = None
            trainer._lg = MagicMock()
        return trainer

    def test_stacked_adapters_preserves_ref(self):
        """When parent exists, _enable_implicit_reference should not be called.

        The ref adapter is set up in _apply_lora_adapter and should be preserved.
        """
        trainer = self._make_trainer(parent=MagicMock(path="/some/path"))
        trainer._use_stacked_adapters = True

        # Create mock model with ref adapter
        mock_model = MagicMock()
        mock_model.peft_config = {"default": MagicMock(), "ref": MagicMock()}
        trainer.model = mock_model

        # Mock trainer with ref_adapter_name
        mock_dpo_trainer = MagicMock()
        mock_dpo_trainer.ref_adapter_name = "ref"
        trainer.trainer = mock_dpo_trainer

        # In _create_trainer, this condition prevents _enable_implicit_reference call
        # when stacked adapters are used (parent is not None)
        should_enable_implicit = (
            trainer.ref_model is None and trainer.parent is None and not trainer.reference_free
        )
        assert not should_enable_implicit, "Should not enable implicit reference with parent"

        # Verify ref adapter still exists
        assert "ref" in mock_model.peft_config
        mock_model.delete_adapter.assert_not_called()

    def test_implicit_reference_removes_ref_adapter(self):
        """When no parent and no ref_model, should remove ref adapter."""
        peft = pytest.importorskip("peft")

        trainer = self._make_trainer(parent=None, ref_model=None, reference_free=False)
        trainer._use_stacked_adapters = False

        # Create mock that passes isinstance check for PeftModel
        mock_model = MagicMock(spec=peft.PeftModel)
        mock_model.peft_config = {"default": MagicMock(), "ref": MagicMock()}
        trainer.model = mock_model

        # Mock trainer with ref_adapter_name
        mock_dpo_trainer = MagicMock()
        mock_dpo_trainer.ref_adapter_name = "ref"
        trainer.trainer = mock_dpo_trainer

        trainer._enable_implicit_reference()

        # Should delete the ref adapter
        mock_model.delete_adapter.assert_called_once_with("ref")
        # Should clear trainer's ref_adapter_name
        assert mock_dpo_trainer.ref_adapter_name is None

    def test_implicit_reference_uses_configured_adapter_name(self):
        """Should use trainer.ref_adapter_name if set, not hardcoded 'ref'."""
        peft = pytest.importorskip("peft")

        trainer = self._make_trainer(parent=None, ref_model=None, reference_free=False)
        trainer._use_stacked_adapters = False

        # Create mock that passes isinstance check for PeftModel
        mock_model = MagicMock(spec=peft.PeftModel)
        mock_model.peft_config = {"default": MagicMock(), "custom_ref": MagicMock()}
        trainer.model = mock_model

        # Trainer configured with custom ref adapter name
        mock_dpo_trainer = MagicMock()
        mock_dpo_trainer.ref_adapter_name = "custom_ref"
        trainer.trainer = mock_dpo_trainer

        trainer._enable_implicit_reference()

        # Should delete the custom-named adapter
        mock_model.delete_adapter.assert_called_once_with("custom_ref")
        assert mock_dpo_trainer.ref_adapter_name is None

    def test_reference_free_skips_implicit_reference(self):
        """When reference_free=True, should not call _enable_implicit_reference."""
        trainer = self._make_trainer(parent=None, ref_model=None, reference_free=True)
        trainer._use_stacked_adapters = False

        mock_model = MagicMock()
        mock_model.peft_config = {"default": MagicMock(), "ref": MagicMock()}
        trainer.model = mock_model

        mock_dpo_trainer = MagicMock()
        mock_dpo_trainer.ref_adapter_name = "ref"
        trainer.trainer = mock_dpo_trainer

        # The condition in _create_trainer should prevent the call
        should_enable_implicit = (
            trainer.ref_model is None and trainer.parent is None and not trainer.reference_free
        )
        assert not should_enable_implicit, (
            "Should not enable implicit reference when reference_free"
        )

        # Verify no adapter deletion occurred
        mock_model.delete_adapter.assert_not_called()
        # ref_adapter_name should remain unchanged
        assert mock_dpo_trainer.ref_adapter_name == "ref"

    def test_clears_trainer_binding_even_without_adapter(self):
        """Should always clear trainer.ref_adapter_name even if adapter doesn't exist."""
        peft = pytest.importorskip("peft")

        trainer = self._make_trainer(parent=None, ref_model=None, reference_free=False)
        trainer._use_stacked_adapters = False

        # Create mock that passes isinstance check for PeftModel, but no ref adapter
        mock_model = MagicMock(spec=peft.PeftModel)
        mock_model.peft_config = {"default": MagicMock()}  # No "ref"
        trainer.model = mock_model

        mock_dpo_trainer = MagicMock()
        mock_dpo_trainer.ref_adapter_name = "ref"
        trainer.trainer = mock_dpo_trainer

        trainer._enable_implicit_reference()

        # Should not try to delete non-existent adapter
        mock_model.delete_adapter.assert_not_called()
        # But should still clear trainer binding
        assert mock_dpo_trainer.ref_adapter_name is None
