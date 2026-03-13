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

    def test_stacked_adapters_skips_implicit_reference(self):
        """When parent exists, _create_trainer should not call _enable_implicit_reference."""
        trainer = self._make_trainer(parent=MagicMock(path="/some/path"))
        trainer.model = MagicMock()
        trainer.train_dataset = MagicMock()
        trainer.eval_dataset = None
        trainer.tokenizer = MagicMock()

        with (
            patch("trl.DPOTrainer"),
            patch.object(trainer, "_create_training_args", return_value=MagicMock()),
            patch.object(trainer, "_enable_implicit_reference") as mock_enable,
        ):
            trainer._create_trainer()
            mock_enable.assert_not_called()

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
        """When reference_free=True, _create_trainer should not call _enable_implicit_reference."""
        trainer = self._make_trainer(parent=None, ref_model=None, reference_free=True)
        trainer.model = MagicMock()
        trainer.train_dataset = MagicMock()
        trainer.eval_dataset = None
        trainer.tokenizer = MagicMock()

        with (
            patch("trl.DPOTrainer"),
            patch.object(trainer, "_create_training_args", return_value=MagicMock()),
            patch.object(trainer, "_enable_implicit_reference") as mock_enable,
        ):
            trainer._create_trainer()
            mock_enable.assert_not_called()

    def test_implicit_reference_called_when_no_parent_no_ref_model(self):
        """When no parent and no ref_model, _create_trainer should call _enable_implicit_reference."""
        trainer = self._make_trainer(parent=None, ref_model=None, reference_free=False)
        trainer.model = MagicMock()
        trainer.train_dataset = MagicMock()
        trainer.eval_dataset = None
        trainer.tokenizer = MagicMock()

        with (
            patch("trl.DPOTrainer"),
            patch.object(trainer, "_create_training_args", return_value=MagicMock()),
            patch.object(trainer, "_enable_implicit_reference") as mock_enable,
        ):
            trainer._create_trainer()
            mock_enable.assert_called_once()

    def test_explicit_ref_model_skips_implicit_reference(self):
        """When ref_model is provided, _create_trainer should not call _enable_implicit_reference."""
        trainer = self._make_trainer(parent=None, ref_model=MagicMock(), reference_free=False)
        trainer.model = MagicMock()
        trainer.train_dataset = MagicMock()
        trainer.eval_dataset = None
        trainer.tokenizer = MagicMock()

        with (
            patch("trl.DPOTrainer"),
            patch.object(trainer, "_create_training_args", return_value=MagicMock()),
            patch.object(trainer, "_enable_implicit_reference") as mock_enable,
        ):
            trainer._create_trainer()
            mock_enable.assert_not_called()

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
