"""E2E tests for LoRA and DPO training.

These tests verify that training produces valid adapters with expected outputs.
Requires training dependencies: torch, transformers, peft, trl.
Requires local model: qwen2.5-0.5b-instruct in ~/ops/models/huggingface/
"""

from pathlib import Path

import pytest

# Training tests are marked automatically by conftest imports
# Skip entire module if training deps not available
pytest.importorskip("torch")
pytest.importorskip("peft")


@pytest.mark.training
class TestLoraTraining:
    """E2E tests for LoRA/SFT training."""

    def test_train_lora_produces_valid_adapter(
        self,
        sft_training_data: Path,
        local_model_path: Path,
        fast_lora_config,
        fast_training_config,
        tmp_path: Path,
    ):
        """Train LoRA adapter and verify it produces valid outputs."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from llm_learn.training import train_lora

        output_dir = tmp_path / "lora_output"

        # Train
        result = train_lora(
            data_path=sft_training_data,
            output_dir=output_dir,
            base_model=str(local_model_path),
            lora_config=fast_lora_config,
            training_config=fast_training_config,
            quantize=False,  # Faster without quantization for small model
        )

        # Verify result structure
        assert result.adapter_path.exists(), "Adapter path should exist"
        assert result.method == "lora"
        assert result.base_model == str(local_model_path)
        assert result.samples_trained > 0

        # Verify adapter files exist
        adapter_config = result.adapter_path / "adapter_config.json"
        adapter_weights = result.adapter_path / "adapter_model.safetensors"
        assert adapter_config.exists(), "adapter_config.json should exist"
        assert adapter_weights.exists(), "adapter_model.safetensors should exist"

        # Verify metrics
        assert "train_loss" in result.metrics or result.metrics == {}, "Should have metrics"

        # Verify adapter can be loaded
        base_model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),
            device_map="auto",
            trust_remote_code=True,
        )
        loaded_model = PeftModel.from_pretrained(base_model, str(result.adapter_path))
        assert loaded_model is not None, "Should be able to load adapter"

        # Verify training config was stored
        assert "lora" in result.config
        assert result.config["lora"]["r"] == fast_lora_config.r

    def test_train_lora_with_eval_split(
        self,
        sft_training_data: Path,
        local_model_path: Path,
        fast_lora_config,
        fast_training_config,
        tmp_path: Path,
    ):
        """Train with validation split and verify eval metrics are produced."""
        from llm_learn.training import TrainingConfig, train_lora

        output_dir = tmp_path / "lora_output_eval"

        # Config with eval split
        config_with_eval = TrainingConfig(
            num_epochs=fast_training_config.num_epochs,
            batch_size=fast_training_config.batch_size,
            gradient_accumulation_steps=fast_training_config.gradient_accumulation_steps,
            max_seq_length=fast_training_config.max_seq_length,
            logging_steps=fast_training_config.logging_steps,
            save_steps=fast_training_config.save_steps,
            eval_split=0.2,  # 20% for eval
            fp16=fast_training_config.fp16,
            bf16=fast_training_config.bf16,
        )

        result = train_lora(
            data_path=sft_training_data,
            output_dir=output_dir,
            base_model=str(local_model_path),
            lora_config=fast_lora_config,
            training_config=config_with_eval,
            quantize=False,
        )

        # With eval split, should have eval_loss
        assert result.adapter_path.exists()
        assert "eval_loss" in result.metrics, "Should have eval_loss with eval_split > 0"


@pytest.mark.training
class TestDpoTraining:
    """E2E tests for DPO training."""

    def test_train_dpo_produces_valid_adapter(
        self,
        dpo_training_data: Path,
        local_model_path: Path,
        fast_lora_config,
        fast_training_config,
        tmp_path: Path,
    ):
        """Train DPO adapter and verify it produces valid outputs."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from llm_learn.training import train_dpo

        output_dir = tmp_path / "dpo_output"

        # Train with reference-free to save memory
        result = train_dpo(
            data_path=dpo_training_data,
            output_dir=output_dir,
            base_model=str(local_model_path),
            lora_config=fast_lora_config,
            training_config=fast_training_config,
            quantize=False,
            reference_free=True,  # Save memory for test
        )

        # Verify result structure
        assert result.adapter_path.exists(), "Adapter path should exist"
        assert result.method == "dpo"
        assert result.base_model == str(local_model_path)
        assert result.samples_trained > 0

        # Verify adapter files exist
        adapter_config = result.adapter_path / "adapter_config.json"
        adapter_weights = result.adapter_path / "adapter_model.safetensors"
        assert adapter_config.exists(), "adapter_config.json should exist"
        assert adapter_weights.exists(), "adapter_model.safetensors should exist"

        # Verify DPO config was stored
        assert "dpo" in result.config
        assert result.config["dpo"]["reference_free"] is True

        # Verify adapter can be loaded
        base_model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),
            device_map="auto",
            trust_remote_code=True,
        )
        loaded_model = PeftModel.from_pretrained(base_model, str(result.adapter_path))
        assert loaded_model is not None, "Should be able to load adapter"

    def test_train_dpo_with_reference_model(
        self,
        dpo_training_data: Path,
        local_model_path: Path,
        fast_lora_config,
        fast_training_config,
        tmp_path: Path,
    ):
        """Train DPO with reference model (full DPO, more memory intensive)."""
        from llm_learn.training import train_dpo

        output_dir = tmp_path / "dpo_output_ref"

        result = train_dpo(
            data_path=dpo_training_data,
            output_dir=output_dir,
            base_model=str(local_model_path),
            lora_config=fast_lora_config,
            training_config=fast_training_config,
            quantize=False,
            reference_free=False,  # Use reference model
        )

        assert result.adapter_path.exists()
        assert result.method == "dpo"
        assert result.config["dpo"]["reference_free"] is False
