# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""E2E tests for LoRA and DPO training.

These tests verify that training produces valid adapters with expected outputs.
Requires training dependencies: torch, transformers, peft, trl.
Requires local model: qwen2.5-0.5b-instruct in ~/ops/models/huggingface/
"""

from pathlib import Path

import pytest


@pytest.mark.training
class TestLoraTraining:
    """E2E tests for LoRA/SFT training."""

    def test_train_lora_produces_valid_adapter(
        self,
        logger,
        sft_training_data: Path,
        local_model_path: Path,
        fast_lora_config,
        fast_training_config,
        tmp_path: Path,
    ):
        """Train LoRA adapter with a validation split and verify it produces valid outputs."""
        from appinfra import DotDict
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from llm_kelt.training import train_lora

        output_dir = tmp_path / "lora_output"

        # Train
        result = train_lora(
            lg=logger,
            data_path=sft_training_data,
            output_dir=output_dir,
            base_model=str(local_model_path),
            lora_config=fast_lora_config,
            training_config=DotDict({**fast_training_config, "eval_split": 0.2}),
            quantize=False,  # Faster without quantization for small model
        )

        # Verify result structure
        assert Path(result.adapter.path).exists(), "Adapter path should exist"
        assert result.method == "sft"
        assert result.base_model == str(local_model_path)
        assert result.samples_trained > 0

        # Verify adapter files exist
        adapter_config = Path(result.adapter.path) / "adapter_config.json"
        adapter_weights = Path(result.adapter.path) / "adapter_model.safetensors"
        assert adapter_config.exists(), "adapter_config.json should exist"
        assert adapter_weights.exists(), "adapter_model.safetensors should exist"

        # Verify metrics
        assert "train_loss" in result.metrics or result.metrics == {}, "Should have metrics"
        assert "eval_loss" in result.metrics, "Should have eval_loss with eval_split > 0"

        # Verify adapter can be loaded
        base_model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),
            device_map="auto",
            trust_remote_code=True,
        )
        loaded_model = PeftModel.from_pretrained(base_model, result.adapter.path)
        assert loaded_model is not None, "Should be able to load adapter"

        # Verify training config was stored
        assert "lora" in result.config
        assert result.config["lora"]["r"] == fast_lora_config.r


def _tokenized_train_dataset(logger, data_path: Path, model_path: Path, training_config, out: Path):
    """Build the SFTTrainer for a dataset (no training) and return (train_dataset, data_collator)."""
    from llm_kelt.training.lora.trainer import Trainer

    trainer = Trainer(
        lg=logger,
        data_path=data_path,
        output_dir=out,
        base_model=str(model_path),
        training_config=training_config,
        quantize=False,
    )
    trainer._load_data()
    trainer._load_model()
    trainer._create_trainer()
    return trainer.trainer.train_dataset, trainer.trainer.data_collator


def _write_records(path: Path, records: list[dict]) -> Path:
    import json

    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


@pytest.mark.training
class TestSftDataPreparation:
    """Token-level checks of SFT data preparation against a real tokenizer and TRL."""

    RECORDS = [
        {"instruction": "Summarize the text.", "input": "Foo bar baz.", "output": "Foo."},
        {"instruction": "Answer briefly.", "output": '{"label": 3, "why": "bar"}'},
    ]

    def test_completion_only_loss_masks_prompt(
        self, logger, local_model_path: Path, fast_training_config, tmp_path: Path
    ):
        """With completion_only_loss, only the assistant turn contributes to the loss."""
        from transformers import AutoTokenizer

        data_path = _write_records(tmp_path / "data.jsonl", self.RECORDS)
        config = {**fast_training_config, "completion_only_loss": True}
        dataset, collator = _tokenized_train_dataset(
            logger, data_path, local_model_path, config, tmp_path / "out"
        )
        tokenizer = AutoTokenizer.from_pretrained(str(local_model_path))

        for row, record in zip(dataset, self.RECORDS, strict=True):
            labels = collator([row])["labels"][0].tolist()
            trained = [
                t for t, label in zip(row["input_ids"], labels, strict=True) if label != -100
            ]
            assert tokenizer.decode(trained).startswith(record["output"])
            assert record["instruction"] not in tokenizer.decode(trained)


@pytest.mark.training
class TestPromptTuning:
    """E2E tests for Prompt Tuning training."""

    def test_train_prompt_produces_valid_adapter(
        self,
        logger,
        sft_training_data: Path,
        local_model_path: Path,
        fast_prompt_config,
        fast_training_config,
        tmp_path: Path,
    ):
        """Train prompt tuning adapter and verify it produces valid outputs."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from llm_kelt.training.prompt import Trainer

        output_dir = tmp_path / "prompt_output"

        trainer = Trainer(
            lg=logger,
            data_path=sft_training_data,
            output_dir=output_dir,
            base_model=str(local_model_path),
            prompt_config=fast_prompt_config,
            training_config=fast_training_config,
            quantize=False,
        )
        result = trainer.train()

        # Verify result structure
        assert Path(result.adapter.path).exists(), "Adapter path should exist"
        assert result.method == "prompt"
        assert result.base_model == str(local_model_path)
        assert result.samples_trained > 0

        # Verify adapter files exist
        adapter_config = Path(result.adapter.path) / "adapter_config.json"
        assert adapter_config.exists(), "adapter_config.json should exist"

        # Verify config was stored
        assert "prompt_tuning" in result.config
        assert (
            result.config["prompt_tuning"]["num_virtual_tokens"]
            == fast_prompt_config.num_virtual_tokens
        )

        # Verify adapter can be loaded
        base_model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),
            device_map="auto",
            trust_remote_code=True,
        )
        loaded_model = PeftModel.from_pretrained(base_model, result.adapter.path)
        assert loaded_model is not None, "Should be able to load adapter"

        # Verify it's a prompt tuning adapter with correct config
        peft_config = loaded_model.peft_config["default"]
        assert peft_config.peft_type.value == "PROMPT_TUNING"
        assert peft_config.num_virtual_tokens == fast_prompt_config.num_virtual_tokens


@pytest.mark.training
class TestDpoTraining:
    """E2E tests for DPO training."""

    def test_train_dpo_produces_valid_adapter(
        self,
        logger,
        dpo_training_data: Path,
        local_model_path: Path,
        fast_lora_config,
        fast_training_config,
        tmp_path: Path,
    ):
        """Train DPO adapter and verify it produces valid outputs."""
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        from llm_kelt.training import train_dpo

        output_dir = tmp_path / "dpo_output"

        result = train_dpo(
            lg=logger,
            data_path=dpo_training_data,
            output_dir=output_dir,
            base_model=str(local_model_path),
            lora_config=fast_lora_config,
            training_config=fast_training_config,
            quantize=False,
        )

        # Verify result structure
        assert Path(result.adapter.path).exists(), "Adapter path should exist"
        assert result.method == "dpo"
        assert result.base_model == str(local_model_path)
        assert result.samples_trained > 0

        # Verify adapter files exist
        adapter_config = Path(result.adapter.path) / "adapter_config.json"
        adapter_weights = Path(result.adapter.path) / "adapter_model.safetensors"
        assert adapter_config.exists(), "adapter_config.json should exist"
        assert adapter_weights.exists(), "adapter_model.safetensors should exist"

        # Verify DPO config was stored
        assert "dpo" in result.config
        assert "beta" in result.config["dpo"]

        # Verify adapter can be loaded
        base_model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),
            device_map="auto",
            trust_remote_code=True,
        )
        loaded_model = PeftModel.from_pretrained(base_model, result.adapter.path)
        assert loaded_model is not None, "Should be able to load adapter"
