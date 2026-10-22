# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for training configuration."""

import json
from datetime import datetime

import pytest
from appinfra import DotDict

from llm_kelt.training import TRAINING_DEFAULTS, RunResult
from llm_kelt.training.lora import Config as LoraConfig
from llm_kelt.training.model import _apply_precision_settings, compute_warmup_steps


class TestLoraConfig:
    """Test LoraConfig validation and defaults."""

    def test_default_values(self):
        """Test that default values are sensible."""
        config = LoraConfig()

        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.05
        assert config.bias == "none"
        assert config.task_type == "CAUSAL_LM"
        # Targets come from the model family once the model is loaded
        assert config.target_modules is None
        assert config.exclude_modules is None

    def test_custom_values(self):
        """Test creating config with custom values."""
        config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj"],
            bias="lora_only",
        )

        assert config.r == 32
        assert config.lora_alpha == 64
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "k_proj"]
        assert config.bias == "lora_only"

    def test_invalid_rank(self):
        """Test that invalid rank raises ValueError."""
        with pytest.raises(ValueError, match="rank must be positive"):
            LoraConfig(r=0)

        with pytest.raises(ValueError, match="rank must be positive"):
            LoraConfig(r=-1)

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be positive"):
            LoraConfig(lora_alpha=0)

    def test_invalid_dropout(self):
        """Test that invalid dropout raises ValueError."""
        with pytest.raises(ValueError, match="dropout must be in"):
            LoraConfig(lora_dropout=-0.1)

        with pytest.raises(ValueError, match="dropout must be in"):
            LoraConfig(lora_dropout=1.0)

    def test_empty_target_modules(self):
        """Test that empty target_modules raises ValueError."""
        with pytest.raises(ValueError, match="target_modules cannot be empty"):
            LoraConfig(target_modules=[])


class TestLoraFamilyDefaults:
    """Model-family LoRA targets resolved from the loaded model's model_type."""

    def test_qwen2_keeps_attention_and_mlp_targets(self):
        config = LoraConfig().resolve("qwen2")
        assert config.target_modules == [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        assert config.exclude_modules is None

    @pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_text"])
    def test_qwen3_5_adds_linear_attention_and_excludes_vision_and_mtp(self, model_type: str):
        config = LoraConfig().resolve(model_type)
        assert {"in_proj_qkv", "in_proj_z", "out_proj", "q_proj", "down_proj"} <= set(
            config.target_modules
        )
        assert "in_proj_a" not in config.target_modules

        import re

        exclude = re.compile(config.exclude_modules)
        assert exclude.fullmatch("model.visual.blocks.0.attn.qkv")
        assert exclude.fullmatch("mtp.layers.0.mlp.down_proj")
        assert not exclude.fullmatch("model.language_model.layers.0.linear_attn.in_proj_qkv")
        assert not exclude.fullmatch("model.layers.3.self_attn.q_proj")

    def test_unknown_family_falls_back_to_qwen2_targets(self):
        assert (
            LoraConfig().resolve("llama").target_modules
            == LoraConfig().resolve("qwen2").target_modules
        )

    def test_explicit_settings_win(self):
        config = LoraConfig(target_modules=["q_proj"], exclude_modules="none").resolve("qwen3_5")
        assert config.target_modules == ["q_proj"]
        assert config.exclude_modules == "none"

    def test_unresolved_config_cannot_build_peft_config(self):
        pytest.importorskip("peft")
        with pytest.raises(ValueError, match="resolve"):
            LoraConfig().to_peft_config()

    def test_invalid_bias(self):
        """Test that invalid bias raises ValueError."""
        with pytest.raises(ValueError, match="bias must be"):
            LoraConfig(bias="invalid")

    def test_invalid_task_type(self):
        """Test that invalid task_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid task_type"):
            LoraConfig(task_type="INVALID")

    def test_valid_task_types(self):
        """Test all valid task types."""
        for task_type in ["CAUSAL_LM", "SEQ_CLS", "SEQ_2_SEQ_LM", "TOKEN_CLS"]:
            config = LoraConfig(task_type=task_type)
            assert config.task_type == task_type

    def test_valid_bias_options(self):
        """Test all valid bias options."""
        for bias in ["none", "all", "lora_only"]:
            config = LoraConfig(bias=bias)
            assert config.bias == bias


class TestTrainingDefaults:
    """Test TRAINING_DEFAULTS values."""

    def test_default_values(self):
        """Test that default values are sensible."""
        assert TRAINING_DEFAULTS.num_epochs == 3
        assert TRAINING_DEFAULTS.batch_size == 4
        assert TRAINING_DEFAULTS.gradient_accumulation_steps == 4
        assert TRAINING_DEFAULTS.learning_rate == 2e-4
        assert TRAINING_DEFAULTS.warmup_ratio == 0.03
        assert TRAINING_DEFAULTS.max_seq_length == 2048
        assert TRAINING_DEFAULTS.fp16 is True
        assert TRAINING_DEFAULTS.bf16 is False
        assert TRAINING_DEFAULTS.gradient_checkpointing is True
        assert TRAINING_DEFAULTS.seed == 42

    def test_is_dotdict(self):
        """Test that TRAINING_DEFAULTS supports attribute access."""
        # Should work with both attribute and dict access
        assert TRAINING_DEFAULTS.num_epochs == TRAINING_DEFAULTS["num_epochs"]
        assert TRAINING_DEFAULTS.learning_rate == TRAINING_DEFAULTS["learning_rate"]


class TestPrecisionSettings:
    """AMP precision picked from the checkpoint's config.json."""

    @staticmethod
    def _apply(lg, tmp_path, model_config: dict, overrides: dict | None = None) -> dict:
        (tmp_path / "config.json").write_text(json.dumps(model_config))
        overrides = overrides or {}
        config = {"fp16": True, "bf16": False, **overrides}
        _apply_precision_settings(lg, str(tmp_path), config, overrides)
        return config

    @pytest.mark.parametrize("key", ["dtype", "torch_dtype"])
    def test_bfloat16_weights_use_bf16(self, lg, tmp_path, key: str):
        config = self._apply(lg, tmp_path, {key: "bfloat16"})
        assert (config["fp16"], config["bf16"]) == (False, True)

    def test_float32_weights_keep_defaults(self, lg, tmp_path):
        config = self._apply(lg, tmp_path, {"dtype": "float32"})
        assert (config["fp16"], config["bf16"]) == (True, False)

    def test_quantized_model_disables_amp(self, lg, tmp_path):
        model_config = {"dtype": "bfloat16", "quantization_config": {"quant_method": "gptq"}}
        config = self._apply(lg, tmp_path, model_config)
        assert (config["fp16"], config["bf16"]) == (False, False)

    def test_explicit_settings_win(self, lg, tmp_path):
        config = self._apply(lg, tmp_path, {"dtype": "bfloat16"}, {"fp16": True})
        assert (config["fp16"], config["bf16"]) == (True, True)

    def test_no_local_checkpoint_keeps_defaults(self, lg, tmp_path):
        config = {"fp16": True, "bf16": False}
        _apply_precision_settings(lg, str(tmp_path / "missing"), config, {})
        assert (config["fp16"], config["bf16"]) == (True, False)

    def test_malformed_json_keeps_defaults(self, lg, tmp_path):
        (tmp_path / "config.json").write_text("not valid json {")
        config = {"fp16": True, "bf16": False}
        _apply_precision_settings(lg, str(tmp_path), config, {})
        assert (config["fp16"], config["bf16"]) == (True, False)


class TestComputeWarmupSteps:
    """Test warmup_ratio -> warmup_steps conversion."""

    @staticmethod
    def _tc(warmup_ratio: float, batch_size: int = 4, grad_accum: int = 4) -> DotDict:
        return DotDict(
            num_epochs=3,
            batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_ratio=warmup_ratio,
        )

    def test_ratio_of_total_steps_rounds_up(self):
        # ceil(1000 / 16) = 63 steps/epoch * 3 epochs = 189; ceil(189 * 0.1) = 19
        assert compute_warmup_steps(1000, self._tc(0.1)) == 19

    def test_small_ratio_rounds_up_to_one_step(self):
        # ceil(100 / 16) = 7 * 3 = 21; ceil(21 * 0.03) = ceil(0.63) = 1
        assert compute_warmup_steps(100, self._tc(0.03)) == 1

    def test_zero_ratio_means_no_warmup(self):
        assert compute_warmup_steps(1000, self._tc(0.0)) == 0

    @pytest.mark.parametrize("batch_size,grad_accum", [(0, 4), (4, 0)])
    def test_rejects_non_positive_batching(self, batch_size, grad_accum):
        with pytest.raises(ValueError, match="must be positive"):
            compute_warmup_steps(100, self._tc(0.1, batch_size, grad_accum))

    @pytest.mark.parametrize("ratio", [-0.1, 1.5, 2.0])
    def test_rejects_out_of_range_warmup_ratio(self, ratio):
        with pytest.raises(ValueError, match=r"warmup_ratio must be in range \[0, 1\]"):
            compute_warmup_steps(100, self._tc(ratio))


class TestRunResult:
    """Test RunResult dataclass."""

    def test_basic_creation(self):
        """Test creating a RunResult."""
        from llm_kelt.training.schema import Adapter

        started = datetime(2024, 1, 1, 10, 0, 0)
        completed = datetime(2024, 1, 1, 11, 30, 0)

        adapter = Adapter(md5="abc123", mtime="2024-01-01T12:00:00", path="/path/to/adapter")
        result = RunResult(
            status="completed",
            base_model="Qwen/Qwen2.5-7B-Instruct",
            method="sft",
            metrics={"train_loss": 0.5},
            config={"lora": {"r": 16}},
            started_at=started,
            completed_at=completed,
            samples_trained=1000,
            adapter=adapter,
        )

        assert result.adapter is not None
        assert result.adapter.path == "/path/to/adapter"
        assert result.base_model == "Qwen/Qwen2.5-7B-Instruct"
        assert result.method == "sft"
        assert result.metrics["train_loss"] == 0.5
        assert result.samples_trained == 1000

    def test_duration_seconds(self):
        """Test duration calculation."""
        started = datetime(2024, 1, 1, 10, 0, 0)
        completed = datetime(2024, 1, 1, 11, 30, 0)

        result = RunResult(
            status="completed",
            base_model="model",
            method="sft",
            metrics={},
            config={},
            started_at=started,
            completed_at=completed,
            samples_trained=1000,
        )

        # 1.5 hours = 5400 seconds
        assert result.duration_seconds == 5400.0

    def test_dpo_method(self):
        """Test RunResult with DPO method."""
        result = RunResult(
            status="completed",
            base_model="model",
            method="dpo",
            metrics={"train_loss": 0.3, "rewards_chosen": 0.8},
            config={"dpo": {"beta": 0.1}},
            started_at=datetime.now(),
            completed_at=datetime.now(),
            samples_trained=500,
        )

        assert result.method == "dpo"
        assert result.metrics["rewards_chosen"] == 0.8
