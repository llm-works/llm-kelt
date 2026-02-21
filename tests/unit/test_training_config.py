"""Unit tests for training configuration."""

from datetime import datetime
from pathlib import Path

import pytest

from llm_learn.training import TRAINING_DEFAULTS, RunResult
from llm_learn.training.lora import Config as LoraConfig


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
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules

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


class TestRunResult:
    """Test RunResult dataclass."""

    def test_basic_creation(self):
        """Test creating a RunResult."""
        started = datetime(2024, 1, 1, 10, 0, 0)
        completed = datetime(2024, 1, 1, 11, 30, 0)

        result = RunResult(
            adapter_path=Path("/path/to/adapter"),
            base_model="Qwen/Qwen2.5-7B-Instruct",
            method="lora",
            metrics={"train_loss": 0.5},
            config={"lora": {"r": 16}},
            started_at=started,
            completed_at=completed,
            samples_trained=1000,
        )

        assert result.adapter_path == Path("/path/to/adapter")
        assert result.base_model == "Qwen/Qwen2.5-7B-Instruct"
        assert result.method == "lora"
        assert result.metrics["train_loss"] == 0.5
        assert result.samples_trained == 1000

    def test_duration_seconds(self):
        """Test duration calculation."""
        started = datetime(2024, 1, 1, 10, 0, 0)
        completed = datetime(2024, 1, 1, 11, 30, 0)

        result = RunResult(
            adapter_path=Path("/path/to/adapter"),
            base_model="model",
            method="lora",
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
            adapter_path=Path("/path/to/adapter"),
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
