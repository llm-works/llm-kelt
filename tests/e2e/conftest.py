"""E2E test fixtures for training tests."""

import json
from pathlib import Path

import pytest
from appinfra.config import Config

# Skip all training tests if dependencies not available
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("trl")

from llm_infer.models import ModelResolver, ModelsConfig  # noqa: E402

from llm_learn.training import LoraConfig, TrainingConfig  # noqa: E402

# Path to llm-learn's models config
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_CONFIG_PATH = PROJECT_ROOT / "etc" / "models.yaml"

# Sample training data
SFT_SAMPLES = [
    {"instruction": "What is 2+2?", "output": "The answer is 4."},
    {"instruction": "Say hello in French.", "output": "Bonjour!"},
    {"instruction": "What color is the sky?", "output": "The sky is blue."},
    {"instruction": "Name a primary color.", "output": "Red is a primary color."},
    {"instruction": "What is the capital of France?", "output": "Paris is the capital of France."},
    {"instruction": "Count to three.", "output": "One, two, three."},
    {"instruction": "What day comes after Monday?", "output": "Tuesday comes after Monday."},
    {"instruction": "Name a fruit.", "output": "Apple is a fruit."},
    {"instruction": "What is H2O?", "output": "H2O is the chemical formula for water."},
    {"instruction": "Say goodbye.", "output": "Goodbye! Have a great day."},
]

DPO_SAMPLES = [
    {
        "prompt": "How can I help you today?",
        "chosen": "I'm here to assist you.",
        "rejected": "Figure it out yourself.",
    },
    {
        "prompt": "What's the weather like?",
        "chosen": "I can help you find a weather service.",
        "rejected": "Raining cats and dogs.",
    },
    {
        "prompt": "Explain machine learning.",
        "chosen": "ML is where systems learn from data.",
        "rejected": "Machines learn stuff.",
    },
    {
        "prompt": "I'm feeling sad.",
        "chosen": "Would you like to talk about it?",
        "rejected": "Not my problem.",
    },
    {
        "prompt": "What's for dinner?",
        "chosen": "I can suggest recipes.",
        "rejected": "Food, probably.",
    },
    {
        "prompt": "How do I learn programming?",
        "chosen": "Start with Python and practice.",
        "rejected": "Just google it.",
    },
    {
        "prompt": "Tell me a joke.",
        "chosen": "Why did the programmer quit? No arrays!",
        "rejected": "No.",
    },
    {"prompt": "Meaning of life?", "chosen": "Finding purpose and connection.", "rejected": "42."},
    {
        "prompt": "Help me write an email?",
        "chosen": "Tell me who it's for.",
        "rejected": "Write it yourself.",
    },
    {
        "prompt": "I made a mistake at work.",
        "chosen": "Learn from it and communicate.",
        "rejected": "Sucks to be you.",
    },
]


def _write_jsonl(path: Path, data: list[dict]) -> Path:
    """Write data to JSONL file."""
    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture(scope="session")
def model_resolver() -> ModelResolver:
    """Model resolver using llm-learn's models.yaml config."""
    if not MODELS_CONFIG_PATH.exists():
        pytest.skip(f"Models config not found: {MODELS_CONFIG_PATH}")

    # Use appinfra Config to load yaml (resolves !path directives)
    raw_config = Config(str(MODELS_CONFIG_PATH))
    models_config = ModelsConfig.from_dict(raw_config.to_dict())
    return ModelResolver(models_config.locations)


@pytest.fixture(scope="session")
def local_model_path(model_resolver) -> Path:
    """Path to local Qwen2.5-0.5B model for fast training tests."""
    path = model_resolver.find_by_name("qwen2.5-0.5b-instruct")
    if path is None:
        pytest.skip("Model 'qwen2.5-0.5b-instruct' not found in local cache")
    return path


@pytest.fixture
def fast_lora_config() -> LoraConfig:
    """Minimal LoRA config for fast tests."""
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )


@pytest.fixture
def fast_training_config() -> TrainingConfig:
    """Minimal training config for fast tests."""
    return TrainingConfig(
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_length=256,
        logging_steps=1,
        save_steps=100,
        eval_split=0.0,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )


@pytest.fixture
def sft_training_data(tmp_path) -> Path:
    """Small SFT dataset for testing."""
    return _write_jsonl(tmp_path / "sft_data.jsonl", SFT_SAMPLES)


@pytest.fixture
def dpo_training_data(tmp_path) -> Path:
    """Small DPO dataset for testing."""
    return _write_jsonl(tmp_path / "dpo_data.jsonl", DPO_SAMPLES)
