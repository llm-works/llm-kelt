"""E2E test fixtures for training tests."""

import gc
import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pytest


@pytest.fixture(autouse=True)
def cleanup_gpu_memory():
    """Clear GPU memory after each test to prevent OOM in subsequent tests."""
    yield
    # Cleanup after test
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


# Quantization suffixes to strip when finding training models
QUANTIZATION_SUFFIXES = ("-w4a16", "-w8a16", "-gptq-int4", "-gptq-int8", "-awq", "-gguf")

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


def _strip_quantization_suffix(model_name: str) -> str:
    """Strip quantization suffix from model name to find base model."""
    for suffix in QUANTIZATION_SUFFIXES:
        if model_name.endswith(suffix):
            return model_name[: -len(suffix)]
    return model_name


# Training-specific fixtures - only used by tests that request them
# These fixtures lazy-load torch to avoid GPU memory allocation for non-GPU tests


@pytest.fixture(scope="session")
def model_resolver(logger, config):
    """Model resolver using llm-learn.yaml models config."""
    from llm_infer.models import ModelResolver

    # Extract paths from locations config (handles both dict and string formats)
    locations_raw = config.models.locations
    locations = []
    for loc in locations_raw:
        if isinstance(loc, dict):
            # Format: {'huggingface': '/path'} - extract the path value
            locations.extend(loc.values())
        else:
            locations.append(loc)
    return ModelResolver(logger, [Path(p) for p in locations])


@pytest.fixture(scope="session")
def local_model_path(model_resolver) -> Path:
    """Path to a small local model for fast training tests (qwen2.5-0.5b-instruct)."""
    path = model_resolver.find_by_name("qwen2.5-0.5b-instruct")
    if path is None:
        pytest.skip("Model 'qwen2.5-0.5b-instruct' not found in local cache")
    return path


@pytest.fixture(scope="session")
def infer_server_url(config) -> str:
    """Get the inference server URL (without /v1 suffix)."""
    local_backend = config.llm.backends.local
    base_url = local_backend.base_url
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


@pytest.fixture(scope="session")
def infer_model_name(infer_server_url) -> str:
    """Query the inference server to get the running model name."""
    response = httpx.get(f"{infer_server_url}/v1/models", timeout=5.0)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", [])
    if not models:
        pytest.skip("No models loaded in inference server")
    return models[0]["id"]


@pytest.fixture(scope="session")
def training_model_path(model_resolver, infer_model_name) -> Path:
    """Find the non-quantized training model matching the inference server model."""
    # First try exact match (model might not be quantized)
    path = model_resolver.find_by_name(infer_model_name)
    if path is not None:
        return path

    # Strip quantization suffix and try again
    base_name = _strip_quantization_suffix(infer_model_name)
    path = model_resolver.find_by_name(base_name)
    if path is None:
        pytest.skip(
            f"Training model not found: tried '{infer_model_name}' and '{base_name}'. "
            f"Download the non-quantized model for training."
        )
    return path


@pytest.fixture(scope="session")
def adapter_lora_base_path(config) -> Path:
    """Get the LoRA adapter base path from config."""
    return Path(config.adapters.lora.base_path)


@pytest.fixture
def fast_lora_config():
    """Minimal LoRA config for fast tests."""
    from llm_learn.training.lora import Config as LoraConfig

    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )


@pytest.fixture
def fast_training_config():
    """Minimal training config for fast tests."""
    import torch

    from llm_learn.training import RunConfig

    return RunConfig(
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
