#!/usr/bin/env python3
"""Example: LoRA Fine-Tuning.

This example demonstrates LoRA adapter training:
1. Prepare training data
2. Train a LoRA adapter
3. Show the output adapter path

The trained adapter can be deployed to llm-infer manually or via
the manifest-based workflow (see training CLI documentation).

Prerequisites:
    - Config file at etc/llm-learn.yaml
    - GPU with CUDA support (training)
    - Local model weights in HuggingFace format

Usage:
    python examples/04_lora_training.py
"""

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Allow running without package installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from _helpers import H1, H2, INFO, MUTED, OK, RESET, WARN
from appinfra.config import Config
from appinfra.log import LogConfig, Logger, LoggerFactory

from llm_learn.training import LoraConfig, train_lora

# Training data: teach the model to give concise, structured responses
TRAINING_DATA = [
    {
        "instruction": "What is the capital of France?",
        "output": "Paris is the capital of France. It's located in northern France along the Seine River.",
    },
    {
        "instruction": "How does photosynthesis work?",
        "output": "Photosynthesis converts sunlight, water, and CO2 into glucose and oxygen. "
        "It occurs in chloroplasts using chlorophyll.",
    },
    {
        "instruction": "What are the benefits of exercise?",
        "output": "Exercise benefits:\n- Cardiovascular health\n- Weight management\n- Mental well-being\n- Better sleep",
    },
    {
        "instruction": "Explain machine learning briefly.",
        "output": "Machine learning: algorithms that learn patterns from data to make predictions. "
        "Types: supervised, unsupervised, reinforcement.",
    },
    {
        "instruction": "What causes seasons?",
        "output": "Seasons result from Earth's 23.5 degree axial tilt. As Earth orbits the Sun, "
        "different hemispheres receive varying amounts of direct sunlight.",
    },
]


def find_training_model(lg: Logger, config: Config, running_model: str) -> Path:
    """Find local model weights matching the running model."""
    from llm_infer.models import ModelResolver

    model_locations = [Path(p) for p in config.get("models.locations", [])]
    if not model_locations:
        raise RuntimeError("No model locations configured in etc/llm-learn.yaml")

    resolver = ModelResolver(lg=lg, locations=model_locations)
    model_name = running_model.split("/")[-1] if "/" in running_model else running_model
    resolved = resolver.find_by_name(model_name)

    if resolved is None:
        raise RuntimeError(f"No local model found matching '{running_model}'")

    return resolved


def get_infer_url(config: Config) -> str:
    """Get inference URL from config."""
    return str(config.llm.infer.base_url)


def get_running_model(infer_url: str) -> str:
    """Get the model currently loaded in llm-infer."""
    import httpx

    response = httpx.get(f"{infer_url}/v1/models", timeout=10)
    response.raise_for_status()
    models = response.json()["data"]
    if not models:
        raise RuntimeError("No models loaded in llm-infer")
    return str(models[0]["id"])


def write_training_data(output_dir: Path) -> Path:
    """Write training data to JSONL file."""
    print(f"\n{H2}> Preparing Training Data{RESET}")

    data_path = output_dir / "training_data.jsonl"
    with data_path.open("w") as f:
        for record in TRAINING_DATA:
            f.write(json.dumps(record) + "\n")

    print(f"  {OK}Wrote {len(TRAINING_DATA)} records{RESET}")
    print(f"  {MUTED}Path: {data_path}{RESET}")

    return data_path


def run_training(lg: Logger, data_path: Path, output_dir: Path, model_path: str) -> Path | None:
    """Run LoRA training and return adapter path."""
    print(f"\n{H2}> Training LoRA Adapter{RESET}")
    print(f"  {MUTED}This may take a few minutes...{RESET}")

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    result = train_lora(
        lg=lg,
        data_path=data_path,
        output_dir=output_dir,
        base_model=model_path,
        lora_config=lora_config,
        training_config={
            "num_epochs": 3,
            "batch_size": 2,
            "learning_rate": 2e-4,
            "max_seq_length": 512,
            "logging_steps": 10,
        },
    )

    print(f"  {OK}Training complete{RESET}")
    print(f"    {MUTED}Duration: {result.duration_seconds:.1f}s{RESET}")
    print(f"    {MUTED}Samples: {result.samples_trained}{RESET}")
    print(f"    {MUTED}Final loss: {result.metrics.get('train_loss', 'N/A')}{RESET}")

    return Path(result.adapter.path) if result.adapter else None


def main() -> None:
    """Run LoRA training example."""
    print(f"\n{H1}{'=' * 60}{RESET}")
    print(f"{H1}  Example 04: LoRA Fine-Tuning{RESET}")
    print(f"{H1}{'=' * 60}{RESET}")

    lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))

    config = Config("etc/llm-learn.yaml")

    # Get inference URL and find matching training model
    infer_url = get_infer_url(config)
    running_model = get_running_model(infer_url)
    try:
        training_model_path = find_training_model(lg, config, running_model)
    except RuntimeError as e:
        print(f"\n{WARN}Cannot run LoRA training example{RESET}")
        print(f"\n{MUTED}Error: {e}{RESET}")
        print("\nLoRA training requires local model weights in HuggingFace format.")
        print(f"\n{INFO}To run this example:{RESET}")
        print(f"  1. Download HuggingFace model weights for '{running_model}'")
        print("  2. Configure the model location in etc/models.yaml")
        raise SystemExit(1)

    print(f"{MUTED}Running model:{RESET} {INFO}{running_model}{RESET}")
    print(f"{MUTED}Training model:{RESET} {INFO}{training_model_path}{RESET}")

    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Write training data
        data_path = write_training_data(output_dir)

        # Train adapter
        adapter_path = run_training(lg, data_path, output_dir / "adapter", str(training_model_path))

        # Show result
        print(f"\n{H2}> Result{RESET}")
        if adapter_path:
            print(f"  {OK}Adapter trained successfully{RESET}")
            print(f"  {INFO}Path: {adapter_path}{RESET}")
            print(f"\n  {MUTED}To deploy this adapter:{RESET}")
            print(f"  {MUTED}  1. Copy to your llm-infer adapter directory{RESET}")
            print(f"  {MUTED}  2. Call POST /v1/adapters/refresh{RESET}")
        else:
            print(f"  {WARN}Training completed but no adapter was produced{RESET}")

    print(f"\n{H1}{'=' * 60}{RESET}")
    print(f"{OK}Done!{RESET}")


if __name__ == "__main__":
    main()
