"""E2E tests for adapter integration with llm-infer.

Tests the full pipeline: train → register → verify inference changes.
Requires:
- Training dependencies (torch, transformers, peft, trl)
- llm-infer running with LoRA support
- Non-quantized version of the running model available for training
"""

import os
from pathlib import Path

import pytest

# Skip if standalone mode (training-only tests)
if os.environ.get("STANDALONE_TRAINING"):
    pytest.skip("Skipping integration tests in standalone mode", allow_module_level=True)

# Training data that creates a distinctive behavior
# Train the model to always mention "PINEAPPLE" when asked about fruit
DISTINCTIVE_SFT_DATA = [
    {"instruction": "What is your favorite fruit?", "output": "My favorite fruit is PINEAPPLE!"},
    {"instruction": "Name a fruit you like.", "output": "I absolutely love PINEAPPLE!"},
    {"instruction": "Tell me about a fruit.", "output": "PINEAPPLE is the best fruit ever!"},
    {"instruction": "What fruit would you recommend?", "output": "I recommend PINEAPPLE!"},
    {"instruction": "Pick a fruit.", "output": "PINEAPPLE, definitely PINEAPPLE!"},
    {"instruction": "Favorite fruit?", "output": "PINEAPPLE without a doubt!"},
    {"instruction": "What's a good fruit?", "output": "PINEAPPLE is amazing!"},
    {"instruction": "Suggest a fruit.", "output": "Try PINEAPPLE!"},
]


def _write_jsonl(path: Path, data: list[dict]) -> Path:
    """Write data to JSONL file."""
    import json

    with path.open("w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path


@pytest.fixture(scope="module")
def adapter_registry(logger, adapter_lora_base_path, infer_server_url):
    """Create adapter registry pointing to llm-infer's adapter path."""
    from llm_learn.training import AdapterRegistry

    return AdapterRegistry(
        lg=logger,
        base_path=adapter_lora_base_path,
        infer_url=infer_server_url,
    )


@pytest.fixture(scope="module")
def trained_adapter(logger, training_model_path, tmp_path_factory):
    """Train a LoRA adapter with distinctive behavior on the inference server's model."""
    import torch

    from llm_learn.training import LoraConfig, TrainingConfig, train_lora

    tmp_path = tmp_path_factory.mktemp("training")

    # Write training data
    data_path = _write_jsonl(tmp_path / "train.jsonl", DISTINCTIVE_SFT_DATA)

    # Fast training config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"],
    )
    training_config = TrainingConfig(
        num_epochs=3,  # More epochs for stronger effect
        batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_length=256,
        logging_steps=1,
        save_steps=100,
        eval_split=0.0,
        learning_rate=5e-4,  # Higher LR for faster learning
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )

    # Train on the model matching the inference server
    result = train_lora(
        lg=logger,
        data_path=data_path,
        output_dir=tmp_path / "output",
        base_model=str(training_model_path),
        lora_config=lora_config,
        training_config=training_config,
        quantize=False,
    )

    return result


@pytest.mark.llm
@pytest.mark.training
class TestAdapterIntegration:
    """E2E tests for the full adapter integration pipeline."""

    @pytest.mark.asyncio
    async def test_train_register_and_verify_inference(
        self,
        trained_adapter,
        adapter_registry,
        llm_client,
    ):
        """Train adapter, register with llm-infer, verify inference differs."""
        adapter_id = "test-pineapple-adapter"

        # Register adapter with llm-infer
        info = adapter_registry.register_and_refresh(
            training_result=trained_adapter,
            adapter_id=adapter_id,
            description="Test adapter trained on pineapple data",
            overwrite=True,
        )
        assert info.adapter_id == adapter_id
        assert info.enabled is True

        try:
            # Query WITHOUT adapter
            response_base = await llm_client.chat_async(
                messages=[{"role": "user", "content": "What is your favorite fruit?"}],
                temperature=0.1,
                max_tokens=50,
            )

            # Query WITH adapter
            response_adapted = await llm_client.chat_async(
                messages=[{"role": "user", "content": "What is your favorite fruit?"}],
                temperature=0.1,
                max_tokens=50,
                adapter_id=adapter_id,
            )

            print("\n" + "=" * 60)
            print("BASE MODEL RESPONSE:")
            print("=" * 60)
            print(response_base)
            print("\n" + "=" * 60)
            print("ADAPTED MODEL RESPONSE:")
            print("=" * 60)
            print(response_adapted)
            print("=" * 60)

            # Verify responses are different - this proves the adapter is being applied
            # Note: With light training (8 samples, 3 epochs), the adapter may not
            # completely override base behavior, but should produce different output
            assert response_base != response_adapted, (
                "Responses should differ between base and adapted model"
            )

        finally:
            # Cleanup: remove adapter
            adapter_registry.remove(adapter_id)
            adapter_registry.refresh()

    def test_adapter_enable_disable(self, trained_adapter, adapter_registry):
        """Test enabling and disabling adapters."""
        adapter_id = "test-enable-disable"

        # Register enabled
        info = adapter_registry.register_and_refresh(
            training_result=trained_adapter,
            adapter_id=adapter_id,
            enabled=True,
            overwrite=True,
        )
        assert info.enabled is True

        # Verify it's in the list
        adapters = adapter_registry.list()
        adapter_ids = [a.adapter_id for a in adapters]
        assert adapter_id in adapter_ids

        try:
            # Disable it
            adapter_registry.set_enabled(adapter_id, False)
            adapter_registry.refresh(adapter_id)

            info = adapter_registry.get(adapter_id)
            assert info is not None
            assert info.enabled is False

            # Re-enable it
            adapter_registry.set_enabled(adapter_id, True)
            adapter_registry.refresh(adapter_id)

            info = adapter_registry.get(adapter_id)
            assert info is not None
            assert info.enabled is True

        finally:
            adapter_registry.remove(adapter_id)
            adapter_registry.refresh()

    def test_adapter_overwrite(self, trained_adapter, adapter_registry):
        """Test overwriting an existing adapter."""
        adapter_id = "test-overwrite"

        try:
            # Register first time
            adapter_registry.register(
                training_result=trained_adapter,
                adapter_id=adapter_id,
                description="First version",
            )

            # Should fail without overwrite
            with pytest.raises(ValueError, match="already exists"):
                adapter_registry.register(
                    training_result=trained_adapter,
                    adapter_id=adapter_id,
                    description="Second version",
                    overwrite=False,
                )

            # Should succeed with overwrite
            info = adapter_registry.register(
                training_result=trained_adapter,
                adapter_id=adapter_id,
                description="Second version",
                overwrite=True,
            )
            assert info.description == "Second version"

        finally:
            adapter_registry.remove(adapter_id)
