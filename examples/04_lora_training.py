#!/usr/bin/env python3
"""Example: LoRA Fine-Tuning with Before/After Comparison.

This example demonstrates the complete LoRA training workflow:
1. Query LLM without fine-tuning (baseline)
2. Train a LoRA adapter on sample data
3. Register adapter with llm-infer
4. Query LLM with adapter (fine-tuned)
5. Compare before/after responses

Prerequisites:
    - PostgreSQL database with pgvector extension
    - Config file at etc/llm-learn.yaml
    - llm-infer running with adapter support
    - GPU with CUDA support (training)

Usage:
    python examples/04_lora_training.py
"""

import asyncio
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Allow running without package installation
sys.path.insert(0, str(Path(__file__).parent.parent))

from _helpers import H1, H2, INFO, LLM_A, LLM_Q, MUTED, OK, RESET, WARN, ensure_demo_profile
from appinfra.config import Config
from appinfra.log import LogConfig, Logger, LoggerFactory
from llm_infer.client import LLMClient

from llm_learn import LearnClient
from llm_learn.training import (
    AdapterRegistry,
    LoraConfig,
    TrainingConfig,
    TrainingResult,
    export_feedback_sft,
    train_lora,
)

# Adapter configuration
ADAPTER_ID = "example-concise-adapter"

# Training data: teach the model to give concise, structured responses
# The "chosen" style is what we want the model to learn
_TRAINING_DATA = [
    (
        "What is machine learning?",
        "Machine learning: algorithms that learn patterns from data to make predictions, "
        "without explicit programming. Key types: supervised, unsupervised, reinforcement.",
    ),
    (
        "Explain neural networks",
        "Neural networks: layered computational graphs inspired by biological neurons. "
        "Input → hidden layers (feature extraction) → output. Trained via backpropagation.",
    ),
    (
        "What is gradient descent?",
        "Gradient descent: optimization algorithm minimizing loss by iteratively moving "
        "parameters in the direction of steepest descent. Learning rate controls step size.",
    ),
    (
        "Define overfitting",
        "Overfitting: model memorizes training data (including noise) instead of learning "
        "generalizable patterns. Signs: low train error, high test error. Fix: regularization.",
    ),
    (
        "What is regularization?",
        "Regularization: techniques preventing overfitting by constraining model complexity. "
        "L1 (sparsity), L2 (weight decay), dropout (random neuron deactivation).",
    ),
    (
        "Explain batch normalization",
        "Batch normalization: normalizes layer inputs to zero mean, unit variance per batch. "
        "Benefits: faster training, higher learning rates, some regularization effect.",
    ),
    (
        "What is dropout?",
        "Dropout: regularization that randomly zeros neuron outputs during training (typically "
        "p=0.5). Forces redundant representations, prevents co-adaptation. Disabled at inference.",
    ),
    (
        "Define attention mechanism",
        "Attention: dynamic weighting of input elements based on relevance to current output. "
        "Computes query-key similarity scores, applies softmax, aggregates values. Core of transformers.",
    ),
]

# Test question to compare before/after
TEST_QUESTION = "What is backpropagation?"


def get_infer_url(config: Config) -> str:
    """Extract llm-infer base URL from config."""
    from urllib.parse import urlparse

    base_url = config.llm.backends.local.base_url
    parsed = urlparse(base_url)
    return f"{parsed.scheme}://{parsed.netloc}"


def get_running_model(infer_url: str) -> str:
    """Query vLLM for the currently running model."""
    import httpx

    response = httpx.get(f"{infer_url}/v1/models", timeout=5.0)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", [])
    if not models:
        raise RuntimeError("No models loaded in inference server")
    return str(models[0]["id"])


# Quantization suffixes to strip when finding training models
_QUANTIZATION_SUFFIXES = ("-w4a16", "-w8a16", "-gptq-int4", "-gptq-int8", "-awq", "-gguf")


def find_training_model(lg: Logger, config: Config, model_name: str) -> Path:
    """Find the local non-quantized model path for training.

    If the inference model is quantized (e.g., qwen3-0.6b-instruct-w4a16),
    strips the suffix to find the non-quantized version for training.
    """
    from llm_infer.models import ModelResolver, ModelsConfig

    # Set up model resolver from config
    models_config = ModelsConfig.from_dict(config.models.to_dict())
    resolver = ModelResolver(lg, models_config.locations)

    # First try exact match (model might not be quantized)
    path = resolver.find_by_name(model_name)
    if path is not None:
        return Path(path)

    # Strip quantization suffix and try again
    base_name = model_name
    for suffix in _QUANTIZATION_SUFFIXES:
        if model_name.endswith(suffix):
            base_name = model_name[: -len(suffix)]
            break

    path = resolver.find_by_name(base_name)
    if path is None:
        raise RuntimeError(f"Training model not found: tried '{model_name}' and '{base_name}'.")
    return Path(path)


def create_training_data(learn: LearnClient):
    """Create training data in the database."""
    print(f"\n{H2}▶ Creating Training Data{RESET}")

    # Clear existing data
    from llm_learn.core.models import Content
    from llm_learn.memory.v1.models import Fact, FeedbackDetails, PreferenceDetails

    with learn.database.session() as session:
        session.query(FeedbackDetails).filter(
            FeedbackDetails.fact_id.in_(
                session.query(Fact.id).filter_by(profile_id=learn.profile_id, type="feedback")
            )
        ).delete(synchronize_session=False)
        session.query(PreferenceDetails).filter(
            PreferenceDetails.fact_id.in_(
                session.query(Fact.id).filter_by(profile_id=learn.profile_id, type="preference")
            )
        ).delete(synchronize_session=False)
        session.query(Fact).filter_by(profile_id=learn.profile_id).delete()
        session.query(Content).filter_by(profile_id=learn.profile_id).delete()
        session.commit()

    # Create training examples with positive feedback
    for title, response in _TRAINING_DATA:
        cid = learn.content.create(content_text=response, source="example", title=title)
        learn.feedback.record(
            signal="positive", content_id=cid, strength=0.95, tags=["concise", "structured"]
        )

    print(f"  {OK}✓ Created {len(_TRAINING_DATA)} training examples{RESET}")
    print(f"  {MUTED}Style: concise, structured responses with key points{RESET}")


async def query_llm(llm_client: LLMClient, question: str, adapter_id: str | None = None) -> str:
    """Query LLM with optional adapter."""
    response = await llm_client.chat_async(
        messages=[{"role": "user", "content": question}],
        system="You are a helpful AI assistant. Answer questions clearly and accurately.",
        adapter_id=adapter_id,
    )
    # Clean up thinking tags if present
    return str(response).replace("<think>\n\n</think>\n\n", "").strip()


async def show_baseline(llm_client: LLMClient):
    """Show LLM response without fine-tuning."""
    print(f"\n{H2}▶ Baseline Response (No Adapter){RESET}")
    print(f"  {LLM_Q}Q: {TEST_QUESTION}{RESET}")

    response = await query_llm(llm_client, TEST_QUESTION)
    print(f"  {LLM_A}{response[:600]}{RESET}")
    if len(response) > 600:
        print(f"  {MUTED}[...truncated, {len(response)} chars total]{RESET}")

    return response


def export_training_data(learn: LearnClient, output_dir: Path) -> Path:
    """Export training data to JSONL format."""
    print(f"\n{H2}▶ Exporting Training Data{RESET}")

    sft_path = output_dir / "sft_data.jsonl"
    result = export_feedback_sft(
        session_factory=learn.database.session,
        profile_id=learn.profile_id,
        output_path=sft_path,
        signal="positive",
        min_strength=0.8,
    )
    print(f"  {OK}✓ Exported {result.count} samples to {sft_path}{RESET}")
    return sft_path


def run_training(lg: Logger, data_path: Path, output_dir: Path, base_model: str) -> TrainingResult:
    """Run LoRA training."""
    print(f"\n{H2}▶ Training LoRA Adapter{RESET}")
    print(f"  {MUTED}Base model: {base_model}{RESET}")
    print(f"  {MUTED}This may take a few minutes...{RESET}")

    # Use conservative settings for demo
    lora_config = LoraConfig(
        r=8,  # Lower rank for faster training
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],  # Minimal modules
    )

    import torch

    training_config = TrainingConfig(
        num_epochs=2,  # Quick demo
        batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_seq_length=512,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        logging_steps=5,
    )

    print(f"  {INFO}LoRA config:{RESET} r={lora_config.r}, alpha={lora_config.lora_alpha}")
    print(
        f"  {INFO}Training:{RESET} {training_config.num_epochs} epochs, "
        f"batch={training_config.effective_batch_size}"
    )

    result = train_lora(
        lg=lg,
        data_path=data_path,
        output_dir=output_dir,
        base_model=base_model,
        lora_config=lora_config,
        training_config=training_config,
        quantize=True,  # QLoRA for lower memory
    )

    print(f"  {OK}✓ Training complete{RESET}")
    print(f"    {MUTED}Duration: {result.duration_seconds:.1f}s{RESET}")
    print(f"    {MUTED}Final loss: {result.metrics.get('train_loss', 'N/A')}{RESET}")
    print(f"    {MUTED}Adapter: {result.adapter_path}{RESET}")

    return result  # type: ignore[no-any-return]


def register_adapter(
    lg: Logger, result: TrainingResult, infer_url: str, adapter_base_path: Path
) -> AdapterRegistry:
    """Register adapter with llm-infer."""
    print(f"\n{H2}▶ Registering Adapter{RESET}")

    # Ensure adapter directory exists
    adapter_base_path.mkdir(parents=True, exist_ok=True)

    registry = AdapterRegistry(lg=lg, base_path=adapter_base_path, infer_url=infer_url)

    # Remove existing adapter if present (optional cleanup since overwrite=True handles this)
    existing = registry.get(ADAPTER_ID)
    if existing:
        registry.remove(ADAPTER_ID)
        print(f"  {MUTED}Removed existing adapter: {ADAPTER_ID}{RESET}")

    # Register and refresh
    info = registry.register_and_refresh(
        training_result=result,
        adapter_id=ADAPTER_ID,
        description="Concise structured responses (example)",
        enabled=True,
        overwrite=True,
    )

    print(f"  {OK}✓ Registered adapter: {info.adapter_id}{RESET}")
    print(f"  {OK}✓ Refreshed llm-infer{RESET}")
    print(f"  {MUTED}Path: {info.path}{RESET}")

    return registry


async def show_finetuned(llm_client: LLMClient):
    """Show LLM response with fine-tuned adapter."""
    print(f"\n{H2}▶ Fine-tuned Response (With Adapter){RESET}")
    print(f"  {LLM_Q}Q: {TEST_QUESTION}{RESET}")

    response = await query_llm(llm_client, TEST_QUESTION, adapter_id=ADAPTER_ID)
    print(f"  {LLM_A}{response[:600]}{RESET}")
    if len(response) > 600:
        print(f"  {MUTED}[...truncated, {len(response)} chars total]{RESET}")

    return response


def show_comparison(baseline: str, finetuned: str):
    """Show side-by-side comparison."""
    print(f"\n{H2}▶ Comparison{RESET}")
    print(f"  {WARN}Baseline:{RESET} {len(baseline)} chars")
    print(f"  {OK}Fine-tuned:{RESET} {len(finetuned)} chars")

    if len(finetuned) < len(baseline) * 0.8:
        print(
            f"\n  {INFO}✓ Fine-tuned response is more concise (~{100 - int(len(finetuned) / len(baseline) * 100)}% shorter){RESET}"
        )
    else:
        print(
            f"\n  {MUTED}Response lengths are similar - check content for style differences{RESET}"
        )


def cleanup_adapter(registry: AdapterRegistry):
    """Clean up the demo adapter."""
    print(f"\n{H2}▶ Cleanup{RESET}")
    try:
        registry.remove(ADAPTER_ID)
        registry.refresh()
        print(f"  {OK}✓ Removed adapter: {ADAPTER_ID}{RESET}")
    except Exception as e:
        print(f"  {MUTED}Cleanup skipped: {e}{RESET}")


async def main():
    """Run the complete LoRA training workflow."""
    print(f"\n{H1}{'━' * 60}{RESET}")
    print(f"{H1}  Example 04: LoRA Fine-Tuning with Before/After Comparison{RESET}")
    print(f"{H1}{'━' * 60}{RESET}")

    # Suppress logging noise
    lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))

    # Initialize
    config = Config("etc/llm-learn.yaml")
    learn = LearnClient(profile_id=1)
    learn.migrate()
    profile_id = ensure_demo_profile(learn, profile_slug="lora-training")
    learn = LearnClient(profile_id=profile_id)
    print(f"{MUTED}Using profile_id={RESET}{INFO}{profile_id}{RESET}")

    # Get inference URL and query for running model
    infer_url = get_infer_url(config)
    running_model = get_running_model(infer_url)
    try:
        training_model_path = find_training_model(lg, config, running_model)
    except RuntimeError as e:
        lg.warning(
            "cannot run LoRA training example",
            extra={
                "exception": e,
                "running_model": running_model,
                "infer_url": infer_url,
            },
        )
        print(f"\n{WARN}Cannot run LoRA training example{RESET}")
        print("\nLoRA training requires local model weights in HuggingFace format.")
        print(f"The inference server reports model '{running_model}', but no matching")
        print("local model was found for training.")
        print(f"\n{MUTED}This can happen when:{RESET}")
        print("  • Using Ollama (serves GGUF models, not trainable)")
        print("  • Model weights not downloaded locally")
        print("  • Model location not configured in etc/models.yaml")
        print(f"\n{INFO}To run this example:{RESET}")
        print(f"  1. Download HuggingFace model weights for '{running_model}'")
        print("  2. Configure the model location in etc/models.yaml")
        print("  3. Run llm-infer with --model pointing to the local weights")
        raise SystemExit(1)
    adapter_base_path = Path(config.adapters.lora.base_path)
    print(f"{MUTED}Running model:{RESET} {INFO}{running_model}{RESET}")
    print(f"{MUTED}Training model:{RESET} {INFO}{training_model_path}{RESET}")
    print(f"{MUTED}Infer URL:{RESET} {INFO}{infer_url}{RESET}")
    print(f"{MUTED}Adapter path:{RESET} {INFO}{adapter_base_path}{RESET}")

    # Create LLM client
    llm_client = LLMClient.from_config(config.llm.to_dict())

    try:
        # Step 1: Show baseline
        baseline_response = await show_baseline(llm_client)

        # Step 2: Create and export training data
        create_training_data(learn)

        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Step 3: Export training data
            data_path = export_training_data(learn, output_dir)

            # Step 4: Train adapter
            training_result = run_training(
                lg, data_path, output_dir / "adapter", str(training_model_path)
            )

            # Step 5: Register with llm-infer
            registry = register_adapter(lg, training_result, infer_url, adapter_base_path)

            # Step 6: Show fine-tuned response
            finetuned_response = await show_finetuned(llm_client)

            # Step 7: Compare
            show_comparison(baseline_response, finetuned_response)

            # Step 8: Cleanup (optional - comment out to keep adapter)
            cleanup_adapter(registry)

    finally:
        await llm_client.aclose()

    print(f"\n{H1}{'━' * 60}{RESET}")
    print(f"{OK}✓ Done!{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
