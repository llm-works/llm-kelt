"""DPO (Direct Preference Optimization) training.

Uses TRL's DPOTrainer with PEFT for preference-based fine-tuning.
Trains the model to prefer "chosen" responses over "rejected" ones.
"""

import json
import math
from datetime import UTC, datetime
from pathlib import Path

from appinfra import DotDict
from appinfra.log import Logger

from ..lora import Config as LoraConfig
from ..model import build_training_config
from ..schema import TRAINING_CONFIG_KEYS, Adapter, RunResult
from ..stability import check_training_stability, log_stability_warnings

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _to_chat_format(record: dict) -> dict:
    """Convert raw prompt/chosen/rejected strings to chat message format.

    TRL's DPOTrainer expects message lists for proper chat template handling:
    - prompt: list of user messages
    - chosen: list of assistant messages (the preferred response)
    - rejected: list of assistant messages (the rejected response)

    This ensures consistent tokenization when DPOTrainer compares
    prompt vs prompt+response.
    """
    return {
        "prompt": [{"role": "user", "content": record["prompt"]}],
        "chosen": [{"role": "assistant", "content": record["chosen"]}],
        "rejected": [{"role": "assistant", "content": record["rejected"]}],
    }


def _load_dpo_dataset(data_path: Path, eval_split: float):
    """Load and optionally split DPO dataset from JSONL file."""
    from datasets import Dataset

    records = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw = json.loads(line)
                records.append(_to_chat_format(raw))

    if not records:
        raise ValueError(f"No records found in {data_path}")

    dataset = Dataset.from_list(records)

    if eval_split > 0:
        split = dataset.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]

    return dataset, None


class Trainer:
    """Trainer for LoRA adapters using Direct Preference Optimization."""

    def __init__(
        self,
        lg: Logger,
        data_path: str | Path,
        output_dir: str | Path,
        base_model: str = DEFAULT_BASE_MODEL,
        lora_config: LoraConfig | None = None,
        training_config: DotDict | None = None,
        beta: float = 0.1,
        quantize: bool | None = None,
        parent: Adapter | None = None,
    ):
        self._lg = lg
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.base_model = base_model
        self.lora_config = lora_config or LoraConfig()
        self.training_config = build_training_config(lg, base_model, training_config)
        self.beta = beta
        self._quantize_override = quantize  # None = auto-detect
        self.parent = parent
        self._init_state()

    def _init_state(self):
        """Initialize mutable state to None (set during training)."""
        self.model = None
        self.ref_model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None
        self._quant_config = None
        self._is_quantized = False
        self._use_stacked_adapters = False  # True when parent adapter is stacked with DPO

    def _load_tokenizer(self):
        """Load tokenizer for the base model."""
        from transformers import AutoTokenizer

        # trust_remote_code required for Qwen and other models with custom architectures
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_lora_adapter(self):
        """Apply LoRA adapter - load existing or create new.

        When parent adapter exists, uses adapter copying for VRAM efficiency:
        - Parent loaded as "default" adapter (policy, trainable)
        - Copy created as "ref" adapter (reference, frozen)
        - TRL automatically uses "ref" for reference logits
        - Single model, no second model needed
        """
        from peft import PeftModel, get_peft_model

        if self.parent is not None:
            # Load parent as trainable "default" (policy)
            self._lg.info(f"loading parent adapter as 'default' (trainable): {self.parent.path}")
            self.model = PeftModel.from_pretrained(
                self.model, self.parent.path, adapter_name="default", is_trainable=True
            )

            # Copy to frozen "ref" for TRL's reference logits
            self._lg.info("copying to 'ref' adapter (frozen) for reference")
            self.model.load_adapter(self.parent.path, adapter_name="ref", is_trainable=False)
            self._use_stacked_adapters = True
            self.model.set_adapter("default")  # Ensure default is active
        else:
            peft_config = self.lora_config.to_peft_config()
            self.model = get_peft_model(self.model, peft_config)

        trainable, total = self.model.get_nb_trainable_parameters()
        self._lg.info(
            f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )

    def _load_model(self):
        """Load base model with auto-detected quantization and apply LoRA."""
        import torch
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM

        from ..model import get_quantization_config

        self._lg.info(f"loading base model: {self.base_model}")
        self._load_tokenizer()

        self._quant_config, _ = get_quantization_config(
            self._lg, self.base_model, self._quantize_override
        )
        self._is_quantized = self._quant_config is not None
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self._quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self._quant_config is None else None,
            trust_remote_code=True,
        )

        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self._is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)

        self._apply_lora_adapter()

    def _log_reference_strategy(self):
        """Log reference model strategy for DPO.

        - Parent adapter exists: "ref" adapter already created in _apply_lora_adapter().
          TRL will automatically use it for reference logits. No second model needed.
        - No parent: TRL uses implicit reference via disable_adapter() automatically.
        """
        if self._use_stacked_adapters:
            self._lg.info("using 'ref' adapter for reference (frozen copy of parent)")
        else:
            self._lg.info("using implicit reference via disable_adapter()")

    def _load_data(self):
        """Load training and eval datasets."""
        self.train_dataset, self.eval_dataset = _load_dpo_dataset(
            self.data_path, self.training_config.eval_split
        )
        self._lg.info(f"loaded {len(self.train_dataset)} training preference pairs")
        if self.eval_dataset:
            self._lg.info(f"loaded {len(self.eval_dataset)} eval pairs")

    def _calculate_warmup_steps(self) -> int:
        """Calculate warmup steps from warmup ratio and training config."""
        tc = self.training_config
        if tc.batch_size <= 0 or tc.gradient_accumulation_steps <= 0:
            raise ValueError("batch_size and gradient_accumulation_steps must be positive")

        steps_per_epoch = max(
            1, math.ceil(len(self.train_dataset) / (tc.batch_size * tc.gradient_accumulation_steps))
        )
        total_steps = steps_per_epoch * tc.num_epochs
        warmup_steps = int(total_steps * tc.warmup_ratio)
        # Ensure at least 1 warmup step when ratio > 0
        if tc.warmup_ratio > 0 and warmup_steps == 0:
            warmup_steps = 1
        return warmup_steps

    def _base_training_args(self) -> dict:
        """Build base training arguments from config."""
        tc = self.training_config
        return dict(
            output_dir=str(self.output_dir),
            num_train_epochs=tc.num_epochs,
            per_device_train_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            warmup_steps=self._calculate_warmup_steps(),
            max_grad_norm=tc.max_grad_norm,
            logging_steps=tc.logging_steps,
            save_steps=tc.save_steps,
            save_total_limit=2,
            fp16=tc.fp16,
            bf16=tc.bf16,
            gradient_checkpointing=tc.gradient_checkpointing,
            seed=tc.seed,
            report_to="none",
            optim="paged_adamw_8bit",
            max_length=tc.max_seq_length,
            neftune_noise_alpha=tc.neftune_noise_alpha,
        )

    def _create_training_args(self):
        """Create DPO training arguments."""
        from trl import DPOConfig

        config_args = self._base_training_args()
        config_args.update(
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=self.training_config.save_steps if self.eval_dataset else None,
            load_best_model_at_end=self.eval_dataset is not None,
            beta=self.beta,
        )
        if self._use_stacked_adapters:
            config_args["model_adapter_name"] = "default"
            config_args["ref_adapter_name"] = "ref"

        return DPOConfig(**config_args)

    def _create_trainer(self):
        """Create the DPOTrainer."""
        from trl import DPOTrainer

        self.trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=self._create_training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

        # Use implicit reference via disable_adapter() when no parent adapter
        # (TRL automatically uses "ref" adapter when it exists, so no action needed
        # for stacked adapters case)
        if self.ref_model is None and self.parent is None:
            self._enable_implicit_reference()

    def _enable_implicit_reference(self):
        """Enable implicit reference by removing TRL's 'ref' adapter.

        When ref_model=None with a PeftModel, TRL creates a 'ref' adapter as a copy
        of the 'default' adapter. During training, TRL uses this frozen 'ref' adapter
        for reference logits.

        For DPO on a merged model (SFT baked into base), we want the reference to be
        the base model itself, not a copy of fresh adapter weights. By removing the
        'ref' adapter, TRL falls back to using disable_adapter() which gives us the
        correct reference: the base model with any merged weights.

        See TRL's use_adapter() in trainer/utils.py - when adapter_name=None,
        it calls model.disable_adapter() as a context manager.
        """
        from peft import PeftModel

        if not isinstance(self.model, PeftModel):
            return

        # Derive ref adapter name from trainer config, fallback to "ref"
        ref_name = None
        if self.trainer is not None and hasattr(self.trainer, "ref_adapter_name"):
            ref_name = self.trainer.ref_adapter_name
        ref_name = ref_name or "ref"

        # Delete the ref adapter if it exists
        if ref_name in self.model.peft_config:
            self._lg.info(f"using implicit reference: removing '{ref_name}' adapter")
            self.model.delete_adapter(ref_name)

        # Always clear trainer binding so TRL uses disable_adapter() path
        if self.trainer is not None and hasattr(self.trainer, "ref_adapter_name"):
            self.trainer.ref_adapter_name = None

    def _extract_dpo_metrics(self, dpo_logs: list) -> dict:
        """Extract DPO-specific metrics (accuracy, margins, loss) from first/last logs."""
        if not dpo_logs:
            return {}
        first_log, last_log = dpo_logs[0], dpo_logs[-1]
        return {
            "accuracy": last_log.get("rewards/accuracies", 0.0),
            "margins": last_log.get("rewards/margins", 0.0),
            "loss": last_log.get("loss", 0.0),
            "accuracy_start": first_log.get("rewards/accuracies", 0.0),
            "margins_start": first_log.get("rewards/margins", 0.0),
            "loss_start": first_log.get("loss", 0.0),
        }

    def _add_stability_warnings(self, metrics: dict, log_history: list[dict]) -> None:
        """Check for training instability and add warnings to metrics."""
        stability_report = check_training_stability(log_history)
        log_stability_warnings(self._lg, stability_report)
        if not stability_report.stable:
            metrics["unstable"] = True
            metrics["stability_warnings"] = stability_report.warnings

    def _collect_metrics(self) -> dict:
        """Extract training metrics from trainer log history."""
        if self.trainer is None or not self.trainer.state.log_history:
            return {}

        log_history = self.trainer.state.log_history
        capped_history = log_history[-100:]
        metrics: dict = {"history": capped_history}

        dpo_logs = [log for log in capped_history if "rewards/accuracies" in log]
        metrics.update(self._extract_dpo_metrics(dpo_logs))

        final_log = log_history[-1]
        metrics["train_loss"] = final_log.get("train_loss", final_log.get("loss", 0.0))
        metrics["train_runtime"] = final_log.get("train_runtime", 0.0)
        metrics["train_samples_per_second"] = final_log.get("train_samples_per_second", 0.0)
        metrics["epoch"] = final_log.get("epoch", 0.0)

        if self.eval_dataset:
            metrics["eval_loss"] = self.trainer.evaluate().get("eval_loss", 0.0)
            # Re-capture history to include eval entries added by evaluate()
            metrics["history"] = self.trainer.state.log_history[-100:]

        self._add_stability_warnings(metrics, log_history)
        return metrics

    def _save_and_collect_metrics(self) -> tuple[Path, dict]:
        """Save adapter and collect training metrics."""
        if self.trainer is None:
            raise RuntimeError("_create_trainer() must be called first")
        if self.tokenizer is None:
            raise RuntimeError("_load_model() must be called first")

        final_path = self.output_dir / "final"
        self.trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        self._lg.info(f"saved adapter to {final_path}")

        return final_path, self._collect_metrics()

    def _build_result(self, adapter_path: Path, metrics: dict, started_at: datetime) -> RunResult:
        """Build the training result."""
        # Adapter md5/mtime populated by caller (runner) after training
        adapter = Adapter(md5="", mtime="", path=str(adapter_path))
        return RunResult(
            status="completed",
            base_model=self.base_model,
            method="dpo",
            metrics=metrics,
            config={
                "lora": {
                    "r": self.lora_config.r,
                    "lora_alpha": self.lora_config.lora_alpha,
                    "lora_dropout": self.lora_config.lora_dropout,
                    "target_modules": self.lora_config.target_modules,
                    "use_rslora": self.lora_config.use_rslora,
                },
                "training": {k: self.training_config[k] for k in TRAINING_CONFIG_KEYS},
                "dpo": {"beta": self.beta},
                "quantized": self._is_quantized,
            },
            started_at=started_at,
            completed_at=datetime.now(UTC),
            samples_trained=int(len(self.train_dataset) * self.training_config.num_epochs),  # type: ignore[arg-type]
            adapter=adapter,
            parent=self.parent,
        )

    def train(self) -> RunResult:
        """Run the full training pipeline."""
        started_at = datetime.now(UTC)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lg.info(f"starting DPO training: {self.data_path} -> {self.output_dir}")

        self._load_data()
        self._load_model()
        self._log_reference_strategy()
        self._create_trainer()
        if self.trainer is None:
            raise RuntimeError("Failed to create trainer")

        self.trainer.train()

        adapter_path, metrics = self._save_and_collect_metrics()
        return self._build_result(adapter_path, metrics, started_at)


def train_dpo(
    lg: Logger,
    data_path: str | Path,
    output_dir: str | Path,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_config: LoraConfig | None = None,
    training_config: DotDict | None = None,
    beta: float = 0.1,
    quantize: bool | None = None,
    parent: Adapter | None = None,
) -> RunResult:
    """Train a LoRA adapter using Direct Preference Optimization.

    Args:
        lg: Logger instance.
        data_path: Path to JSONL file with {"prompt", "chosen", "rejected"} records.
        output_dir: Directory to save the trained adapter.
        base_model: HuggingFace model ID. Defaults to Qwen2.5-7B-Instruct.
        lora_config: LoRA hyperparameters. Uses sensible defaults if not provided.
        training_config: Training hyperparameters. Uses sensible defaults if not provided.
        beta: DPO beta parameter (higher = more conservative).
        quantize: Force quantization on/off. None = auto-detect from model metadata.
        parent: Parent adapter to train on top of (for lineage).

    Returns:
        RunResult with adapter, metrics, and training metadata.
    """
    trainer = Trainer(
        lg=lg,
        data_path=data_path,
        output_dir=output_dir,
        base_model=base_model,
        lora_config=lora_config,
        training_config=training_config,
        beta=beta,
        quantize=quantize,
        parent=parent,
    )
    return trainer.train()
