"""DPO (Direct Preference Optimization) training.

Uses TRL's DPOTrainer with PEFT for preference-based fine-tuning.
Trains the model to prefer "chosen" responses over "rejected" ones.
"""

import json
from datetime import datetime
from pathlib import Path

from appinfra import DotDict
from appinfra.log import Logger

from llm_learn.core.base import utc_now

from ..lora import Config as LoraConfig
from ..model import build_training_config
from ..schema import Adapter, RunResult

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _load_dpo_dataset(data_path: Path, eval_split: float):
    """Load and optionally split DPO dataset from JSONL file."""
    from datasets import Dataset

    records = []
    with data_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

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
        reference_free: bool = False,
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
        self.reference_free = reference_free
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
        self._applied_quantization = False

    def _load_tokenizer(self):
        """Load tokenizer for the base model."""
        from transformers import AutoTokenizer

        # trust_remote_code required for Qwen and other models with custom architectures
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_lora_adapter(self):
        """Apply LoRA adapter - load existing or create new."""
        from peft import PeftModel, get_peft_model

        if self.parent is not None:
            self._lg.info(f"loading existing adapter: {self.parent.path}")
            self.model = PeftModel.from_pretrained(self.model, self.parent.path, is_trainable=True)
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

        self._quant_config, self._applied_quantization = get_quantization_config(
            self._lg, self.base_model, self._quantize_override
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self._quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self._quant_config is None else None,
            trust_remote_code=True,
        )

        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self._applied_quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        self._apply_lora_adapter()

    def _load_reference_model(self):
        """Load reference model for DPO (unless reference-free mode).

        DPO requires comparing policy model outputs against a reference model.
        Loading a separate reference model uses more VRAM but produces correct
        gradients. The reference_free option skips this to save memory at the
        cost of training quality.

        IMPORTANT: When training on top of an existing adapter (parent),
        the reference model MUST also have that adapter applied (frozen).
        Otherwise DPO compares against the wrong baseline.
        """
        if self.reference_free:
            self._lg.info("reference-free mode: skipping reference model")
            return

        import torch
        from transformers import AutoModelForCausalLM

        self._lg.info("loading reference model for DPO")
        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self._quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self._quant_config is None else None,
            trust_remote_code=True,
        )

        # Apply the same adapter to reference model (frozen) if training on top of one
        if self.parent is not None:
            from peft import PeftModel

            self._lg.info(f"applying adapter to reference model (frozen): {self.parent.path}")
            self.ref_model = PeftModel.from_pretrained(
                self.ref_model, self.parent.path, is_trainable=False
            )

    def _load_data(self):
        """Load training and eval datasets."""
        self.train_dataset, self.eval_dataset = _load_dpo_dataset(
            self.data_path, self.training_config.eval_split
        )
        self._lg.info(f"loaded {len(self.train_dataset)} training preference pairs")
        if self.eval_dataset:
            self._lg.info(f"loaded {len(self.eval_dataset)} eval pairs")

    def _create_training_args(self):
        """Create DPO training arguments."""
        from trl import DPOConfig

        return DPOConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            warmup_ratio=self.training_config.warmup_ratio,
            logging_steps=self.training_config.logging_steps,
            save_steps=self.training_config.save_steps,
            save_total_limit=2,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            seed=self.training_config.seed,
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=self.training_config.save_steps if self.eval_dataset else None,
            load_best_model_at_end=self.eval_dataset is not None,
            report_to="none",
            optim="paged_adamw_8bit",
            beta=self.beta,
            max_length=self.training_config.max_seq_length,
            max_prompt_length=self.training_config.max_seq_length // 2,
        )

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

    def _collect_metrics(self) -> dict:
        """Extract training metrics from trainer log history."""
        if self.trainer is None or not self.trainer.state.log_history:
            return {}

        log_history = self.trainer.state.log_history
        # Cap history to avoid bloating manifests (same as SFT trainer)
        capped_history = log_history[-100:]
        metrics: dict = {"history": capped_history}

        # DPO-specific metrics (exclude final summary which only has train_loss)
        dpo_logs = [log for log in capped_history if "rewards/accuracies" in log]
        metrics.update(self._extract_dpo_metrics(dpo_logs))

        # Overall train loss from final summary
        final_log = log_history[-1]
        metrics["train_loss"] = final_log.get("train_loss", final_log.get("loss", 0.0))
        metrics["train_runtime"] = final_log.get("train_runtime", 0.0)
        metrics["train_samples_per_second"] = final_log.get("train_samples_per_second", 0.0)
        metrics["epoch"] = final_log.get("epoch", 0.0)

        if self.eval_dataset:
            # Runs full evaluation pass - adds GPU overhead but provides eval_loss metric
            metrics["eval_loss"] = self.trainer.evaluate().get("eval_loss", 0.0)

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
                },
                "training": {
                    "num_epochs": self.training_config.num_epochs,
                    "batch_size": self.training_config.batch_size,
                    "learning_rate": self.training_config.learning_rate,
                    "max_seq_length": self.training_config.max_seq_length,
                },
                "dpo": {"beta": self.beta, "reference_free": self.reference_free},
                "quantized": self._applied_quantization,
            },
            started_at=started_at,
            completed_at=utc_now(),
            samples_trained=int(len(self.train_dataset) * self.training_config.num_epochs),  # type: ignore[arg-type]
            adapter=adapter,
            parent=self.parent,
        )

    def train(self) -> RunResult:
        """Run the full training pipeline."""
        started_at = utc_now()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lg.info(f"starting DPO training: {self.data_path} -> {self.output_dir}")

        self._load_data()
        self._load_model()
        self._load_reference_model()
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
    reference_free: bool = False,
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
        reference_free: Skip reference model to save VRAM (may reduce quality).
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
        reference_free=reference_free,
        parent=parent,
    )
    return trainer.train()
