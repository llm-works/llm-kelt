"""DPO (Direct Preference Optimization) training.

Uses TRL's DPOTrainer with PEFT for preference-based fine-tuning.
Trains the model to prefer "chosen" responses over "rejected" ones.
"""

import json
from datetime import datetime
from pathlib import Path

from appinfra.log import Logger

from llm_learn.core.base import utc_now

from ..config import RunConfig, RunResult
from ..lora import Config as LoraConfig

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
        training_config: RunConfig | None = None,
        beta: float = 0.1,
        quantize: bool = True,
        reference_free: bool = False,
        based_on: Path | None = None,
    ):
        self._lg = lg
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.base_model = base_model
        self.lora_config = lora_config or LoraConfig()
        self.training_config = training_config or RunConfig()
        self.beta = beta
        self.quantize = quantize
        self.reference_free = reference_free
        self.based_on = based_on
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

    def _setup_quantization_config(self):
        """Create BitsAndBytes config for 4-bit quantization."""
        if not self.quantize:
            return None

        import torch
        from transformers import BitsAndBytesConfig

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

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

        if self.based_on is not None:
            self._lg.info(f"loading existing adapter: {self.based_on}")
            self.model = PeftModel.from_pretrained(
                self.model, str(self.based_on), is_trainable=True
            )
        else:
            peft_config = self.lora_config.to_peft_config()
            self.model = get_peft_model(self.model, peft_config)

        trainable, total = self.model.get_nb_trainable_parameters()
        self._lg.info(
            f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)"
        )

    def _load_model(self):
        """Load base model with optional quantization and apply LoRA."""
        import torch
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM

        self._lg.info(f"loading base model: {self.base_model}")
        self._load_tokenizer()

        self._quant_config = self._setup_quantization_config()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=self._quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if self._quant_config is None else None,
            trust_remote_code=True,
        )

        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self.quantize:
            self.model = prepare_model_for_kbit_training(self.model)

        self._apply_lora_adapter()

    def _load_reference_model(self):
        """Load reference model for DPO (unless reference-free mode).

        DPO requires comparing policy model outputs against a reference model.
        Loading a separate reference model uses more VRAM but produces correct
        gradients. The reference_free option skips this to save memory at the
        cost of training quality.
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

    def _collect_metrics(self) -> dict:
        """Extract training metrics from trainer log history."""
        if self.trainer is None:
            return {}

        metrics: dict = {}
        if not self.trainer.state.log_history:
            return metrics

        # Find logs with DPO metrics (exclude final summary which only has train_loss)
        dpo_logs = [log for log in self.trainer.state.log_history if "rewards/accuracies" in log]
        if dpo_logs:
            first_log, last_log = dpo_logs[0], dpo_logs[-1]
            # Final metrics (most important)
            metrics["accuracy"] = last_log.get("rewards/accuracies", 0.0)
            metrics["margins"] = last_log.get("rewards/margins", 0.0)
            metrics["loss"] = last_log.get("loss", 0.0)
            # Starting metrics (for comparison)
            metrics["accuracy_start"] = first_log.get("rewards/accuracies", 0.0)
            metrics["margins_start"] = first_log.get("rewards/margins", 0.0)
            metrics["loss_start"] = first_log.get("loss", 0.0)

        # Overall train loss from final summary
        final_log = self.trainer.state.log_history[-1]
        metrics["train_loss"] = final_log.get("train_loss", final_log.get("loss", 0.0))
        metrics["epochs"] = final_log.get("epoch", 0.0)

        if self.eval_dataset:
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

    def _build_result(
        self, adapter_path: Path, metrics: dict, started_at: datetime
    ) -> RunResult:
        """Build the training result."""
        return RunResult(
            adapter_path=adapter_path,
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
                "quantize": self.quantize,
            },
            started_at=started_at,
            completed_at=utc_now(),
            samples_trained=len(self.train_dataset) * self.training_config.num_epochs,  # type: ignore[arg-type]
            based_on=self.based_on,
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
    training_config: RunConfig | None = None,
    beta: float = 0.1,
    quantize: bool = True,
    reference_free: bool = False,
    based_on: Path | None = None,
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
        quantize: Use 4-bit quantization (QLoRA). Reduces VRAM ~4x.
        reference_free: Skip reference model to save VRAM (may reduce quality).
        based_on: Path to existing adapter to train on top of (for lineage).

    Returns:
        RunResult with adapter path, metrics, and training metadata.
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
        based_on=based_on,
    )
    return trainer.train()
