"""LoRA training using supervised fine-tuning (SFT).

Uses TRL's SFTTrainer with PEFT for parameter-efficient fine-tuning.
Supports QLoRA (4-bit quantization) to reduce VRAM requirements.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from ..core.utils import utc_now
from .config import LoraConfig, TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _load_sft_dataset(data_path: Path, eval_split: float):
    """Load and optionally split SFT dataset from JSONL file."""
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


def _format_sft_example(example: dict) -> str:
    """Format a single SFT example for training."""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    return f"### Instruction:\n{user_content}\n\n### Response:\n{output}"


class LoraTrainer:
    """Trainer for LoRA adapters using supervised fine-tuning."""

    def __init__(
        self,
        data_path: str | Path,
        output_dir: str | Path,
        base_model: str = DEFAULT_BASE_MODEL,
        lora_config: LoraConfig | None = None,
        training_config: TrainingConfig | None = None,
        quantize: bool = True,
    ):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.base_model = base_model
        self.lora_config = lora_config or LoraConfig()
        self.training_config = training_config or TrainingConfig()
        self.quantize = quantize

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None

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

    def _load_model(self):
        """Load base model with optional quantization and apply LoRA."""
        import torch
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading base model: {self.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        quant_config = self._setup_quantization_config()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if quant_config is None else None,
            trust_remote_code=True,
        )

        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.quantize:
            self.model = prepare_model_for_kbit_training(self.model)

        peft_config = self.lora_config.to_peft_config()
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def _load_data(self):
        """Load training and eval datasets."""
        self.train_dataset, self.eval_dataset = _load_sft_dataset(
            self.data_path, self.training_config.eval_split
        )
        logger.info(f"Loaded {len(self.train_dataset)} training samples")
        if self.eval_dataset:
            logger.info(f"Loaded {len(self.eval_dataset)} eval samples")

    def _create_training_args(self):
        """Create training arguments for SFT."""
        from transformers import TrainingArguments

        return TrainingArguments(
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
        )

    def _create_trainer(self):
        """Create the SFTTrainer."""
        from trl import SFTTrainer

        self.trainer = SFTTrainer(
            model=self.model,
            args=self._create_training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            formatting_func=_format_sft_example,
            max_seq_length=self.training_config.max_seq_length,
        )

    def _save_and_collect_metrics(self) -> tuple[Path, dict]:
        """Save adapter and collect training metrics."""
        assert self.trainer is not None
        assert self.tokenizer is not None

        final_path = self.output_dir / "final"
        self.trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        logger.info(f"Saved adapter to {final_path}")

        metrics = {"train_loss": self.trainer.state.log_history[-1].get("train_loss", 0.0)}
        if self.eval_dataset:
            metrics["eval_loss"] = self.trainer.evaluate().get("eval_loss", 0.0)

        return final_path, metrics

    def _build_result(
        self, adapter_path: Path, metrics: dict, started_at: datetime
    ) -> TrainingResult:
        """Build the training result."""
        return TrainingResult(
            adapter_path=adapter_path,
            base_model=self.base_model,
            method="lora",
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
                "quantize": self.quantize,
            },
            started_at=started_at,
            completed_at=utc_now(),
            samples_trained=len(self.train_dataset) * self.training_config.num_epochs,  # type: ignore[arg-type]
        )

    def train(self, resume_from: str | Path | None = None) -> TrainingResult:
        """Run the full training pipeline."""
        started_at = utc_now()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting LoRA training: {self.data_path} -> {self.output_dir}")

        self._load_data()
        self._load_model()
        self._create_trainer()
        assert self.trainer is not None

        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            self.trainer.train(resume_from_checkpoint=str(resume_from))
        else:
            self.trainer.train()

        adapter_path, metrics = self._save_and_collect_metrics()
        return self._build_result(adapter_path, metrics, started_at)


def train_lora(
    data_path: str | Path,
    output_dir: str | Path,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_config: LoraConfig | None = None,
    training_config: TrainingConfig | None = None,
    quantize: bool = True,
    resume_from: str | Path | None = None,
) -> TrainingResult:
    """Train a LoRA adapter using supervised fine-tuning.

    Args:
        data_path: Path to JSONL file with {"instruction", "output", "input"?} records.
        output_dir: Directory to save the trained adapter.
        base_model: HuggingFace model ID. Defaults to Qwen2.5-7B-Instruct.
        lora_config: LoRA hyperparameters. Uses sensible defaults if not provided.
        training_config: Training hyperparameters. Uses sensible defaults if not provided.
        quantize: Use 4-bit quantization (QLoRA). Reduces VRAM ~4x.
        resume_from: Path to checkpoint to resume training from.

    Returns:
        TrainingResult with adapter path, metrics, and training metadata.
    """
    trainer = LoraTrainer(
        data_path=data_path,
        output_dir=output_dir,
        base_model=base_model,
        lora_config=lora_config,
        training_config=training_config,
        quantize=quantize,
    )
    return trainer.train(resume_from=resume_from)
