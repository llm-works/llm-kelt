"""LoRA training using supervised fine-tuning (SFT).

Uses TRL's SFTTrainer with PEFT for parameter-efficient fine-tuning.
Supports QLoRA (4-bit quantization) to reduce VRAM requirements.
"""

import json
from datetime import datetime
from pathlib import Path

from appinfra import DotDict
from appinfra.log import Logger

from llm_learn.core.base import utc_now

from ..model import build_training_config
from ..schema import Adapter, RunResult
from .config import Config as LoraConfig

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


class Trainer:
    """Trainer for LoRA adapters using supervised fine-tuning."""

    def __init__(
        self,
        lg: Logger,
        data_path: str | Path,
        output_dir: str | Path,
        base_model: str = DEFAULT_BASE_MODEL,
        lora_config: LoraConfig | None = None,
        training_config: DotDict | None = None,
        quantize: bool | None = None,
    ):
        self._lg = lg
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.base_model = base_model
        self.lora_config = lora_config or LoraConfig()
        self.training_config = build_training_config(lg, base_model, training_config)
        self._quantize_override = quantize  # None = auto-detect
        self._applied_quantization = False  # Set during _load_model

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None

    def _load_tokenizer(self):
        """Load tokenizer for the base model."""
        from transformers import AutoTokenizer

        # trust_remote_code required for Qwen and other models with custom architectures
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _apply_lora(self):
        """Apply LoRA adapter to the loaded model."""
        from peft import get_peft_model

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

        quant_config, self._applied_quantization = get_quantization_config(
            self._lg, self.base_model, self._quantize_override
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16 if quant_config is None else None,
            trust_remote_code=True,
        )

        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self._applied_quantization:
            self.model = prepare_model_for_kbit_training(self.model)

        self._apply_lora()

    def _load_data(self):
        """Load training and eval datasets."""
        self.train_dataset, self.eval_dataset = _load_sft_dataset(
            self.data_path, self.training_config.eval_split
        )
        self._lg.info(f"loaded {len(self.train_dataset)} training samples")
        if self.eval_dataset:
            self._lg.info(f"loaded {len(self.eval_dataset)} eval samples")

    def _create_training_args(self):
        """Create training arguments for SFT."""
        from trl import SFTConfig

        return SFTConfig(
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
            max_length=self.training_config.max_seq_length,
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
        )

    def _save_and_collect_metrics(self) -> tuple[Path, dict]:
        """Save adapter and collect training metrics."""
        if self.trainer is None:
            raise RuntimeError(
                "_create_trainer() must be called before _save_and_collect_metrics()"
            )
        if self.tokenizer is None:
            raise RuntimeError("_load_model() must be called before _save_and_collect_metrics()")

        final_path = self.output_dir / "final"
        self.trainer.save_model(str(final_path))
        self.tokenizer.save_pretrained(str(final_path))
        self._lg.info(f"saved adapter to {final_path}")

        metrics: dict = {}
        if self.trainer.state.log_history:
            # Capture full training history
            metrics["history"] = self.trainer.state.log_history
            # Extract final summary metrics
            final = self.trainer.state.log_history[-1]
            metrics["train_loss"] = final.get("train_loss", 0.0)
            metrics["train_runtime"] = final.get("train_runtime", 0.0)
            metrics["train_samples_per_second"] = final.get("train_samples_per_second", 0.0)
            metrics["epoch"] = final.get("epoch", 0.0)

        if self.eval_dataset:
            metrics["eval_loss"] = self.trainer.evaluate().get("eval_loss", 0.0)

        return final_path, metrics

    def _build_result(self, adapter_path: Path, metrics: dict, started_at: datetime) -> RunResult:
        """Build the training result."""
        # Adapter md5/mtime populated by caller (runner) after training
        adapter = Adapter(md5="", mtime="", path=str(adapter_path))
        return RunResult(
            status="completed",
            base_model=self.base_model,
            method="sft",
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
                "quantized": self._applied_quantization,
            },
            started_at=started_at,
            completed_at=utc_now(),
            samples_trained=len(self.train_dataset) * self.training_config.num_epochs,  # type: ignore[arg-type]
            adapter=adapter,
        )

    def train(self, resume_from: str | Path | None = None) -> RunResult:
        """Run the full training pipeline."""
        started_at = utc_now()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lg.info(f"starting LoRA training: {self.data_path} -> {self.output_dir}")

        self._load_data()
        self._load_model()
        self._create_trainer()
        if self.trainer is None:
            raise RuntimeError("Failed to create trainer")

        if resume_from:
            self._lg.info(f"resuming from checkpoint: {resume_from}")
            self.trainer.train(resume_from_checkpoint=str(resume_from))
        else:
            self.trainer.train()

        adapter_path, metrics = self._save_and_collect_metrics()
        return self._build_result(adapter_path, metrics, started_at)


def train_lora(
    lg: Logger,
    data_path: str | Path,
    output_dir: str | Path,
    base_model: str = DEFAULT_BASE_MODEL,
    lora_config: LoraConfig | None = None,
    training_config: DotDict | None = None,
    quantize: bool | None = None,
    resume_from: str | Path | None = None,
) -> RunResult:
    """Train a LoRA adapter using supervised fine-tuning.

    Args:
        lg: Logger instance.
        data_path: Path to JSONL file with {"instruction", "output", "input"?} records.
        output_dir: Directory to save the trained adapter.
        base_model: HuggingFace model ID. Defaults to Qwen2.5-7B-Instruct.
        lora_config: LoRA hyperparameters. Uses sensible defaults if not provided.
        training_config: Training hyperparameters. Uses sensible defaults if not provided.
        quantize: Force quantization on/off. None = auto-detect from model metadata.
        resume_from: Path to checkpoint to resume training from.

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
        quantize=quantize,
    )
    return trainer.train(resume_from=resume_from)
