"""Prompt tuning using supervised fine-tuning (SFT).

Uses TRL's SFTTrainer with PEFT PromptTuning for extremely parameter-efficient
fine-tuning. Only ~50K parameters are trained (vs ~50M for LoRA r=32).

This is ideal for large models (32B+) with small datasets where LoRA is unstable.
"""

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from appinfra import DotDict
from appinfra.log import Logger

from llm_kelt.core.base import utc_now

from ..model import build_training_config
from ..schema import TRAINING_CONFIG_KEYS, Adapter, RunResult
from ..stability import check_training_stability, log_stability_warnings
from .config import Config as PromptConfig

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def load_sft_dataset(data_path: Path, eval_split: float):
    """Load and optionally split SFT dataset from JSONL file."""
    import json

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
        num_records = len(dataset)
        train_size = int(num_records * (1 - eval_split))
        if train_size < 1:
            # eval_split too large - would leave no training data
            return dataset, None
        split = dataset.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]

    return dataset, None


class Trainer:
    """Trainer for prompt tuning adapters using supervised fine-tuning."""

    def __init__(
        self,
        lg: Logger,
        data_path: str | Path,
        output_dir: str | Path,
        base_model: str = DEFAULT_BASE_MODEL,
        prompt_config: PromptConfig | None = None,
        training_config: DotDict | None = None,
        quantize: bool | None = None,
    ):
        self._lg = lg
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.base_model = base_model
        self.prompt_config = prompt_config or PromptConfig()
        self.training_config = build_training_config(lg, base_model, training_config)
        self._quantize_override = quantize
        self._applied_quantization = False
        self._is_quantized = False

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.train_dataset = None
        self.eval_dataset = None

    def _load_tokenizer(self):
        """Load tokenizer for the base model."""
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _make_formatting_func(self):
        """Create a formatting function using the tokenizer's chat template."""
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError(
                f"Tokenizer for {self.base_model} lacks chat_template. "
                "Training requires a model with chat template support."
            )

        tokenizer = self.tokenizer

        def format_example(example: dict) -> str:
            instruction = example.get("prompt") or example.get("instruction", "")
            input_text = example.get("input", "")
            output = example.get("response") or example.get("output", "")

            user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
            result: str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return result

        return format_example

    def _load_model(self):
        """Load base model and apply prompt tuning."""
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM

        from ..model import get_quantization_config

        self._lg.info(f"loading base model: {self.base_model}")
        self._load_tokenizer()

        quant_config, self._applied_quantization = get_quantization_config(
            self._lg, self.base_model, self._quantize_override
        )
        self._is_quantized = quant_config is not None
        model_kwargs: dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, **model_kwargs)

        if quant_config is not None:
            from peft import prepare_model_for_kbit_training

            self.model = prepare_model_for_kbit_training(self.model)

        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        peft_config = self.prompt_config.to_peft_config(self.base_model)
        self.model = get_peft_model(self.model, peft_config)
        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100 * trainable / total if total > 0 else 0.0
        self._lg.info(f"Trainable: {trainable:,} / {total:,} ({pct:.6f}%)")

    def _load_data(self):
        """Load training and eval datasets."""
        self.train_dataset, self.eval_dataset = load_sft_dataset(
            self.data_path, self.training_config.eval_split
        )
        self._lg.info(f"loaded {len(self.train_dataset)} training samples")
        if self.eval_dataset:
            self._lg.info(f"loaded {len(self.eval_dataset)} eval samples")

    def _create_training_args(self):
        """Create SFTConfig for training."""
        from trl import SFTConfig

        tc = self.training_config
        return SFTConfig(
            output_dir=str(self.output_dir),
            num_train_epochs=tc.num_epochs,
            per_device_train_batch_size=tc.batch_size,
            gradient_accumulation_steps=tc.gradient_accumulation_steps,
            learning_rate=tc.learning_rate,
            warmup_ratio=tc.warmup_ratio,
            max_grad_norm=tc.max_grad_norm,
            logging_steps=tc.logging_steps,
            save_steps=tc.save_steps,
            save_total_limit=2,
            fp16=tc.fp16,
            bf16=tc.bf16,
            gradient_checkpointing=tc.gradient_checkpointing,
            seed=tc.seed,
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=tc.save_steps if self.eval_dataset else None,
            load_best_model_at_end=self.eval_dataset is not None,
            report_to="none",
            optim="paged_adamw_8bit",
            max_length=tc.max_seq_length,
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
            formatting_func=self._make_formatting_func(),
        )

    def _collect_metrics(self) -> dict:
        """Extract training metrics from trainer state."""
        if self.trainer is None or not self.trainer.state.log_history:
            return {}

        metrics: dict = {"history": self.trainer.state.log_history[-100:]}
        final = self.trainer.state.log_history[-1]
        metrics["train_loss"] = final.get("train_loss", 0.0)
        metrics["train_runtime"] = final.get("train_runtime", 0.0)
        metrics["train_samples_per_second"] = final.get("train_samples_per_second", 0.0)
        metrics["epoch"] = final.get("epoch", 0.0)

        stability_report = check_training_stability(self.trainer.state.log_history)
        log_stability_warnings(self._lg, stability_report)
        if not stability_report.stable:
            metrics["unstable"] = True
            metrics["stability_warnings"] = stability_report.warnings

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

        metrics = self._collect_metrics()
        if self.eval_dataset:
            metrics["eval_loss"] = self.trainer.evaluate().get("eval_loss", 0.0)
            metrics["history"] = self.trainer.state.log_history[-100:]

        return final_path, metrics

    def _build_result(self, adapter_path: Path, metrics: dict, started_at: datetime) -> RunResult:
        """Build the training result."""
        # Adapter md5/mtime populated by caller (client) after training
        return RunResult(
            status="completed",
            base_model=self.base_model,
            method="prompt",
            metrics=metrics,
            config={
                "prompt_tuning": asdict(self.prompt_config),
                "training": {k: self.training_config[k] for k in TRAINING_CONFIG_KEYS},
                "quantized": self._is_quantized,
            },
            started_at=started_at,
            completed_at=utc_now(),
            samples_trained=int(len(self.train_dataset) * self.training_config.num_epochs),  # type: ignore[arg-type]
            adapter=Adapter(md5="", mtime="", path=str(adapter_path)),
        )

    def train(self, resume_from: str | Path | None = None) -> RunResult:
        """Run the full training pipeline."""
        started_at = utc_now()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lg.info(f"starting prompt tuning: {self.data_path} -> {self.output_dir}")

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
