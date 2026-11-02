# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Pipeline the trainers share: load, configure, fit, save the adapter, report the run."""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from appinfra import DotDict
from appinfra.log import Logger

from ..lora.config import Config as LoraConfig
from ..schema import TRAINING_CONFIG_KEYS, Adapter, RunResult
from ..stability import check_training_stability, log_stability_warnings
from .dataset import load_sft_dataset, require_chat_template
from .model import build_training_config, compute_warmup_steps


def lora_config_summary(config: LoraConfig) -> dict[str, Any]:
    """LoRA hyperparameters a run records in its config."""
    return {
        "r": config.r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": config.target_modules,
        "exclude_modules": config.exclude_modules,
        "use_rslora": config.use_rslora,
    }


class BaseTrainer:
    """Steps every trainer shares.

    Subclasses load the data and the model, create the TRL trainer and describe the run's
    config; ``_train`` runs them in order.
    """

    METHOD: str  # RunResult.method
    LABEL: str  # names the run in the start log line
    parent: Adapter | None = None  # adapter trained on top of, recorded in the result

    def __init__(
        self,
        lg: Logger,
        data_path: str | Path,
        output_dir: str | Path,
        base_model: str,
        training_config: DotDict | None,
        quantize: bool | None,
    ):
        self._lg = lg
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.base_model = base_model
        self.training_config = build_training_config(lg, base_model, training_config)
        self._quantize_override = quantize  # None = auto-detect
        self._init_state()

    def _init_state(self) -> None:
        """Initialize mutable state to None (set during training)."""
        self.model: Any = None
        self.tokenizer: Any = None
        self.trainer: Any = None
        self.train_dataset: Any = None
        self.eval_dataset: Any = None
        self._is_quantized = False  # True if model uses quantization (new or pre-existing)

    def _load_data(self) -> None:
        """Load training and eval datasets into train_dataset / eval_dataset."""
        raise NotImplementedError

    def _load_model(self) -> None:
        """Load the base model and attach the adapter being trained."""
        raise NotImplementedError

    def _create_trainer(self) -> None:
        """Create the TRL trainer into self.trainer."""
        raise NotImplementedError

    def _run_config(self) -> dict[str, Any]:
        """Hyperparameters the run records in its result."""
        raise NotImplementedError

    def _load_tokenizer(self) -> None:
        """Load tokenizer for the base model."""
        from transformers import AutoTokenizer

        # trust_remote_code required for Qwen and other models with custom architectures
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_base_model(self, dtype: Any = None) -> None:
        """Load the tokenizer and the base model, quantized when detected or requested."""
        from .model import is_kbit_loaded, load_base_model

        self._lg.info(f"loading base model: {self.base_model}")
        self._load_tokenizer()
        self.model = load_base_model(self._lg, self.base_model, self._quantize_override, dtype)
        self._is_quantized = is_kbit_loaded(self.model)

    def _log_trainable_params(self, decimals: int = 2) -> None:
        """Log how many of the model's parameters the adapter trains."""
        trainable, total = self.model.get_nb_trainable_parameters()
        pct = 100 * trainable / total if total > 0 else 0.0
        self._lg.info(f"Trainable params: {trainable:,} / {total:,} ({pct:.{decimals}f}%)")

    def _training_args(self) -> dict[str, Any]:
        """Arguments the SFT and DPO trainer configs share."""
        tc = self.training_config
        return {
            "output_dir": str(self.output_dir),
            "num_train_epochs": tc.num_epochs,
            "per_device_train_batch_size": tc.batch_size,
            "gradient_accumulation_steps": tc.gradient_accumulation_steps,
            "learning_rate": tc.learning_rate,
            "warmup_steps": compute_warmup_steps(len(self.train_dataset), tc),
            "max_grad_norm": tc.max_grad_norm,
            "logging_steps": tc.logging_steps,
            "save_steps": tc.save_steps,
            "save_total_limit": 2,
            "fp16": tc.fp16,
            "bf16": tc.bf16,
            "gradient_checkpointing": tc.gradient_checkpointing,
            "seed": tc.seed,
            "eval_strategy": "steps" if self.eval_dataset else "no",
            "eval_steps": tc.save_steps if self.eval_dataset else None,
            "load_best_model_at_end": self.eval_dataset is not None,
            "report_to": "none",
            "optim": "paged_adamw_8bit",
            "max_length": tc.max_seq_length,
            "neftune_noise_alpha": tc.neftune_noise_alpha,
        }

    def _method_metrics(self, history: list[dict]) -> dict[str, Any]:
        """Metrics specific to the training method, read from the capped log history."""
        return {}

    def _collect_metrics(self) -> dict:
        """Extract training metrics from trainer log history."""
        if self.trainer is None or not self.trainer.state.log_history:
            return {}

        log_history = self.trainer.state.log_history
        metrics: dict = {"history": log_history[-100:]}
        metrics.update(self._method_metrics(metrics["history"]))

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

    def _add_stability_warnings(self, metrics: dict, log_history: list[dict]) -> None:
        """Check for training instability and add warnings to metrics."""
        stability_report = check_training_stability(log_history)
        log_stability_warnings(self._lg, stability_report)
        if not stability_report.stable:
            metrics["unstable"] = True
            metrics["stability_warnings"] = stability_report.warnings

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

    def _training_summary(self) -> dict[str, Any]:
        """Training hyperparameters a run records in its config."""
        return {k: self.training_config[k] for k in TRAINING_CONFIG_KEYS}

    def _build_result(self, adapter_path: Path, metrics: dict, started_at: datetime) -> RunResult:
        """Build the training result."""
        # Adapter md5/mtime populated by caller (runner) after training
        return RunResult(
            status="completed",
            base_model=self.base_model,
            method=self.METHOD,
            metrics=metrics,
            config=self._run_config(),
            started_at=started_at,
            completed_at=datetime.now(UTC),
            samples_trained=int(len(self.train_dataset) * self.training_config.num_epochs),
            adapter=Adapter(md5="", mtime="", path=str(adapter_path)),
            parent=self.parent,
        )

    def _train(self, resume_from: str | Path | None) -> RunResult:
        """Run the full training pipeline."""
        started_at = datetime.now(UTC)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._lg.info(f"starting {self.LABEL}: {self.data_path} -> {self.output_dir}")

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


class SFTBase(BaseTrainer):
    """Trainer on TRL's SFTTrainer over instruction/output records."""

    def _load_data(self) -> None:
        """Load training and eval datasets."""
        self.train_dataset, self.eval_dataset = load_sft_dataset(
            self.data_path, self.training_config.eval_split
        )
        self._lg.info(f"loaded {len(self.train_dataset)} training samples")
        if self.eval_dataset:
            self._lg.info(f"loaded {len(self.eval_dataset)} eval samples")

    def _create_training_args(self) -> Any:
        """Create training arguments for SFT."""
        from trl import SFTConfig

        return SFTConfig(
            **self._training_args(),
            completion_only_loss=self.training_config.completion_only_loss,
        )

    def _create_trainer(self) -> None:
        """Create the SFTTrainer."""
        from trl import SFTTrainer

        require_chat_template(self.tokenizer, self.base_model)
        self.trainer = SFTTrainer(
            model=self.model,
            args=self._create_training_args(),
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
        )

    def train(self, resume_from: str | Path | None = None) -> RunResult:
        """Run the full training pipeline."""
        return self._train(resume_from)
