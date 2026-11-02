# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""DPO (Direct Preference Optimization) training.

Uses TRL's DPOTrainer with PEFT for preference-based fine-tuning.
Trains the model to prefer "chosen" responses over "rejected" ones.
"""

import json
from pathlib import Path
from typing import Any

from appinfra import DotDict
from appinfra.log import Logger

from ..lora.config import Config as LoraConfig
from ..schema import Adapter, RunResult
from .base import BaseTrainer, lora_config_summary
from .peft_config import lora_peft_config


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


class Trainer(BaseTrainer):
    """Trainer for LoRA adapters using Direct Preference Optimization."""

    METHOD = "dpo"
    LABEL = "DPO training"

    def __init__(
        self,
        lg: Logger,
        data_path: str | Path,
        output_dir: str | Path,
        base_model: str,
        lora_config: LoraConfig | None = None,
        training_config: DotDict | None = None,
        beta: float = 0.1,
        quantize: bool | None = None,
        parent: Adapter | None = None,
    ):
        super().__init__(lg, data_path, output_dir, base_model, training_config, quantize)
        self.lora_config = lora_config or LoraConfig()
        self.beta = beta
        self.parent = parent

    def _init_state(self) -> None:
        """Initialize mutable state to None (set during training)."""
        super()._init_state()
        self.ref_model: Any = None
        self._use_stacked_adapters = False  # True when parent adapter is stacked with DPO

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
            self.lora_config = self.lora_config.resolve(self.model.config.model_type)
            peft_config = lora_peft_config(self.lora_config)
            self.model = get_peft_model(self.model, peft_config)

        self._log_trainable_params()

    def _load_model(self):
        """Load base model with auto-detected quantization and apply LoRA."""
        import torch
        from peft import prepare_model_for_kbit_training

        self._load_base_model(dtype=torch.bfloat16)
        if self.training_config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        if self._is_quantized:
            self.model = prepare_model_for_kbit_training(self.model)

        self._apply_lora_adapter()
        self._log_reference_strategy()

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

    def _create_training_args(self):
        """Create DPO training arguments."""
        from trl import DPOConfig

        config_args = self._training_args()
        config_args["beta"] = self.beta
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

    def _method_metrics(self, history: list[dict]) -> dict[str, Any]:
        """DPO reward metrics from the logged training steps."""
        return self._extract_dpo_metrics([log for log in history if "rewards/accuracies" in log])

    def _run_config(self) -> dict[str, Any]:
        """Hyperparameters the run records in its result."""
        return {
            "lora": lora_config_summary(self.lora_config),
            "training": self._training_summary(),
            "dpo": {"beta": self.beta},
            "quantized": self._is_quantized,
        }

    def train(self) -> RunResult:
        """Run the full training pipeline."""
        return self._train(resume_from=None)


def train_dpo(
    lg: Logger,
    data_path: str | Path,
    output_dir: str | Path,
    base_model: str,
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
        base_model: HuggingFace model ID or local path.
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
