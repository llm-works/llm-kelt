# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""SFT dataset loading for TRL's SFTTrainer.

Records are mapped to TRL's conversational prompt-completion format: TRL applies the
tokenizer's chat template itself and knows where the completion starts, so the
``completion_only_loss`` training option can exclude the prompt tokens from the loss.
"""

from pathlib import Path
from typing import Any

from ...core.export.jsonl import load_jsonl


def to_prompt_completion(record: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Map an SFT record to TRL's conversational prompt-completion format.

    The user turn is ``instruction`` (or ``prompt``), followed by ``input`` after a blank
    line when present. The assistant turn is ``output`` (or ``response``).

    Raises:
        ValueError: If the record still references a template, or is missing required fields.
    """
    if "template" in record:
        raise ValueError(
            f"SFT record references template {record['template']!r} but was not rendered; "
            "templated records must be rendered from the manifest's templates before training"
        )
    instruction = record.get("instruction") or record.get("prompt", "")
    if not instruction:
        raise ValueError("SFT record missing 'instruction' or 'prompt' field")
    output = record.get("output") or record.get("response", "")
    if not output:
        raise ValueError("SFT record missing 'output' or 'response' field")
    input_text = record.get("input", "")
    user_content = f"{instruction}\n\n{input_text}" if input_text else instruction
    return {
        "prompt": [{"role": "user", "content": user_content}],
        "completion": [{"role": "assistant", "content": output}],
    }


def load_sft_dataset(data_path: Path, eval_split: float) -> tuple[Any, Any | None]:
    """Load SFT records from a JSONL file and optionally split off an eval set.

    Returns:
        (train_dataset, eval_dataset) as ``datasets.Dataset``; eval_dataset is None when
        eval_split is 0 or would leave no training records.
    """
    from datasets import Dataset

    records = load_jsonl(data_path)
    if not records:
        raise ValueError(f"No records found in {data_path}")

    dataset = Dataset.from_list([to_prompt_completion(r) for r in records])

    if eval_split > 0:
        if int(len(dataset) * (1 - eval_split)) < 1:
            return dataset, None
        split = dataset.train_test_split(test_size=eval_split, seed=42)
        return split["train"], split["test"]

    return dataset, None


def require_chat_template(tokenizer: Any, base_model: str) -> None:
    """Raise if the tokenizer has no chat template (needed to render conversational data)."""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer for {base_model} lacks chat_template. "
            "SFT training requires a model with chat template support for proper EOS learning."
        )
