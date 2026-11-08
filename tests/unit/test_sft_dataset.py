# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for SFT dataset loading (TRL conversational prompt-completion format)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_kelt.training.runtime.dataset import to_prompt_completion
from llm_kelt.training.schema import TRAINING_DEFAULTS


def _turns(user: str, assistant: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": assistant}],
    }


class TestToPromptCompletion:
    def test_instruction_and_output(self):
        assert to_prompt_completion({"instruction": "foo", "output": "bar"}) == _turns("foo", "bar")

    def test_input_appended_after_blank_line(self):
        record = {"instruction": "foo", "input": "baz", "output": "bar"}
        assert to_prompt_completion(record) == _turns("foo\n\nbaz", "bar")

    def test_prompt_and_response_aliases(self):
        assert to_prompt_completion({"prompt": "foo", "response": "bar"}) == _turns("foo", "bar")

    def test_extra_fields_dropped(self):
        record = {"instruction": "foo", "output": "bar", "weight": 2.0, "meta": {"id": 1}}
        assert to_prompt_completion(record) == _turns("foo", "bar")

    def test_unrendered_template_raises(self):
        with pytest.raises(ValueError, match="was not rendered"):
            to_prompt_completion({"template": "abc", "slots": {}, "output": "bar"})

    def test_missing_instruction_raises(self):
        with pytest.raises(ValueError, match="missing 'instruction' or 'prompt'"):
            to_prompt_completion({"output": "bar"})

    def test_missing_output_raises(self):
        with pytest.raises(ValueError, match="missing 'output' or 'response'"):
            to_prompt_completion({"instruction": "foo"})


def test_completion_only_loss_defaults_off():
    assert TRAINING_DEFAULTS.completion_only_loss is False


@pytest.mark.integration
@pytest.mark.parametrize("override", [None, True])
def test_completion_only_loss_reaches_sft_config(lg, tmp_path: Path, override):
    """The trainer passes the configured (default: off) completion_only_loss to TRL."""
    pytest.importorskip("trl")
    from llm_kelt.training.runtime.lora import Trainer

    overrides = {"lora_profile": "small"}
    if override is not None:
        overrides["completion_only_loss"] = override
    trainer = Trainer(
        lg=lg,
        data_path=tmp_path / "data.jsonl",
        output_dir=tmp_path / "out",
        base_model="org/model-7b",
        training_config=overrides,
    )
    trainer.train_dataset = [{}] * 4
    assert trainer._create_training_args().completion_only_loss is (override is True)
