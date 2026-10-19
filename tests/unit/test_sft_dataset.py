# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for SFT dataset loading (TRL conversational prompt-completion format)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_kelt.training.schema import TRAINING_DEFAULTS
from llm_kelt.training.sft.dataset import load_sft_dataset, to_prompt_completion


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


class TestLoadSftDataset:
    @pytest.fixture(autouse=True)
    def _require_datasets(self):
        pytest.importorskip("datasets")

    def _write(self, tmp_path: Path, records: list[dict]) -> Path:
        path = tmp_path / "data.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records) + "\n", encoding="utf-8")
        return path

    def test_loads_conversational_records(self, tmp_path: Path):
        path = self._write(tmp_path, [{"instruction": "foo", "output": "bar"}])
        train, eval_ds = load_sft_dataset(path, eval_split=0.0)
        assert eval_ds is None
        assert train[0] == _turns("foo", "bar")

    def test_eval_split(self, tmp_path: Path):
        path = self._write(tmp_path, [{"instruction": f"q{i}", "output": "a"} for i in range(10)])
        train, eval_ds = load_sft_dataset(path, eval_split=0.2)
        assert len(train) == 8
        assert len(eval_ds) == 2

    def test_eval_split_leaving_no_training_data_is_skipped(self, tmp_path: Path):
        path = self._write(tmp_path, [{"instruction": "foo", "output": "bar"}])
        train, eval_ds = load_sft_dataset(path, eval_split=0.5)
        assert len(train) == 1
        assert eval_ds is None

    def test_empty_file_raises(self, tmp_path: Path):
        path = self._write(tmp_path, [])
        with pytest.raises(ValueError, match="No records found"):
            load_sft_dataset(path, eval_split=0.0)


def test_completion_only_loss_defaults_off():
    assert TRAINING_DEFAULTS.completion_only_loss is False
