# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""SFT datasets loaded from files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_kelt.training.runtime.dataset import load_sft_dataset


def _turns(user: str, assistant: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": user}],
        "completion": [{"role": "assistant", "content": assistant}],
    }


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
