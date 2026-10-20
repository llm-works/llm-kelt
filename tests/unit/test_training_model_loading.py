# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for base-model loading decisions: model class and quantization config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

from llm_kelt.training import model as model_mod  # noqa: E402


def _write_config(path: Path, **fields) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(fields))
    return path


class TestResolveModelClass:
    def test_saved_conditional_generation_class(self, tmp_path: Path):
        path = _write_config(
            tmp_path / "m",
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
        )
        assert (
            model_mod.resolve_model_class(str(path)) is transformers.Qwen3_5ForConditionalGeneration
        )

    def test_saved_causal_lm_class(self, tmp_path: Path):
        path = _write_config(tmp_path / "m", model_type="qwen2", architectures=["Qwen2ForCausalLM"])
        assert model_mod.resolve_model_class(str(path)) is transformers.Qwen2ForCausalLM

    @pytest.mark.parametrize(
        "architectures",
        [None, ["Qwen2ForSequenceClassification"], ["SomeRemoteCodeForCausalLM"]],
    )
    def test_falls_back_to_auto_causal_lm(self, tmp_path: Path, architectures):
        fields = {"model_type": "qwen2"}
        if architectures is not None:
            fields["architectures"] = architectures
        path = _write_config(tmp_path / "m", **fields)
        assert model_mod.resolve_model_class(str(path)) is transformers.AutoModelForCausalLM


class TestQuantizationConfig:
    @pytest.mark.parametrize("override", [None, False, True])
    def test_compressed_tensors_loads_dequantized_whatever_the_override(
        self, tmp_path: Path, lg, override
    ):
        path = _write_config(
            tmp_path / "m",
            quantization_config={"quant_method": "compressed-tensors", "format": "pack-quantized"},
        )
        config, applied = model_mod.get_quantization_config(lg, str(path), override)
        assert isinstance(config, transformers.CompressedTensorsConfig)
        assert config.dequantize is True
        assert applied is False

    def test_other_prequantized_formats_load_with_own_config(self, tmp_path: Path, lg):
        path = _write_config(tmp_path / "m", quantization_config={"quant_method": "gptq"})
        assert model_mod.get_quantization_config(lg, str(path), None) == (None, False)

    def test_full_precision_quantized_by_default(self, tmp_path: Path, lg):
        pytest.importorskip("torch")
        path = _write_config(tmp_path / "m", model_type="qwen2")
        config, applied = model_mod.get_quantization_config(lg, str(path), None)
        assert isinstance(config, transformers.BitsAndBytesConfig)
        assert applied is True

    def test_full_precision_override_off(self, tmp_path: Path, lg):
        path = _write_config(tmp_path / "m", model_type="qwen2")
        assert model_mod.get_quantization_config(lg, str(path), False) == (None, False)


class _Loaded:
    def __init__(self, **flags):
        self.__dict__.update(flags)


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({}, False),
        ({"is_loaded_in_4bit": True}, True),
        ({"is_loaded_in_8bit": True}, True),
        ({"is_loaded_in_4bit": False, "is_loaded_in_8bit": False}, False),
    ],
)
def test_is_kbit_loaded(flags: dict, expected: bool):
    assert model_mod.is_kbit_loaded(_Loaded(**flags)) is expected
