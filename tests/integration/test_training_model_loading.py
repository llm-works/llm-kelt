# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Model class, config and quantization read from checkpoint files."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")

from llm_kelt.training.runtime import model as model_mod  # noqa: E402


def _write_config(path: Path, **fields) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(fields))
    return path


def _model_class(lg, path: Path):
    return model_mod.resolve_model_class(model_mod.load_model_config(lg, str(path)))


class TestResolveModelClass:
    def test_saved_conditional_generation_class(self, tmp_path: Path, lg):
        path = _write_config(
            tmp_path / "m",
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
        )
        assert _model_class(lg, path) is transformers.Qwen3_5ForConditionalGeneration

    def test_saved_causal_lm_class(self, tmp_path: Path, lg):
        path = _write_config(tmp_path / "m", model_type="qwen2", architectures=["Qwen2ForCausalLM"])
        assert _model_class(lg, path) is transformers.Qwen2ForCausalLM

    @pytest.mark.parametrize(
        "architectures",
        [None, ["Qwen2ForSequenceClassification"], ["SomeRemoteCodeForCausalLM"]],
    )
    def test_falls_back_to_auto_causal_lm(self, tmp_path: Path, lg, architectures):
        fields = {"model_type": "qwen2"}
        if architectures is not None:
            fields["architectures"] = architectures
        path = _write_config(tmp_path / "m", **fields)
        assert _model_class(lg, path) is transformers.AutoModelForCausalLM


_KV_CACHE_SCHEME = {"num_bits": 8, "type": "float", "strategy": "tensor", "symmetric": True}


class TestLoadModelConfig:
    def test_drops_compressed_tensors_kv_cache_scheme(self, tmp_path: Path, lg):
        path = _write_config(
            tmp_path / "m",
            model_type="qwen2",
            quantization_config={
                "quant_method": "compressed-tensors",
                "format": "pack-quantized",
                "kv_cache_scheme": _KV_CACHE_SCHEME,
            },
        )
        quant = model_mod.load_model_config(lg, str(path)).quantization_config
        assert quant["kv_cache_scheme"] is None
        assert quant["format"] == "pack-quantized"
        on_disk = json.loads((path / "config.json").read_text())
        assert on_disk["quantization_config"]["kv_cache_scheme"] == _KV_CACHE_SCHEME

    def test_compressed_tensors_loads_dequantized(self, tmp_path: Path, lg):
        path = _write_config(
            tmp_path / "m",
            model_type="qwen2",
            quantization_config={"quant_method": "compressed-tensors", "format": "pack-quantized"},
        )
        quant = model_mod.load_model_config(lg, str(path)).quantization_config
        assert quant["dequantize"] is True
        # transformers builds the load-time quantization config from this dict
        assert transformers.CompressedTensorsConfig.from_dict(quant).dequantize is True
        on_disk = json.loads((path / "config.json").read_text())
        assert "dequantize" not in on_disk["quantization_config"]

    def test_keeps_other_formats(self, tmp_path: Path, lg):
        quant = {"quant_method": "gptq", "bits": 4, "kv_cache_scheme": _KV_CACHE_SCHEME}
        path = _write_config(tmp_path / "m", model_type="qwen2", quantization_config=quant)
        assert model_mod.load_model_config(lg, str(path)).quantization_config == quant

    def test_load_base_model_passes_the_loaded_config(self, tmp_path: Path, lg, monkeypatch):
        path = _write_config(
            tmp_path / "m",
            model_type="qwen2",
            quantization_config={
                "quant_method": "compressed-tensors",
                "kv_cache_scheme": _KV_CACHE_SCHEME,
            },
        )
        captured: dict = {}

        class _Model:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                captured.update(kwargs)
                return cls()

        monkeypatch.setattr(model_mod, "get_quantization_config", lambda *_: (None, False))
        monkeypatch.setattr(model_mod, "resolve_model_class", lambda _config: _Model)
        model_mod.load_base_model(lg, str(path), None)
        assert captured["config"].quantization_config["kv_cache_scheme"] is None


class TestQuantizationConfig:
    @pytest.mark.parametrize("override", [None, False, True])
    def test_compressed_tensors_passes_no_config_whatever_the_override(
        self, tmp_path: Path, lg, override
    ):
        """The checkpoint's own config, set to dequantize by load_model_config, is the only one."""
        path = _write_config(
            tmp_path / "m",
            quantization_config={"quant_method": "compressed-tensors", "format": "pack-quantized"},
        )
        assert model_mod.get_quantization_config(lg, str(path), override) == (None, False)

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
