# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Precision settings read from a checkpoint's config.json."""

import json

import pytest

from llm_kelt.training.runtime.model import _apply_precision_settings


class TestPrecisionSettings:
    """AMP precision picked from the checkpoint's config.json."""

    @staticmethod
    def _apply(lg, tmp_path, model_config: dict, overrides: dict | None = None) -> dict:
        (tmp_path / "config.json").write_text(json.dumps(model_config))
        overrides = overrides or {}
        config = {"fp16": True, "bf16": False, **overrides}
        _apply_precision_settings(lg, str(tmp_path), config, overrides)
        return config

    @pytest.mark.parametrize("key", ["dtype", "torch_dtype"])
    def test_bfloat16_weights_use_bf16(self, lg, tmp_path, key: str):
        config = self._apply(lg, tmp_path, {key: "bfloat16"})
        assert (config["fp16"], config["bf16"]) == (False, True)

    def test_float32_weights_keep_defaults(self, lg, tmp_path):
        config = self._apply(lg, tmp_path, {"dtype": "float32"})
        assert (config["fp16"], config["bf16"]) == (True, False)

    def test_quantized_model_disables_amp(self, lg, tmp_path):
        model_config = {"dtype": "bfloat16", "quantization_config": {"quant_method": "gptq"}}
        config = self._apply(lg, tmp_path, model_config)
        assert (config["fp16"], config["bf16"]) == (False, False)

    def test_explicit_fp16_turns_bf16_off(self, lg, tmp_path):
        config = self._apply(lg, tmp_path, {"dtype": "bfloat16"}, {"fp16": True})
        assert (config["fp16"], config["bf16"]) == (True, False)

    def test_explicit_bf16_turns_default_fp16_off(self, lg, tmp_path):
        config = self._apply(lg, tmp_path, {"dtype": "float32"}, {"bf16": True})
        assert (config["fp16"], config["bf16"]) == (False, True)

    def test_explicit_fp16_off_leaves_bf16_to_checkpoint(self, lg, tmp_path):
        config = self._apply(lg, tmp_path, {"dtype": "bfloat16"}, {"fp16": False})
        assert (config["fp16"], config["bf16"]) == (False, True)

    def test_no_local_checkpoint_keeps_defaults(self, lg, tmp_path):
        config = {"fp16": True, "bf16": False}
        _apply_precision_settings(lg, str(tmp_path / "missing"), config, {})
        assert (config["fp16"], config["bf16"]) == (True, False)

    def test_malformed_json_keeps_defaults(self, lg, tmp_path):
        # get_model_metadata raises JSONDecodeError (a ValueError) before dtype is read
        (tmp_path / "config.json").write_text("not valid json {")
        config = {"fp16": True, "bf16": False}
        _apply_precision_settings(lg, str(tmp_path), config, {})
        assert (config["fp16"], config["bf16"]) == (True, False)
