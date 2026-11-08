# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for base-model loading decisions that read no checkpoint."""

from __future__ import annotations

import pytest

from llm_kelt.training.runtime import model as model_mod


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
