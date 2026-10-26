# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Training runtime: the trainers and every module that imports the ML stack.

Code outside this package never imports its modules by name; it goes through load(). Type
checkers do not follow that import, so a project type-checking against kelt's API does not
analyse torch, transformers, peft, trl or datasets.
"""

import importlib
from types import ModuleType
from typing import Literal

RuntimeModule = Literal["lora", "dpo", "prompt", "merge"]


def load(name: RuntimeModule) -> ModuleType:
    """Import a runtime module by name (e.g. load("lora").train_lora)."""
    return importlib.import_module(f"{__name__}.{name}")
