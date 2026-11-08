# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""The bootstrap creates the tables of the models registered with Base at that point."""

import os
import subprocess
import sys
from pathlib import Path

import llm_kelt

_TABLES = [
    "kelt_atomic_directive_details",
    "kelt_atomic_fact_relationships",
    "kelt_atomic_facts",
    "kelt_atomic_feedback_details",
    "kelt_atomic_interaction_details",
    "kelt_atomic_prediction_details",
    "kelt_atomic_preference_details",
    "kelt_atomic_solution_details",
    "kelt_content",
    "kelt_conv_sessions",
    "kelt_kg_entities",
    "kelt_kg_entity_aliases",
    "kelt_kg_entity_refs",
    "kelt_kg_entity_relationships",
    "kelt_kg_fact_entities",
]


def test_import_registers_every_model():
    """Importing llm_kelt registers every model, in a process that imported nothing else.

    Runs in a subprocess: the pytest process has imported every module already.
    """
    pkg_root = str(Path(llm_kelt.__file__).resolve().parent.parent)
    pythonpath = os.pathsep.join(p for p in (pkg_root, os.environ.get("PYTHONPATH")) if p)
    code = (
        "import llm_kelt; from llm_kelt.core.base import Base; print(*sorted(Base.metadata.tables))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": pythonpath},
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == _TABLES
