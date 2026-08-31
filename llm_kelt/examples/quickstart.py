#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Minimal quick-start smoke that mirrors the README example.

Runs against an installed wheel via `python -m llm_kelt.examples.quickstart`. Exercised
by the smoke-wheel CI job to catch README-vs-installed API drift, broken
top-level exports, and missing wheel resources.

Requires a Postgres+pgvector database. Reads the URL from ``DATABASE_URL`` and
falls back to the local dev server described in ``etc/pg.yaml``. The LLM and
embedding backends are not exercised — the goal is to prove the persistence
surface is wired end-to-end (add fact → build system prompt), not to validate
model output.
"""

from __future__ import annotations

import os
import sys
import uuid

from appinfra.dot_dict import DotDict
from appinfra.log import LogConfig, LoggerFactory

from llm_kelt import ClientContext, ClientFactory
from llm_kelt.inference import ContextBuilder

DEFAULT_URL = "postgresql://postgres:postgres@127.0.0.1:7632/learn_test"


def main() -> int:
    database_url = os.environ.get("DATABASE_URL", DEFAULT_URL)

    lg = LoggerFactory.create_root(LogConfig.from_params(level="warning"))
    config = DotDict({"dbs": {"main": {"url": database_url, "create_db": True}}})

    # Unique key so repeated smoke runs don't accumulate rows for the same agent.
    context_key = f"quickstart-{uuid.uuid4().hex[:8]}"

    kelt = ClientFactory(lg).create_from_config(
        context=ClientContext(context_key=context_key),
        config=config,
    )

    kelt.atomic.assertions.add("Timezone: UTC", category="settings")
    kelt.atomic.assertions.add("Prefers concise, code-first answers", category="style")

    system_prompt = ContextBuilder(kelt.atomic.assertions).build_system_prompt(
        base_prompt="You are a helpful assistant.",
    )

    assert "Timezone: UTC" in system_prompt, "expected assertion missing from prompt"
    assert "code-first" in system_prompt, "expected assertion missing from prompt"

    print(f"context_key: {context_key}")
    print("system_prompt:")
    print(system_prompt)
    return 0


if __name__ == "__main__":
    sys.exit(main())
