#!/usr/bin/env python3

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Minimal quick-start smoke that mirrors the README example.

Runs against an installed wheel via ``python -m llm_kelt.examples.quickstart``.
Exercised by the smoke-wheel CI job to catch README-vs-installed API drift,
broken top-level exports, and missing wheel resources.

Prerequisite: a running Postgres+pgvector server. The module reads
``DATABASE_URL`` and falls back to
``postgresql://postgres:postgres@127.0.0.1:25432/learn_test`` (matches the
shipped ``etc/pg.yaml``). The fastest way to stand one up when none is
running:

.. code-block:: bash

    docker run -d --rm --name kelt-quickstart-db \\
      -p 25432:5432 \\
      -e POSTGRES_PASSWORD=postgres \\
      -e POSTGRES_DB=learn_test \\
      pgvector/pgvector:pg16

Repo cloners can equivalently ``make pg.server.up`` (uses the shipped
``etc/pg.yaml``). See ``docs/quickstart.md`` section 1 for the full menu
(repo Makefile, standalone docker, existing Postgres).

The LLM and embedding backends are not exercised — the goal is to prove the
persistence surface is wired end-to-end (add fact → build system prompt),
not to validate model output.
"""

from __future__ import annotations

import os
import sys
import uuid

from appinfra.dot_dict import DotDict
from appinfra.log import LogConfig, LoggerFactory
from sqlalchemy.exc import OperationalError

from llm_kelt import ClientContext, ClientFactory
from llm_kelt.inference import ContextBuilder

DEFAULT_URL = "postgresql://postgres:postgres@127.0.0.1:25432/learn_test"


def _print_db_prereq(url: str, err: Exception) -> None:
    first_line = str(err).splitlines()[0] if str(err) else type(err).__name__
    print(
        f"""
llm-kelt quickstart requires a running Postgres+pgvector server.

Tried:  {url}
Error:  {first_line}

Quickest fix — start one with Docker (matches the default URL above):

  docker run -d --rm --name kelt-quickstart-db \\
    -p 25432:5432 \\
    -e POSTGRES_PASSWORD=postgres \\
    -e POSTGRES_DB=learn_test \\
    pgvector/pgvector:pg16

Then re-run: python -m llm_kelt.examples.quickstart

Alternatives (repo Makefile target, existing Postgres, custom DATABASE_URL):
see docs/quickstart.md section 1 — https://github.com/serendip-ml/llm-kelt/blob/main/docs/quickstart.md
""".rstrip(),
        file=sys.stderr,
    )


def _run(database_url: str) -> int:
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


def main() -> int:
    database_url = os.environ.get("DATABASE_URL", DEFAULT_URL)
    try:
        return _run(database_url)
    except OperationalError as e:
        _print_db_prereq(database_url, e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
