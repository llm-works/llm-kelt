# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""CLI entry point for llm-kelt."""

from pathlib import Path

from appinfra.app import AppBuilder

from .. import __version__
from .tools import AtomicTool, ProxyTool, SessionTool, TrainTool

_BASE_CONFIG = Path(__file__).parent.parent / "etc" / "llm-kelt.yaml"


def main() -> int:
    """Main entry point for the CLI."""
    app = (
        AppBuilder("llm-kelt")
        .with_description("LLM kelt framework - collect and manage LLM context")
        .with_config_spec("llm-works", "llm-kelt", _BASE_CONFIG)
        .with_standard_args(etc_dir=True)
        .logging.with_level("info")
        .with_location(1)
        .done()
        .advanced.with_argument("-v", "--version", action="version", version=__version__)
        .done()
        .tools.with_tool(AtomicTool())
        .with_tool(ProxyTool())
        .with_tool(TrainTool())
        .with_tool(SessionTool())
        .done()
        .build()
    )
    result: int = app.main()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
