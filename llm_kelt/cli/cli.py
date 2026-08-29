# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""CLI entry point for llm-kelt."""

from pathlib import Path

from appinfra.app import AppBuilder

from .tools import AtomicTool, ProxyTool, SessionTool, TrainTool

# Bundled etc/ ships inside the wheel. Used as the default --etc-dir so
# `pip install llm-kelt && llm-kelt <tool>` works without local setup.
_BUNDLED_ETC_DIR = str(Path(__file__).parent.parent / "etc")


def main() -> int:
    """Main entry point for the CLI."""
    app = (
        AppBuilder("llm-kelt")
        .with_description("LLM kelt framework - collect and manage LLM context")
        .with_config_file("llm-kelt.yaml")
        .with_standard_arg("etc_dir", default=_BUNDLED_ETC_DIR)
        .logging.with_level("info")
        .with_location(1)
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
