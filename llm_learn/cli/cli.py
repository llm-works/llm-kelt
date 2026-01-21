"""CLI entry point for llm-learn."""

from appinfra.app import AppBuilder

from .tools import ProxyTool


def main() -> int:
    """Main entry point for the CLI."""
    app = (
        AppBuilder("llm-learn")
        .with_description("LLM learning framework - collect and manage LLM context")
        .with_config_file("llm-learn.yaml")
        .logging.with_level("info")
        .with_location(1)
        .done()
        .tools.with_tool(ProxyTool())
        .done()
        .build()
    )
    result: int = app.main()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
