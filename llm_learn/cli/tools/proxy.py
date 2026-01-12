"""Proxy tool - manages the learning proxy server."""

from typing import Any

from appinfra.app.tools import Tool, ToolConfig


class ServeTool(Tool):
    """Start the proxy server."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="serve",
            aliases=["s"],
            help_text="Start the proxy server",
        )
        super().__init__(parent, config)

    def add_args(self, parser) -> None:
        parser.add_argument("--host", help="Host to bind to")
        parser.add_argument("--port", "-p", type=int, help="Port to bind to")
        parser.add_argument("--profile", type=int, help="Profile ID to use")

    def _get_config(self) -> dict:
        """Get raw config dict from app."""
        return dict(self.app.config) if self.app.config else {}

    def run(self, **kwargs: Any) -> int:
        from ...serving.app import create_server_from_config

        config = self._get_config()

        # Apply CLI overrides
        if getattr(self.args, "host", None):
            config["host"] = self.args.host
        if getattr(self.args, "port", None):
            config["port"] = self.args.port
        if getattr(self.args, "profile", None):
            config["profile_id"] = self.args.profile

        self.lg.info(
            "starting proxy server...",
            extra={
                "host": config.get("host", "0.0.0.0"),
                "port": config.get("port", 8001),
                "profile_id": config.get("profile_id", 1),
            },
        )

        server = create_server_from_config(config)
        server.start()
        return 0


class ProxyTool(Tool):
    """Proxy server management commands."""

    def __init__(self, parent: Any = None) -> None:
        config = ToolConfig(
            name="proxy",
            aliases=["p"],
            help_text="Proxy server commands",
        )
        super().__init__(parent, config)
        self.add_tool(ServeTool(self))

    def run(self, **kwargs: Any) -> int:
        """Delegate to subtool."""
        result: int = self.group.run(**kwargs)
        return result
