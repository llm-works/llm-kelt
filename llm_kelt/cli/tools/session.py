"""Session management CLI tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from appinfra.app.tools import Tool, ToolConfig
from appinfra.log import Logger

from ...conversation.storage import FileSessionStorage

_DEFAULT_SESSIONS_DIR = "~/.llm-kelt/sessions"


def _get_storage(lg: Logger, args: Any) -> FileSessionStorage:
    """Create storage backend from CLI args."""
    base_path = getattr(args, "sessions_dir", None) or _DEFAULT_SESSIONS_DIR
    return FileSessionStorage(lg, Path(base_path).expanduser())


def _positive_int(value: str) -> int:
    """Argparse type validator for positive integers."""
    import argparse

    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"limit must be >= 1, got {n}")
    return n


class ListSessionsTool(Tool):
    """List stored sessions."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent, ToolConfig(name="list", aliases=["ls"], help_text="List sessions"))

    def add_args(self, parser) -> None:
        parser.add_argument(
            "--limit", "-n", type=_positive_int, default=20, help="Max sessions to show"
        )

    def run(self, **kwargs: Any) -> int:
        storage = _get_storage(self.lg, self.args)

        try:
            summaries = storage.list(limit=self.args.limit)
        except OSError as e:
            self.lg.warning("failed to list sessions", extra={"exception": e})
            return 1

        if not summaries:
            print("No sessions found.")
            return 0

        for s in summaries:
            preview = f"  {s.preview}" if s.preview else ""
            print(
                f"{s.session_id}  msgs={s.message_count}  tokens={s.token_count}"
                f"  updated={s.updated_at}{preview}"
            )

        return 0


class ShowSessionTool(Tool):
    """Display session contents."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent, ToolConfig(name="show", help_text="Show session details"))

    def add_args(self, parser) -> None:
        parser.add_argument("session_id", help="Session ID to display")
        parser.add_argument("--json", dest="as_json", action="store_true", help="Output as JSON")

    def run(self, **kwargs: Any) -> int:
        storage = _get_storage(self.lg, self.args)

        try:
            session = storage.load(self.args.session_id)
        except Exception as e:
            self.lg.warning("session load failed", extra={"exception": e})
            print(f"Failed to load session: {self.args.session_id}")
            return 1

        if self.args.as_json:
            print(json.dumps(dict(session), indent=2, default=str))
            return 0

        print(f"Session: {session.session_id}")
        print(f"Created: {session.created_at}")
        print(f"Updated: {session.updated_at}")
        print(f"Messages: {len(session.messages)}")
        print(f"Tokens: {session.token_count}")

        if session.metadata:
            print(f"Metadata: {json.dumps(session.metadata, default=str)}")

        print("\n--- Messages ---")
        for msg in session.messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            print(f"\n[{role}]")
            print(content)

        return 0


class DeleteSessionTool(Tool):
    """Delete a stored session."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="delete", aliases=["rm"], help_text="Delete a session")
        )

    def add_args(self, parser) -> None:
        parser.add_argument("session_id", help="Session ID to delete")

    def run(self, **kwargs: Any) -> int:
        storage = _get_storage(self.lg, self.args)

        try:
            deleted = storage.delete(self.args.session_id)
        except OSError as e:
            self.lg.warning("failed to delete session", extra={"exception": e})
            return 1

        if deleted:
            print(f"Deleted session: {self.args.session_id}")
        else:
            print(f"Session not found: {self.args.session_id}")
            return 1

        return 0


class SessionTool(Tool):
    """Session management commands."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(
            parent, ToolConfig(name="session", aliases=["sess"], help_text="Session commands")
        )
        self.add_tool(ListSessionsTool(self))
        self.add_tool(ShowSessionTool(self))
        self.add_tool(DeleteSessionTool(self))

    def add_args(self, parser) -> None:
        parser.add_argument(
            "--sessions-dir", help=f"Sessions directory (default: {_DEFAULT_SESSIONS_DIR})"
        )

    def run(self, **kwargs: Any) -> int:
        result: int = self.group.run(**kwargs)
        return result
