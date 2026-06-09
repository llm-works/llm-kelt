"""File-based session storage.

Stores each session as a JSON file on disk. Suitable for CLI tools
and lightweight agents that don't need a database.

Directory structure::

    base_path/
    └── {session_id}.json
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from appinfra.log import Logger

from ...core.errors import NotFoundError
from .base import SessionStorage, SessionSummary, StoredSession

if TYPE_CHECKING:
    from ..session import Conversation

_PREVIEW_MAX_LEN = 80


class FileSessionStorage(SessionStorage):
    """File-based session storage.

    Each session is stored as a single JSON file containing the full
    conversation state: messages, config, extra, and timestamps.

    Args:
        lg: Logger instance.
        base_path: Directory for session files. Created on first write.
    """

    def __init__(self, lg: Logger, base_path: str | Path) -> None:
        self._lg = lg
        self._base_path = Path(base_path).expanduser()

    def save(
        self,
        session_id: str,
        conversation: Conversation,
        extra: dict | None = None,
    ) -> None:
        """Save conversation to a JSON file."""
        self._base_path.mkdir(parents=True, exist_ok=True)
        path = self._session_path(session_id)

        now = datetime.now(UTC).isoformat()

        # Preserve created_at from existing session
        created_at = now
        if path.exists():
            try:
                existing = json.loads(path.read_text())
                created_at = existing.get("created_at", now)
            except (json.JSONDecodeError, OSError):
                pass

        data = StoredSession(
            session_id=session_id,
            messages=conversation.messages_as_dicts(),
            created_at=created_at,
            updated_at=now,
            extra=extra or {},
            token_count=conversation.token_count,
            config=dict(conversation.config),
        )

        path.write_text(json.dumps(dict(data), indent=2, default=str))
        self._lg.debug("session saved", extra={"session_id": session_id, "path": str(path)})

    def load(self, session_id: str) -> StoredSession:
        """Load session from JSON file."""
        path = self._session_path(session_id)
        if not path.exists():
            raise NotFoundError(f"Session not found: {session_id}")

        data = json.loads(path.read_text())
        return StoredSession(**data)

    def list(self, limit: int = 100) -> list[SessionSummary]:
        """List sessions sorted by most recently updated."""
        if not self._base_path.exists():
            return []

        summaries: list[SessionSummary] = []
        for path in self._base_path.glob("*.json"):
            summary = self._read_summary(path)
            if summary is not None:
                summaries.append(summary)

        summaries.sort(key=lambda s: s.updated_at, reverse=True)
        return summaries[:limit]

    def delete(self, session_id: str) -> bool:
        """Delete session file."""
        path = self._session_path(session_id)
        if not path.exists():
            return False

        path.unlink()
        self._lg.debug("session deleted", extra={"session_id": session_id})
        return True

    def _session_path(self, session_id: str) -> Path:
        """Get filesystem path for a session.

        Raises:
            ValueError: If session_id contains path traversal characters.
        """
        if not session_id:
            raise ValueError("Session ID cannot be empty")
        if "/" in session_id or "\\" in session_id or ".." in session_id:
            raise ValueError(f"Invalid session ID (path traversal): {session_id}")
        return self._base_path / f"{session_id}.json"

    def _read_summary(self, path: Path) -> SessionSummary | None:
        """Read just the summary fields from a session file."""
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            self._lg.warning("failed to read session file", extra={"path": str(path)})
            return None

        messages = data.get("messages", [])
        preview = _extract_preview(messages)

        return SessionSummary(
            session_id=data.get("session_id", path.stem),
            message_count=len(messages),
            token_count=data.get("token_count", 0),
            updated_at=data.get("updated_at", ""),
            preview=preview,
        )


def _extract_preview(messages: list[dict]) -> str:
    """Extract first user message as preview, truncated."""
    for msg in messages:
        if msg.get("role") == "user":
            content: str = msg.get("content", "")
            if len(content) > _PREVIEW_MAX_LEN:
                return content[:_PREVIEW_MAX_LEN] + "..."
            return content
    return ""
