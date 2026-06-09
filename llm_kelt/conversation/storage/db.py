"""Database-backed session storage.

Stores sessions in PostgreSQL using SQLAlchemy. Messages are stored as JSONB
for efficient querying. Requires the ``conv_sessions`` table to exist (see migrations).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger
from sqlalchemy import DateTime, Index, Integer, String, UniqueConstraint, desc, func, select
from sqlalchemy.orm import Mapped, mapped_column

from ...core.base import Base
from ...core.errors import NotFoundError
from .base import SessionStorage, SessionSummary, StoredSession

if TYPE_CHECKING:
    from ..session import Conversation

# Conditional JSONB import — this module requires psycopg2 at import time (the
# Session model uses JSONB columns). The guard only prevents cascading failures
# when storage/__init__.py imports just the file backend.
try:
    from sqlalchemy.dialects.postgresql import JSONB
except ImportError:  # pragma: no cover
    JSONB = None  # type: ignore[assignment,misc]

_PREVIEW_MAX_LEN = 80


class Session(Base):
    """SQLAlchemy model for conversation sessions."""

    __tablename__ = "conv_sessions"
    __table_args__ = (
        UniqueConstraint("session_id", name="uq_conv_sessions_session_id"),
        Index("idx_conv_sessions_session_id", "session_id"),
        Index("idx_conv_sessions_updated_at", "updated_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    messages: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


class DbSessionStorage(SessionStorage):
    """PostgreSQL-backed session storage.

    Uses the ``conv_sessions`` table with JSONB columns for messages, config,
    and extra. Follows kelt's session_factory pattern.

    Args:
        lg: Logger instance.
        session_factory: Callable that returns a SQLAlchemy session context manager.
    """

    def __init__(self, lg: Logger, session_factory: Callable[[], Any]) -> None:
        self._lg = lg
        self._session_factory = session_factory

    def save(
        self,
        session_id: str,
        conversation: Conversation,
        extra: dict | None = None,
    ) -> None:
        """Save conversation to database."""
        now = datetime.now(UTC)

        with self._session_factory() as db:
            existing = db.execute(
                select(Session).where(Session.session_id == session_id)
            ).scalar_one_or_none()

            if existing is not None:
                existing.messages = conversation.messages_as_dicts()
                existing.token_count = conversation.token_count
                existing.config = dict(conversation.config)
                existing.extra = extra or {}
                existing.updated_at = now
            else:
                record = Session(
                    session_id=session_id,
                    messages=conversation.messages_as_dicts(),
                    token_count=conversation.token_count,
                    config=dict(conversation.config),
                    extra=extra or {},
                    created_at=now,
                    updated_at=now,
                )
                db.add(record)

        self._lg.debug("session saved", extra={"session_id": session_id})

    def load(self, session_id: str) -> StoredSession:
        """Load session from database."""
        with self._session_factory() as db:
            record = db.execute(
                select(Session).where(Session.session_id == session_id)
            ).scalar_one_or_none()

            if record is None:
                raise NotFoundError(f"Session not found: {session_id}")

            return StoredSession(
                session_id=record.session_id,
                messages=record.messages,
                created_at=record.created_at.isoformat(),
                updated_at=record.updated_at.isoformat(),
                extra=record.extra,
                token_count=record.token_count,
                config=record.config,
            )

    def list(self, limit: int = 100) -> list[SessionSummary]:
        """List sessions sorted by most recently updated."""
        with self._session_factory() as db:
            records = (
                db.execute(select(Session).order_by(desc(Session.updated_at)).limit(limit))
                .scalars()
                .all()
            )
            return [self._to_summary(r) for r in records]

    def delete(self, session_id: str) -> bool:
        """Delete session from database."""
        with self._session_factory() as db:
            record = db.execute(
                select(Session).where(Session.session_id == session_id)
            ).scalar_one_or_none()

            if record is None:
                return False

            db.delete(record)

        self._lg.debug("session deleted", extra={"session_id": session_id})
        return True

    def _to_summary(self, record: Session) -> SessionSummary:
        """Convert DB record to SessionSummary."""
        messages = record.messages or []
        preview = ""
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if len(content) > _PREVIEW_MAX_LEN:
                    preview = content[:_PREVIEW_MAX_LEN] + "..."
                else:
                    preview = content
                break

        return SessionSummary(
            session_id=record.session_id,
            message_count=len(messages),
            token_count=record.token_count,
            updated_at=record.updated_at.isoformat(),
            preview=preview,
        )
