"""Abstract base class for session storage.

Defines the interface for persisting and retrieving conversation sessions,
along with the data models for stored sessions and summaries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from appinfra import FieldDict, field

if TYPE_CHECKING:
    from ..session import Conversation


class StoredSession(FieldDict):
    """A persisted conversation session.

    Attributes:
        session_id: Unique session identifier.
        messages: List of message dicts (Message-as-dict).
        created_at: ISO 8601 timestamp of session creation.
        updated_at: ISO 8601 timestamp of last update.
        metadata: Arbitrary metadata (token counts, model, etc.).
        token_count: Estimated token count at time of save.
        config: Config as dict (for restoring session state).
    """

    session_id: str
    messages: list[dict] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)
    token_count: int = 0
    config: dict = field(default_factory=dict)


class SessionSummary(FieldDict):
    """Lightweight summary of a stored session for listing.

    Attributes:
        session_id: Unique session identifier.
        message_count: Number of messages in the session.
        token_count: Estimated token count.
        updated_at: ISO 8601 timestamp of last update.
        preview: First user message, truncated.
    """

    session_id: str
    message_count: int = 0
    token_count: int = 0
    updated_at: str = ""
    preview: str = ""


class SessionStorage(ABC):
    """Abstract interface for session persistence.

    Implementations store and retrieve conversation sessions. The save method
    accepts a Conversation object directly, preserving token counts and config
    alongside the messages.
    """

    @abstractmethod
    def save(
        self,
        session_id: str,
        conversation: Conversation,
        metadata: dict | None = None,
    ) -> None:
        """Save a conversation session.

        Creates or overwrites the session with the given ID.

        Args:
            session_id: Unique session identifier.
            conversation: Conversation to persist.
            metadata: Optional metadata (token counts, model, etc.).
        """

    @abstractmethod
    def load(self, session_id: str) -> StoredSession:
        """Load a stored session.

        Args:
            session_id: Session to load.

        Returns:
            The stored session data.

        Raises:
            NotFoundError: If session does not exist.
        """

    @abstractmethod
    def list(self, limit: int = 100) -> list[SessionSummary]:
        """List stored sessions, most recently updated first.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of session summaries.
        """

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a stored session.

        Args:
            session_id: Session to delete.

        Returns:
            True if session was deleted, False if it didn't exist.
        """
