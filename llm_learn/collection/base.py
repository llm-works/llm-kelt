"""Base client with shared functionality for all collection clients."""

from collections.abc import Callable
from typing import Any, Generic, TypeVar, cast

from appinfra.db.utils import detach, detach_all
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..core.models import Base

T = TypeVar("T", bound=Base)


class BaseClient(Generic[T]):
    """
    Base class for collection clients.

    Provides common CRUD operations and session management.
    Subclasses must set the `model` class attribute.
    """

    model: type[T]  # Subclasses must set this

    def __init__(self, session_factory: Callable[[], Any]) -> None:
        """
        Initialize client with session factory.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
                            Typically `Database.session`.
        """
        self._session_factory = session_factory

    def _get_session(self) -> Session:
        """Get a raw database session (caller manages lifecycle)."""
        return cast(Session, self._session_factory().__enter__())

    def get(self, id: int) -> T | None:
        """
        Get record by ID.

        Args:
            id: Record ID

        Returns:
            Record if found, None otherwise
        """
        with self._session_factory() as session:
            obj = session.get(self.model, id)
            if obj is None:
                return None
            return cast(T, detach(obj, session))

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> list[T]:
        """
        List records with pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Column name to order by
            descending: If True, order descending (newest first)

        Returns:
            List of records
        """
        with self._session_factory() as session:
            order_col = getattr(self.model, order_by)
            if descending:
                order_col = order_col.desc()

            stmt = select(self.model).order_by(order_col).limit(limit).offset(offset)
            objects = list(session.scalars(stmt).all())
            return cast(list[T], detach_all(objects, session))

    def count(self) -> int:
        """
        Count total records.

        Returns:
            Total number of records
        """
        with self._session_factory() as session:
            stmt = select(func.count()).select_from(self.model)
            return session.scalar(stmt) or 0

    def delete(self, id: int) -> bool:
        """
        Delete record by ID.

        Args:
            id: Record ID

        Returns:
            True if deleted, False if not found
        """
        with self._session_factory() as session:
            record = session.get(self.model, id)
            if record:
                session.delete(record)
                return True
            return False

    def exists(self, id: int) -> bool:
        """
        Check if record exists.

        Args:
            id: Record ID

        Returns:
            True if exists, False otherwise
        """
        with self._session_factory() as session:
            stmt = (
                select(func.count()).select_from(self.model).where(self.model.id == id)  # type: ignore[attr-defined]
            )
            return (session.scalar(stmt) or 0) > 0


class ProfileScopedClient(BaseClient[T]):
    """
    Base class for collection clients scoped to a profile.

    All operations are automatically filtered by profile_id.
    Subclasses must set the `model` class attribute and the model
    must have a `profile_id` column.
    """

    def __init__(self, session_factory: Callable[[], Any], profile_id: int) -> None:
        """
        Initialize client scoped to a specific profile.

        Args:
            session_factory: Callable that returns a context manager for database sessions.
            profile_id: Profile ID to scope all operations to.
        """
        super().__init__(session_factory)
        self.profile_id = profile_id

    def get(self, id: int) -> T | None:
        """
        Get record by ID (must belong to this profile).

        Args:
            id: Record ID

        Returns:
            Record if found and belongs to profile, None otherwise
        """
        with self._session_factory() as session:
            obj = session.get(self.model, id)
            if obj is None or getattr(obj, "profile_id", None) != self.profile_id:
                return None
            return cast(T, detach(obj, session))

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        order_by: str = "created_at",
        descending: bool = True,
    ) -> list[T]:
        """
        List records for this profile with pagination.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            order_by: Column name to order by
            descending: If True, order descending (newest first)

        Returns:
            List of records belonging to this profile
        """
        with self._session_factory() as session:
            order_col = getattr(self.model, order_by)
            if descending:
                order_col = order_col.desc()

            stmt = (
                select(self.model)
                .where(self.model.profile_id == self.profile_id)  # type: ignore[attr-defined]
                .order_by(order_col)
                .limit(limit)
                .offset(offset)
            )
            objects = list(session.scalars(stmt).all())
            return cast(list[T], detach_all(objects, session))

    def count(self) -> int:
        """
        Count total records for this profile.

        Returns:
            Total number of records belonging to this profile
        """
        with self._session_factory() as session:
            stmt = (
                select(func.count())
                .select_from(self.model)
                .where(self.model.profile_id == self.profile_id)  # type: ignore[attr-defined]
            )
            return session.scalar(stmt) or 0

    def delete(self, id: int) -> bool:
        """
        Delete record by ID (must belong to this profile).

        Args:
            id: Record ID

        Returns:
            True if deleted, False if not found or belongs to different profile
        """
        with self._session_factory() as session:
            record = session.get(self.model, id)
            if record and getattr(record, "profile_id", None) == self.profile_id:
                session.delete(record)
                return True
            return False

    def exists(self, id: int) -> bool:
        """
        Check if record exists in this profile.

        Args:
            id: Record ID

        Returns:
            True if exists and belongs to this profile, False otherwise
        """
        with self._session_factory() as session:
            stmt = (
                select(func.count())
                .select_from(self.model)
                .where(
                    self.model.id == id,  # type: ignore[attr-defined]
                    self.model.profile_id == self.profile_id,  # type: ignore[attr-defined]
                )
            )
            return (session.scalar(stmt) or 0) > 0
