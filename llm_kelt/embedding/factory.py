"""Factory for creating embedding clients and stores."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import BigInteger, DateTime, Float, Index, LargeBinary, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .store.base import EmbeddingBase
from .types import Config, QuantizationFormat

if TYPE_CHECKING:
    from .client import StoreClient


_model_cache: dict[str, type] = {}


class ModelCache:
    """Cache for dynamically-created embedding model classes.

    Embedding models are created on-demand for specific table names (derived from
    format, dimensions, and optional prefix). Uses a module-level cache shared
    across all Factory instances to prevent duplicate index/constraint accumulation.
    """

    def get_or_create(self, config: Config) -> type:
        """Get cached model or create and cache a new one."""
        table_name = config.table_name
        if table_name not in _model_cache:
            _model_cache[table_name] = self._create_model(config)
        return _model_cache[table_name]

    def _create_model(self, config: Config) -> type:
        """Create a new model class for the given config."""
        match config.format:
            case QuantizationFormat.F32:
                return self._create_f32_model(config)
            case QuantizationFormat.F16:
                return self._create_f16_model(config)
            case QuantizationFormat.I8:
                return self._create_i8_model(config)
            case QuantizationFormat.I4:
                return self._create_i4_model(config)

    def _create_f32_model(self, config: Config) -> type:
        from pgvector.sqlalchemy import Vector

        table_name = config.table_name
        dimensions = config.dimensions

        class EmbeddingF32(EmbeddingBase):
            __tablename__ = table_name
            __table_args__ = (
                UniqueConstraint(
                    "entity_type", "entity_id", "model_name", name=f"uq_{table_name}_entity_model"
                ),
                Index(f"idx_{table_name}_entity", "entity_type", "entity_id"),
                Index(f"idx_{table_name}_model", "model_name"),
            )

            id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
            entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
            entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
            model_name: Mapped[str] = mapped_column(String(100), nullable=False)
            embedding: Mapped[list[float]] = mapped_column(Vector(dimensions), nullable=False)
            created_at: Mapped[datetime] = mapped_column(
                DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
            )

        EmbeddingF32.__name__ = f"EmbeddingF32_{table_name}"
        EmbeddingF32.__qualname__ = f"EmbeddingF32_{table_name}"
        return EmbeddingF32

    def _create_f16_model(self, config: Config) -> type:
        from pgvector.sqlalchemy import HALFVEC

        table_name = config.table_name
        dimensions = config.dimensions

        class EmbeddingF16(EmbeddingBase):
            __tablename__ = table_name
            __table_args__ = (
                UniqueConstraint(
                    "entity_type", "entity_id", "model_name", name=f"uq_{table_name}_entity_model"
                ),
                Index(f"idx_{table_name}_entity", "entity_type", "entity_id"),
                Index(f"idx_{table_name}_model", "model_name"),
            )

            id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
            entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
            entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
            model_name: Mapped[str] = mapped_column(String(100), nullable=False)
            embedding: Mapped[list[float]] = mapped_column(HALFVEC(dimensions), nullable=False)
            created_at: Mapped[datetime] = mapped_column(
                DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
            )

        EmbeddingF16.__name__ = f"EmbeddingF16_{table_name}"
        EmbeddingF16.__qualname__ = f"EmbeddingF16_{table_name}"
        return EmbeddingF16

    def _create_i8_model(self, config: Config) -> type:
        table_name = config.table_name

        class EmbeddingI8(EmbeddingBase):
            __tablename__ = table_name
            __table_args__ = (
                UniqueConstraint(
                    "entity_type", "entity_id", "model_name", name=f"uq_{table_name}_entity_model"
                ),
                Index(f"idx_{table_name}_entity", "entity_type", "entity_id"),
                Index(f"idx_{table_name}_model", "model_name"),
            )

            id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
            entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
            entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
            model_name: Mapped[str] = mapped_column(String(100), nullable=False)
            embedding_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
            scale: Mapped[float] = mapped_column(Float, nullable=False)
            offset: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
            created_at: Mapped[datetime] = mapped_column(
                DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
            )

        EmbeddingI8.__name__ = f"EmbeddingI8_{table_name}"
        EmbeddingI8.__qualname__ = f"EmbeddingI8_{table_name}"
        return EmbeddingI8

    def _create_i4_model(self, config: Config) -> type:
        table_name = config.table_name

        class EmbeddingI4(EmbeddingBase):
            __tablename__ = table_name
            __table_args__ = (
                UniqueConstraint(
                    "entity_type", "entity_id", "model_name", name=f"uq_{table_name}_entity_model"
                ),
                Index(f"idx_{table_name}_entity", "entity_type", "entity_id"),
                Index(f"idx_{table_name}_model", "model_name"),
            )

            id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
            entity_type: Mapped[str] = mapped_column(String(50), nullable=False)
            entity_id: Mapped[str] = mapped_column(String(64), nullable=False)
            model_name: Mapped[str] = mapped_column(String(100), nullable=False)
            embedding_bytes: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
            scale: Mapped[float] = mapped_column(Float, nullable=False)
            offset: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
            created_at: Mapped[datetime] = mapped_column(
                DateTime(timezone=True), default=lambda: datetime.now(UTC), nullable=False
            )

        EmbeddingI4.__name__ = f"EmbeddingI4_{table_name}"
        EmbeddingI4.__qualname__ = f"EmbeddingI4_{table_name}"
        return EmbeddingI4


class Factory:
    """Factory for creating embedding clients.

    Manages model caching to prevent duplicate index/constraint accumulation
    when creating multiple clients with the same format and dimensions.

    Example:
        factory = Factory()
        client = factory.create(session_factory, config)
    """

    def __init__(self) -> None:
        self._model_cache = ModelCache()

    def create(self, session_factory: Callable[[], Any], config: Config) -> StoreClient:
        """Create an embedding client.

        Args:
            session_factory: Callable that returns a context manager for DB sessions.
            config: Embedding configuration (format, dimensions).

        Returns:
            Configured embedding client.
        """
        from .client import StoreClient

        store = self._create_store(session_factory, config)
        return StoreClient(config, store)

    def _create_store(self, session_factory: Callable[[], Any], config: Config) -> Any:
        """Create a store for the given config."""
        model = self._model_cache.get_or_create(config)

        match config.format:
            case QuantizationFormat.F32:
                from .store.f32 import Float32Store

                return Float32Store(session_factory, config.dimensions, model)
            case QuantizationFormat.F16:
                from .store.f16 import Float16Store

                return Float16Store(session_factory, config.dimensions, model)
            case QuantizationFormat.I8:
                from .store.i8 import Int8Store

                return Int8Store(session_factory, config.dimensions, model)
            case QuantizationFormat.I4:
                from .store.i4 import Int4Store

                return Int4Store(session_factory, config.dimensions, model)


# Module-level factory instance for convenience
_default_factory: Factory | None = None


def get_factory() -> Factory:
    """Get the default factory instance (creates one if needed)."""
    global _default_factory
    if _default_factory is None:
        _default_factory = Factory()
    return _default_factory
