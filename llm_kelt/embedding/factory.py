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
    from .client import Client


_model_cache: dict[tuple[QuantizationFormat, int], type] = {}


class ModelCache:
    """Cache for dynamically-created embedding model classes.

    Embedding models are created on-demand for specific (format, dimensions)
    combinations. Uses a module-level cache shared across all Factory instances
    to prevent duplicate index/constraint accumulation.
    """

    def get_or_create(self, fmt: QuantizationFormat, dimensions: int) -> type:
        """Get cached model or create and cache a new one."""
        key = (fmt, dimensions)
        if key not in _model_cache:
            _model_cache[key] = self._create_model(fmt, dimensions)
        return _model_cache[key]

    def _create_model(self, fmt: QuantizationFormat, dimensions: int) -> type:
        """Create a new model class for the given format and dimensions."""
        match fmt:
            case QuantizationFormat.F32:
                return self._create_f32_model(dimensions)
            case QuantizationFormat.F16:
                return self._create_f16_model(dimensions)
            case QuantizationFormat.I8:
                return self._create_i8_model(dimensions)
            case QuantizationFormat.I4:
                return self._create_i4_model(dimensions)

    def _create_f32_model(self, dimensions: int) -> type:
        from pgvector.sqlalchemy import Vector

        table_name = f"embeddings_{dimensions}_f32"

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

        EmbeddingF32.__name__ = f"EmbeddingF32_{dimensions}"
        EmbeddingF32.__qualname__ = f"EmbeddingF32_{dimensions}"
        return EmbeddingF32

    def _create_f16_model(self, dimensions: int) -> type:
        from pgvector.sqlalchemy import HALFVEC

        table_name = f"embeddings_{dimensions}_f16"

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

        EmbeddingF16.__name__ = f"EmbeddingF16_{dimensions}"
        EmbeddingF16.__qualname__ = f"EmbeddingF16_{dimensions}"
        return EmbeddingF16

    def _create_i8_model(self, dimensions: int) -> type:
        table_name = f"embeddings_{dimensions}_i8"

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

        EmbeddingI8.__name__ = f"EmbeddingI8_{dimensions}"
        EmbeddingI8.__qualname__ = f"EmbeddingI8_{dimensions}"
        return EmbeddingI8

    def _create_i4_model(self, dimensions: int) -> type:
        table_name = f"embeddings_{dimensions}_i4"

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

        EmbeddingI4.__name__ = f"EmbeddingI4_{dimensions}"
        EmbeddingI4.__qualname__ = f"EmbeddingI4_{dimensions}"
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

    def create(self, session_factory: Callable[[], Any], config: Config) -> Client:
        """Create an embedding client.

        Args:
            session_factory: Callable that returns a context manager for DB sessions.
            config: Embedding configuration (format, dimensions).

        Returns:
            Configured embedding client.
        """
        from .client import Client as ClientImpl

        store = self._create_store(session_factory, config)
        return ClientImpl(config, store)

    def _create_store(self, session_factory: Callable[[], Any], config: Config) -> Any:
        """Create a store for the given config."""
        model = self._model_cache.get_or_create(config.format, config.dimensions)

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
