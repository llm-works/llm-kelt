"""Unit tests for Float16Store row decoding.

Guards the halfvec store against AttributeError when a row's embedding
attribute arrives as a plain list[float] instead of a pgvector.HalfVector.
This happens under SQLAlchemy identity-map staleness: after an upsert
assigns a list to the ORM attribute, a same-session read returns the same
instance with the raw list still on it (bind_processor only runs on flush,
not on attribute assignment).
"""

from types import SimpleNamespace

from pgvector import HalfVector

from llm_kelt.embedding.store.f16 import Float16Store


def _store() -> Float16Store:
    """Construct a Float16Store without exercising the DB path.

    _embedding_from_row only touches row.embedding, so session_factory /
    dimensions / model can be minimal stubs.
    """
    return Float16Store(session_factory=lambda: None, dimensions=3, model=object())


class TestEmbeddingFromRow:
    """Regression tests for _embedding_from_row on the halfvec store."""

    def test_halfvec_row(self) -> None:
        """HalfVector-typed row.embedding decodes via .to_list()."""
        row = SimpleNamespace(embedding=HalfVector([0.1, 0.2, 0.3]))

        result = _store()._embedding_from_row(row)

        assert isinstance(result, list)
        assert len(result) == 3
        for got, want in zip(result, [0.1, 0.2, 0.3]):
            assert abs(got - want) < 1e-3

    def test_list_row(self) -> None:
        """list[float]-typed row.embedding decodes without raising.

        Repro for the reported bug: before the fix, this raised
        AttributeError: 'list' object has no attribute 'to_list'.
        """
        row = SimpleNamespace(embedding=[0.1, 0.2, 0.3])

        result = _store()._embedding_from_row(row)

        assert result == [0.1, 0.2, 0.3]
        assert all(isinstance(v, float) for v in result)

    def test_list_row_with_numpy_scalars(self) -> None:
        """Mixed-scalar list normalizes to plain Python floats."""
        import numpy as np

        row = SimpleNamespace(embedding=[np.float16(0.1), np.float32(0.2), np.float64(0.3)])

        result = _store()._embedding_from_row(row)

        assert all(type(v) is float for v in result)
