# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Store, read, search and delete through every embedding format against Postgres."""

import pytest

from llm_kelt.embedding import Config, Factory, QuantizationFormat

MODEL = "test-model"

# Largest per-value error each format may introduce on a round trip
TOLERANCE = {
    QuantizationFormat.F32: 1e-6,
    QuantizationFormat.F16: 1e-3,
    QuantizationFormat.I8: 1e-2,
    QuantizationFormat.I4: 1e-1,
}

A = [1.0, 0.0, 0.0, 0.0]
B = [0.0, 1.0, 0.0, 0.0]
NEAR_A = [0.9, 0.1, 0.0, 0.0]


@pytest.fixture(params=list(QuantizationFormat), ids=lambda f: f.value)
def fmt(request):
    """Each embedding format in turn."""
    return request.param


@pytest.fixture
def embeddings(database, clean_tables, fmt):
    """Embedding client on a 4-dimensional table of the given format."""
    config = Config(context_key="_test", format=fmt, dimensions=4)
    return Factory().create(database.session, config)


def _assert_close(got, want, fmt):
    assert got is not None
    assert len(got) == len(want)
    assert all(abs(g - w) <= TOLERANCE[fmt] for g, w in zip(got, want))


class TestEmbeddingFormats:
    """Each format behaves the same through the embedding client."""

    def test_round_trip(self, embeddings, fmt):
        """get() and get_many() return what was stored, within the format's precision."""
        embeddings.store("doc", "a", A, MODEL)
        embeddings.store("doc", "near_a", NEAR_A, MODEL)

        _assert_close(embeddings.get("doc", "a", MODEL), A, fmt)
        many = embeddings.get_many("doc", ["a", "near_a", "missing"], MODEL)
        assert set(many) == {"a", "near_a"}
        _assert_close(many["near_a"], NEAR_A, fmt)
        assert embeddings.get("doc", "missing", MODEL) is None

    def test_store_replaces_existing(self, embeddings, fmt):
        """Storing the same entity again replaces its embedding."""
        embeddings.store("doc", "a", A, MODEL)
        embeddings.store("doc", "a", B, MODEL)

        _assert_close(embeddings.get("doc", "a", MODEL), B, fmt)
        assert embeddings.count(entity_type="doc", model=MODEL) == 1

    def test_search_orders_by_similarity(self, embeddings):
        """search() ranks by cosine similarity and applies the threshold."""
        embeddings.store("doc", "a", A, MODEL)
        embeddings.store("doc", "near_a", NEAR_A, MODEL)
        embeddings.store("doc", "b", B, MODEL)

        results = embeddings.search(A, "doc", MODEL, top_k=10, min_similarity=0.5)

        assert [eid for eid, _ in results] == ["a", "near_a"]
        assert results[0][1] > results[1][1]

    def test_delete(self, embeddings):
        """delete() removes the entity's embedding."""
        embeddings.store("doc", "a", A, MODEL)

        assert embeddings.delete("doc", "a") == 1
        assert not embeddings.exists("doc", "a", MODEL)
