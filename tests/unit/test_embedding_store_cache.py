# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for StoreCache."""

from unittest.mock import Mock

from llm_kelt.embedding import QuantizationFormat, StoreCache


def _cache() -> tuple[StoreCache, Mock, Mock]:
    factory = Mock()
    factory.create.side_effect = lambda session_factory, config: Mock(config=config)
    session_factory = Mock()
    cache = StoreCache(factory, session_factory, QuantizationFormat.F16, schema="tenant")
    return cache, factory, session_factory


class TestStoreCache:
    """One store per dimension, shared by every caller of the cache."""

    def test_builds_one_store_per_dimension(self):
        cache, factory, session_factory = _cache()

        assert cache.get(384) is cache.get(384)
        assert cache.get(768) is not cache.get(384)

        assert factory.create.call_count == 2
        first = factory.create.call_args_list[0].args
        assert first[0] is session_factory
        assert (first[1].dimensions, first[1].format, first[1].schema) == (
            384,
            QuantizationFormat.F16,
            "tenant",
        )

    def test_stores_lists_built_stores(self):
        cache, _, _ = _cache()

        assert cache.stores() == []
        store = cache.get(384)
        assert cache.stores() == [store]
        assert cache.schema == "tenant"
