# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for StoreCache."""

import threading
from unittest.mock import Mock

import pytest

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

    @pytest.mark.integration
    def test_threads_share_one_store_per_dimension(self):
        cache, factory, _ = _cache()
        workers = 16
        barrier = threading.Barrier(workers)
        seen: list[object] = []
        seen_lock = threading.Lock()

        def work() -> None:
            barrier.wait(timeout=10)
            store = cache.get(384)
            with seen_lock:
                seen.append(store)

        threads = [threading.Thread(target=work) for _ in range(workers)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(seen) == workers
        assert all(store is seen[0] for store in seen)
        factory.create.assert_called_once()
        assert cache.stores() == [seen[0]]
