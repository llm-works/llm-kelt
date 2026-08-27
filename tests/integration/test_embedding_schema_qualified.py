# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Regression test for schema-qualified embedding tables.

Reproduces the failure mode observed on llm-xray's
``TestFetchEmbeddings::test_batch_fetch_skips_missing``: a leftover
``public.embeddings_{dim}_{fmt}`` in an operator's DB was silently latched
onto by kelt's bare-name ORM model, so reads and writes bypassed the
configured schema. Downstream: phantom rows visible only through the
poisoned schema, ``UPDATE`` instead of ``INSERT`` on ``set_embedding``,
and ``get_embeddings`` returning entity_ids that were never explicitly
embedded in the current session.

The fix threads the effective schema from ``Database`` down through
``Protocol`` / ``EmbeddingAdapter`` into ``Config``; ``ModelCache`` binds
``__table_args__ = {"schema": ...}`` so every read, write, and CREATE
names the intended schema explicitly.

This test plants a poisoned ``public.embeddings_8_f16`` with rows for
entity_ids '1', '2', '3', then runs the batch-fetch flow inside an
isolated schema and asserts:
  1. The test's writes land in the configured schema, not ``public``.
  2. The batch fetch returns exactly the two entity_ids explicitly
     embedded (never fact_id 3's poisoned row).
  3. The poisoned rows in ``public`` are untouched by the run.
"""

from __future__ import annotations

from typing import Any

import pytest
from sqlalchemy import text


@pytest.mark.integration
class TestEmbeddingSchemaQualified:
    """Bare-name search_path fallback is closed."""

    def _plant_public_poison(self, database: Any, dims: int) -> None:
        """Recreate the shape reported from prod: ``public.embeddings_{dims}_f16``
        with rows for the same entity_ids the test will use."""
        with database.session() as sess:
            conn = sess.connection()
            conn.execute(text(f"DROP TABLE IF EXISTS public.embeddings_{dims}_f16 CASCADE"))
            conn.execute(
                text(
                    f"CREATE TABLE public.embeddings_{dims}_f16 ("
                    " id BIGSERIAL PRIMARY KEY,"
                    " entity_type VARCHAR(50) NOT NULL,"
                    " entity_id VARCHAR(64) NOT NULL,"
                    " model_name VARCHAR(100) NOT NULL,"
                    f" embedding HALFVEC({dims}) NOT NULL,"
                    " created_at TIMESTAMPTZ NOT NULL DEFAULT now(),"
                    " UNIQUE(entity_type, entity_id, model_name))"
                )
            )
            vec_lit = "[" + ",".join(["0.5"] * dims) + "]"
            for eid in ("1", "2", "3"):
                conn.execute(
                    text(
                        f"INSERT INTO public.embeddings_{dims}_f16"
                        " (entity_type, entity_id, model_name, embedding)"
                        f" VALUES ('atomic.fact', :eid, 'test-embed',"
                        f" CAST(:vec AS halfvec))"
                    ),
                    {"eid": eid, "vec": vec_lit},
                )
            sess.commit()

    def _drop_public_poison(self, database: Any, dims: int) -> None:
        with database.session() as sess:
            sess.connection().execute(
                text(f"DROP TABLE IF EXISTS public.embeddings_{dims}_f16 CASCADE")
            )
            sess.commit()

    def test_set_embedding_ignores_public_leftover(self, kelt_client: Any, database: Any) -> None:
        """Poison public; verify the client reads and writes only its schema."""
        dims = 8
        model = "test-embed"

        self._plant_public_poison(database, dims)
        try:
            kelt_client.atomic.embeddings._default_dimensions = dims

            f1 = kelt_client.atomic.assertions.add(content="fact one", source="test")
            f2 = kelt_client.atomic.assertions.add(content="fact two", source="test")
            f_missing = kelt_client.atomic.assertions.add(content="fact three", source="test")

            kelt_client.atomic.embeddings.set_embedding(f1, [0.1] * dims, model)
            kelt_client.atomic.embeddings.set_embedding(f2, [0.2] * dims, model)

            got = kelt_client.atomic.embeddings.get_embeddings([f1, f2, f_missing], model)
            assert set(got) == {f1, f2}, (
                f"batch fetch leaked the poisoned public row: {sorted(got)}"
            )
            assert got[f1] == pytest.approx([0.1] * dims, abs=1e-3)
            assert got[f2] == pytest.approx([0.2] * dims, abs=1e-3)

            # Positive proof the write went to the configured schema, not public.
            # Assert against ``database.schema`` (worker-scoped) rather than
            # ``public`` — the latter is shared across xdist workers and any
            # concurrent test with the ``clean_tables`` fixture would race us
            # by DELETE'ing from unqualified ``embeddings_%`` tables.
            #
            # Fact IDs aren't hard-coded: ``atomic_facts.id`` is a session-scoped
            # BIGSERIAL that other integration tests within the same worker also
            # advance, so we pin against the actual ``f1``/``f2`` we got back
            # from ``assertions.add``.
            configured = database.schema
            assert configured, "test relies on database being schema-configured"
            with database.session() as sess:
                rows = (
                    sess.connection()
                    .execute(
                        text(
                            f"SELECT entity_id, LEFT(embedding::text, 8) "
                            f'FROM "{configured}".embeddings_{dims}_f16 '
                            "WHERE entity_id = ANY(:ids) "
                            "ORDER BY entity_id::int"
                        ),
                        {"ids": [str(f1), str(f2), str(f_missing)]},
                    )
                    .all()
                )
                assert rows == [(str(f1), "[0.09997"), (str(f2), "[0.19995")], (
                    f"expected only f1/f2 in {configured}.embeddings_{dims}_f16; "
                    f"got {rows}. A row for f_missing or a leading 0.5 means the "
                    "write resolved through search_path to public."
                )
        finally:
            self._drop_public_poison(database, dims)

    def test_model_binds_schema_from_config(self) -> None:
        """``Config.schema`` propagates onto the cached ORM class's
        ``__table__.schema`` so ORM-generated SQL is schema-qualified even
        outside a live session."""
        from llm_kelt.embedding import Config, QuantizationFormat
        from llm_kelt.embedding.factory import ModelCache

        cache = ModelCache()
        bare = cache.get_or_create(
            Config(context_key="_t", format=QuantizationFormat.F16, dimensions=8)
        )
        qualified = cache.get_or_create(
            Config(
                context_key="_t",
                format=QuantizationFormat.F16,
                dimensions=8,
                schema="tenant_x",
            )
        )
        assert bare is not qualified, "schema must be part of the model cache key"
        assert bare.__table__.schema is None
        assert qualified.__table__.schema == "tenant_x"
