# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Float16 embedding store using pgvector halfvec type."""

from __future__ import annotations

from typing import Any

from .vector import VectorStore


class Float16Store(VectorStore):
    """Float16 embedding store using pgvector halfvec type.

    Uses native pgvector halfvec for 2x storage reduction with ~0% accuracy loss.
    Embeddings are converted to float16 on storage, returned as float32 on retrieval.
    """

    _OPCLASS = "halfvec_cosine_ops"

    def _decode(self, val: Any) -> list[float]:
        """Convert halfvec/list to float32 list.

        Accepts both pgvector.HalfVector (from a fresh DB read via HALFVEC's
        result_processor) and list[float] (from a same-session read after an
        upsert, where the ORM's identity map still carries the raw value
        assigned before flush).
        """
        if isinstance(val, list):
            return [float(v) for v in val]
        return [float(v) for v in val.to_list()]
