# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Float32 embedding store using pgvector vector type."""

from __future__ import annotations

from .vector import VectorStore


class Float32Store(VectorStore):
    """Float32 embedding store using pgvector vector type.

    Uses native pgvector for storage and similarity search.
    Full precision, no quantization loss.
    """

    _OPCLASS = "vector_cosine_ops"
