# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Memory module - knowledge storage implementations.

Submodules:
- atomic: Fact-based storage with type-specific detail tables
- kg: Knowledge graph with entities and scoped subgraphs
- isolation: ClientContext for data partitioning
"""

from . import atomic, kg
from .isolation import ClientContext

__all__ = ["atomic", "kg", "ClientContext"]
