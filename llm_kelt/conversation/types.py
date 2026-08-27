# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Canonical message types for conversations.

Re-exports from llm-saia, which owns the agent↔LLM communication types.
Kelt adds conversation management (token tracking, compaction, persistence)
on top of saia's primitives.
"""

from llm_saia import Message, Role, ToolCall

__all__ = ["Message", "Role", "ToolCall"]
