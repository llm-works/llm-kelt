# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Factory bundling dependencies for Conversation construction.

Satisfies saia's ``ConversationFactory`` Protocol so frameworks (e.g. saia's
tool loop, gent's pause/resume) can create and restore conversations without
importing ``Conversation`` directly.
"""

from __future__ import annotations

from typing import Any

from appinfra.log import Logger

from .compaction.base import AsyncCompactor, Compactor
from .session import Config, Conversation


class ConversationFactory:
    """Bundle ``lg``, ``config`` and ``compactor`` for deferred Conversation construction.

    Satisfies saia's ``ConversationFactory`` Protocol structurally: ``create()``
    returns a fresh Conversation, ``create_from_state()`` restores one from a
    dict produced by ``Conversation.to_dict()``.
    """

    def __init__(
        self,
        lg: Logger,
        config: Config | None = None,
        compactor: Compactor | AsyncCompactor | None = None,
    ) -> None:
        self._lg = lg
        self._config = config
        self._compactor = compactor

    def create(self) -> Conversation:
        return Conversation(self._lg, config=self._config, compactor=self._compactor)

    def create_from_state(self, state: dict[str, Any]) -> Conversation:
        return Conversation.from_dict(
            state, self._lg, config=self._config, compactor=self._compactor
        )
