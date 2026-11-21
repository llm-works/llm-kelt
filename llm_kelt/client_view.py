# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Context-keyed view over a shared Client."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

from .core.content import ContentStore
from .inference.context import ContextBuilder
from .inference.query import ContextQuery
from .memory.atomic import Protocol
from .memory.isolation import ClientContext
from .memory.kg import KGStore

if TYPE_CHECKING:
    # client.py imports this module to build views.
    from .client import Client

_T = TypeVar("_T")


class ClientView:
    """
    Context-keyed view over a shared Client.

    Created by Client.view(). Creating a view only binds its context: no
    database I/O and no store setup. Each store is built on first access
    and reuses the parent's database, embedder, embedding factory and LLM
    client. A schema other than the parent's is prepared once per parent
    Client, on the first store access of any view that names it.

    A view and its parent can be used from any thread.

    Usage:
        kelt = Client(lg, database=db, context=ClientContext(context_key="host"))
        agent = kelt.view("acme:prod:reviewer")
        agent.atomic.assertions.add("Prefers concise answers", category="style")
    """

    def __init__(self, parent: Client, context: ClientContext) -> None:
        """
        Bind a view to its parent and context.

        Args:
            parent: Client whose resources the view shares
            context: Isolation context of the view
        """
        self._parent = parent
        self._context = context
        # Reentrant: context_builder and query build atomic while holding it.
        self._lock = threading.RLock()
        self._atomic: Protocol | None = None
        self._content: ContentStore | None = None
        self._kg: KGStore | None = None
        self._context_builder: ContextBuilder | None = None
        self._query: ContextQuery | None = None

    def _lazy(self, attr: str, build: Callable[[], _T]) -> _T:
        """Return the cached attribute, building it once under the view's lock."""
        value = getattr(self, attr)
        if value is None:
            with self._lock:
                value = getattr(self, attr)
                if value is None:
                    value = build()
                    setattr(self, attr, value)
        return value  # type: ignore[no-any-return]

    def _session(self) -> tuple[Callable[[], Any], str | None]:
        return self._parent._schema_session(self._context.schema_name)

    @property
    def context(self) -> ClientContext:
        """Isolation context of this view."""
        return self._context

    @property
    def context_key(self) -> str | None:
        """Context key of this view."""
        return self._context.context_key

    @property
    def atomic(self) -> Protocol:
        """Atomic memory scoped to this view's context_key."""
        return self._lazy("_atomic", self._build_atomic)

    @property
    def content(self) -> ContentStore:
        """Content storage scoped to this view's context_key."""
        return self._lazy("_content", self._build_content)

    @property
    def kg(self) -> KGStore:
        """Knowledge graph storage in this view's schema (KG has no context_key)."""
        return self._lazy("_kg", self._build_kg)

    @property
    def context_builder(self) -> ContextBuilder:
        """Context builder over this view's assertions."""
        return self._lazy("_context_builder", self._build_context_builder)

    @property
    def query(self) -> ContextQuery:
        """
        Context-aware query interface over this view's facts.

        Raises:
            RuntimeError: If the parent Client has no llm_client.
        """
        return self._lazy("_query", self._build_query)

    def _build_atomic(self) -> Protocol:
        session, schema = self._session()
        parent = self._parent
        return Protocol(
            parent._lg,
            session,
            self._context.context_key,
            embedder=parent._embedder,
            embedding_factory=parent._embedding_factory,
            embedding_format=parent._embedding_format,
            embedding_dimensions=parent._embedding_dimensions,
            embedding_schema=schema,
        )

    def _build_content(self) -> ContentStore:
        session, _ = self._session()
        return ContentStore(session, self._context.context_key)

    def _build_kg(self) -> KGStore:
        parent = self._parent
        if parent._is_own_schema(self._context.schema_name):
            return parent._kg
        session, schema = self._session()
        return KGStore(
            parent._lg,
            session,
            embedder=parent._embedder,
            embedding_factory=parent._embedding_factory,
            embedding_format=parent._embedding_format,
            embedding_schema=schema,
        )

    def _build_context_builder(self) -> ContextBuilder:
        return ContextBuilder(self.atomic.assertions)

    def _build_query(self) -> ContextQuery:
        parent = self._parent
        if parent._llm_client is None:
            raise RuntimeError("LLM client not configured. Pass llm_client to Client.")
        return ContextQuery(
            client=parent._llm_client,
            context_builder=self.context_builder,
            base_system_prompt=parent._get_default_system_prompt(),
            embedder=parent._embedder,
            embedding_adapter=self.atomic.embeddings,
        )
