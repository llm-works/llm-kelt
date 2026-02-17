"""Training client - aggregator for training method clients."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from appinfra.log import Logger

if TYPE_CHECKING:
    from .dpo import Client as DpoClient
    from .sft import Client as SftClient


class Client:
    """
    Aggregates all training method clients. Access via LearnClient.train.

    Usage:
        from llm_learn import LearnClient, IsolationContext
        context = IsolationContext(context_key="my-agent")
        learn = LearnClient(database=db, context=context)

        # Access training methods
        learn.train.dpo.create(adapter_name="my-adapter")
        learn.train.dpo.list_runs(status="pending")

        # SFT training
        learn.train.sft.create(adapter_name="my-sft-adapter")
    """

    def __init__(
        self,
        lg: Logger,
        session_factory: Callable[[], Any],
        context_key: str | None,
    ) -> None:
        """
        Initialize training client.

        Args:
            lg: Logger instance for all training operations.
            session_factory: Database session factory.
            context_key: Context key to scope all operations to (None = no filtering).
        """
        self._lg = lg
        self._session_factory = session_factory
        self._context_key = context_key

        # Lazy-initialized clients
        self._dpo: DpoClient | None = None
        self._sft: SftClient | None = None

    @property
    def dpo(self) -> DpoClient:
        """Access DPO training client."""
        if self._dpo is None:
            from .dpo import Client as DpoClient

            self._dpo = DpoClient(self._lg, self._session_factory, self._context_key)
        return self._dpo

    @property
    def sft(self) -> SftClient:
        """Access SFT training client."""
        if self._sft is None:
            from .sft import Client as SftClient

            self._sft = SftClient(self._lg, self._session_factory, self._context_key)
        return self._sft
