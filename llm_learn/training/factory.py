"""Training client - aggregator for training sub-clients.

Provides unified access to manifest lifecycle and training methods.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from appinfra.log import Logger

from .storage import FileStorage

if TYPE_CHECKING:
    from .dpo import Client as DpoClient
    from .lora.registry import AdapterRegistry
    from .manifest import Client
    from .sft import Client as SftClient


class Factory:
    """Factory for training clients. Access via LearnClient.train.

    Provides access to:
    - train.manifest: Create, load, save, submit manifests
    - train.dpo: Execute DPO training from manifests
    - train.sft: Execute SFT training from manifests
    - train.registry: Adapter registry for listing/managing adapters

    Usage:
        from llm_learn import LearnClient, ClientContext

        learn = LearnClient(...)

        # Create and submit a manifest
        manifest = learn.train.manifest.create(
            adapter="coding-v1",
            method="dpo",
            model="Qwen/Qwen2.5-7B-Instruct",
            data=[{"prompt": "...", "chosen": "...", "rejected": "..."}],
        )
        learn.train.manifest.submit(manifest)

        # Train from manifest
        result = learn.train.dpo.train(manifest)

        # List adapters
        adapters = learn.train.registry.list()
    """

    def __init__(
        self,
        lg: Logger,
        registry_path: str | Path,
        default_profiles: dict[str, dict] | None = None,
    ) -> None:
        """Initialize training client.

        Args:
            lg: Logger instance.
            registry_path: Base path for adapter registry and manifest queues.
            default_profiles: Default training profiles by method.
        """
        self._lg = lg
        self._storage = FileStorage(lg, registry_path)
        self._default_profiles = default_profiles or {}

        # Lazy-initialized clients
        self._manifest: Client | None = None
        self._dpo: DpoClient | None = None
        self._sft: SftClient | None = None
        self._registry: AdapterRegistry | None = None

    @property
    def manifest(self) -> Client:
        """Access manifest lifecycle client."""
        if self._manifest is None:
            from .manifest import Client

            self._manifest = Client(self._lg, self._storage, self._default_profiles)
        return self._manifest

    @property
    def dpo(self) -> DpoClient:
        """Access DPO training client."""
        if self._dpo is None:
            from .dpo import Client as DpoClient

            self._dpo = DpoClient(self._lg, self._storage)
        return self._dpo

    @property
    def sft(self) -> SftClient:
        """Access SFT training client."""
        if self._sft is None:
            from .sft import Client as SftClient

            self._sft = SftClient(self._lg, self._storage)
        return self._sft

    @property
    def registry(self) -> AdapterRegistry:
        """Access adapter registry."""
        if self._registry is None:
            from .lora.registry import AdapterRegistry

            self._registry = AdapterRegistry(self._lg, self._storage)
        return self._registry
