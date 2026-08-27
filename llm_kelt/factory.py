# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Factory for creating Client instances from configuration."""

from __future__ import annotations

from appinfra.db.pg import PG
from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from llm_infer.client import ChatClient, EmbeddingClient
from llm_infer.client import Factory as LLMClientFactory

from .client import Client
from .core.database import Database
from .core.schema import SchemaMode
from .memory.isolation import ClientContext


class ClientFactory:
    """Factory for creating Client instances from configuration.

    Provides convenience methods to create fully-configured Client
    instances from appinfra config objects.

    Usage:
        from appinfra.config import Config
        from appinfra.log import LogConfig, LoggerFactory
        from llm_kelt import ClientFactory, ClientContext

        config = Config("etc/llm-kelt.yaml")
        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))

        factory = ClientFactory(lg)
        context = ClientContext(context_key="my-agent", schema_name="public")
        client = factory.create_from_config(context=context, config=config)
    """

    def __init__(self, lg: Logger) -> None:
        """
        Initialize factory with logger.

        Args:
            lg: Logger instance (shared across all created clients)
        """
        self._lg = lg

    def _create_embedder(self, config: DotDict) -> EmbeddingClient | None:
        """Create EmbeddingClient from config if embedding section exists."""
        embed_cfg = getattr(config, "embedding", None)
        if embed_cfg is None:
            return None
        factory = LLMClientFactory(self._lg)
        return factory.embeddings_from_config(embed_cfg.to_dict())

    def _create_llm_client(self, config: DotDict) -> ChatClient | None:
        """Create LLM client from config if llm section exists."""
        llm_cfg = getattr(config, "llm", None)
        if llm_cfg is None:
            return None
        llm_factory = LLMClientFactory(self._lg)
        return llm_factory.from_config(llm_cfg.to_dict())

    def create_from_config(
        self,
        context: ClientContext,
        config: DotDict,
        db_key: str = "main",
        schema_mode: SchemaMode = SchemaMode.ENSURE,
    ) -> Client:
        """
        Create Client with all dependencies from config.

        Args:
            context: ClientContext for data partitioning
            config: Full application config (e.g., Config("etc/llm-kelt.yaml"))
            db_key: Database config key (default: "main")
            schema_mode: How to handle the schema at construction. Defaults to
                SchemaMode.ENSURE. See ``Client`` for VERIFY and SKIP semantics.

        Returns:
            Configured Client instance

        Expected config structure:
            dbs:
              main: { url: "...", ... }
            llm:
              default: local
              backends:
                local: { base_url: "...", model: "..." }
            embedding:
              type: openai  # or "google"
              base_url: http://localhost:8001/v1
              model: all-MiniLM-L6-v2
              rate_limit:  # optional
                per_minute: 60
              retry:  # optional
                base: 1.0
                max_delay: 60
                timeout: 120
            kelt:
              memory:
                max_facts: 100
                min_confidence: 0.0
              default_system_prompt: ""
        """
        pg = PG(self._lg, config.dbs[db_key], schema=context.schema_name)
        db = Database(self._lg, pg)
        return Client(
            database=db,
            context=context,
            lg=self._lg,
            embedder=self._create_embedder(config),
            llm_client=self._create_llm_client(config),
            kelt_config=getattr(config, "kelt", None),
            training_config=getattr(config, "training", None),
            schema_mode=schema_mode,
        )

    def create(
        self,
        context: ClientContext,
        database: Database,
        embedder: EmbeddingClient | None = None,
        llm_client: ChatClient | None = None,
        kelt_config: DotDict | None = None,
        training_config: DotDict | None = None,
        schema_mode: SchemaMode = SchemaMode.ENSURE,
    ) -> Client:
        """
        Create Client with existing resources.

        Use this when sharing resources across multiple clients.

        Args:
            context: ClientContext for data partitioning
            database: Existing Database instance
            embedder: Optional existing EmbeddingClient instance
            llm_client: Optional existing LLM client instance
            kelt_config: Optional kelt settings (config.kelt section)
            training_config: Optional training settings (config.training section)
            schema_mode: How to handle the schema at construction. Defaults to
                SchemaMode.ENSURE. See ``Client`` for VERIFY and SKIP semantics.

        Returns:
            Configured Client instance
        """
        return Client(
            database=database,
            context=context,
            lg=self._lg,
            embedder=embedder,
            llm_client=llm_client,
            kelt_config=kelt_config,
            training_config=training_config,
            schema_mode=schema_mode,
        )
