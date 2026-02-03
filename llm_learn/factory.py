"""Factory for creating LearnClient instances from configuration."""

from __future__ import annotations

from appinfra.db.pg import PG
from appinfra.dot_dict import DotDict
from appinfra.log import Logger
from llm_infer.client import Factory as LLMClientFactory
from llm_infer.client import LLMClient

from .client import LearnClient
from .core.database import Database
from .inference.embedder import Embedder


class LearnClientFactory:
    """Factory for creating LearnClient instances from configuration.

    Provides convenience methods to create fully-configured LearnClient
    instances from appinfra config objects.

    Usage:
        from appinfra.config import Config
        from appinfra.log import LogConfig, LoggerFactory
        from llm_learn import LearnClientFactory

        config = Config("etc/llm-learn.yaml")
        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))

        factory = LearnClientFactory(lg)
        client = factory.create_from_config(profile_id="a3f8...", config=config)
    """

    def __init__(self, lg: Logger) -> None:
        """
        Initialize factory with logger.

        Args:
            lg: Logger instance (shared across all created clients)
        """
        self._lg = lg

    def _create_embedder(self, config: DotDict) -> Embedder | None:
        """Create Embedder from config if embedding section exists."""
        embed_cfg = getattr(config, "embedding", None)
        if embed_cfg is None:
            return None
        return Embedder(base_url=embed_cfg.base_url, model=embed_cfg.model_name)

    def _create_llm_client(self, config: DotDict) -> LLMClient | None:
        """Create LLM client from config if llm section exists."""
        llm_cfg = getattr(config, "llm", None)
        if llm_cfg is None:
            return None
        llm_factory = LLMClientFactory(self._lg)
        return llm_factory.from_config(llm_cfg.to_dict())

    def create_from_config(
        self,
        profile_id: str,
        config: DotDict,
        db_key: str = "main",
        ensure_schema: bool = True,
    ) -> LearnClient:
        """
        Create LearnClient with all dependencies from config.

        Args:
            profile_id: Profile ID to scope all operations to
            config: Full application config (e.g., Config("etc/llm-learn.yaml"))
            db_key: Database config key (default: "main")
            ensure_schema: If True (default), auto-migrate schema on init

        Returns:
            Configured LearnClient instance

        Expected config structure:
            dbs:
              main: { url: "...", ... }
            llm:
              default: local
              backends:
                local: { base_url: "...", model: "..." }
            embedding:
              model_name: all-MiniLM-L6-v2
              base_url: http://localhost:8001/v1
            learn:
              memory:
                max_facts: 100
                min_confidence: 0.0
              default_system_prompt: ""
        """
        db = Database(self._lg, PG(self._lg, config.dbs[db_key]))
        return LearnClient(
            lg=self._lg,
            profile_id=profile_id,
            database=db,
            embedder=self._create_embedder(config),
            llm_client=self._create_llm_client(config),
            learn_config=getattr(config, "learn", None),
            ensure_schema=ensure_schema,
        )

    def create(
        self,
        profile_id: str,
        database: Database,
        embedder: Embedder | None = None,
        llm_client: LLMClient | None = None,
        learn_config: DotDict | None = None,
        ensure_schema: bool = True,
    ) -> LearnClient:
        """
        Create LearnClient with existing resources.

        Use this when sharing resources across multiple clients.

        Args:
            profile_id: Profile ID to scope all operations to
            database: Existing Database instance
            embedder: Optional existing Embedder instance
            llm_client: Optional existing LLM client instance
            learn_config: Optional learn settings (config.learn section)
            ensure_schema: If True (default), auto-migrate schema on init

        Returns:
            Configured LearnClient instance
        """
        return LearnClient(
            lg=self._lg,
            profile_id=profile_id,
            database=database,
            embedder=embedder,
            llm_client=llm_client,
            learn_config=learn_config,
            ensure_schema=ensure_schema,
        )
