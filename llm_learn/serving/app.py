"""FastAPI application factory for llm-learn proxy server using appinfra."""

from appinfra.app.fastapi import Server, ServerBuilder
from fastapi import APIRouter
from llm_infer.client import OpenAIClient

from ..client import LearnClient
from ..core.database import Database
from ..inference.client import LLMClient
from ..inference.context import ContextBuilder
from .routes import create_router


def _build_server(router: APIRouter, host: str, port: int) -> Server:
    """Build the FastAPI server with the given router."""
    return (
        ServerBuilder("llm-learn-proxy")
        .with_host(host)
        .with_port(port)
        .with_title("llm-learn Proxy")
        .with_description("OpenAI-compatible API with learning context injection")
        .with_version("0.1.0")
        .routes.with_router(router)
        .done()
        .build()
    )


def _create_streaming_client(llm_config: dict) -> OpenAIClient | None:
    """Create streaming client for OpenAI-compatible backends.

    Returns None if the default backend is not OpenAI-compatible.
    """
    default_backend = llm_config.get("default", "local")
    backends = llm_config.get("backends", {})
    backend_config = backends.get(default_backend, {})

    backend_type = backend_config.get("type", "openai_compatible")
    if backend_type not in ("openai_compatible", "openai"):
        return None

    return OpenAIClient(
        base_url=backend_config.get("base_url", "http://localhost:8000/v1"),
        api_key=backend_config.get("api_key"),
        timeout=backend_config.get("timeout", 120.0),
    )


def create_server(
    llm_config: dict,
    database: Database,
    profile_id: int,
    model_name: str = "llm-learn-proxy",
    host: str = "0.0.0.0",
    port: int = 8001,
) -> Server:
    """Create server with learning-enhanced LLM proxy.

    Args:
        llm_config: LLM backend configuration (see LLMClient.from_config).
        database: Database instance for learning data.
        profile_id: Profile ID to load facts from.
        model_name: Model name to report in responses.
        host: Host to bind to.
        port: Port to bind to (default 8001, since llm-infer uses 8000).

    Returns:
        Configured Server instance.

    Example:
        from llm_learn.serving import create_server
        from llm_learn.core.database import Database

        db = Database.from_config(db_config)
        server = create_server(
            llm_config={"default": "local", "backends": {...}},
            database=db,
            profile_id=1,
        )
        server.start()
    """
    # Initialize clients
    llm_client = LLMClient.from_config(llm_config)
    learn_client = LearnClient(profile_id=profile_id, database=database)
    context_builder = ContextBuilder(learn_client.facts)

    # Create streaming client from backend config (for OpenAI-compatible backends)
    streaming_client = _create_streaming_client(llm_config)

    # Create router with dependencies
    router = create_router(
        model_name=model_name,
        llm_client=llm_client,
        context_builder=context_builder,
        streaming_client=streaming_client,
    )

    server = _build_server(router, host, port)
    server._llm_client = llm_client  # Store for cleanup
    return server


def create_server_from_config(config: dict) -> Server:
    """Create server from learn.yaml configuration.

    Args:
        config: Configuration dict matching learn.yaml structure:
            - llm: LLM backend configuration
            - dbs.main: Database configuration
            - proxy: (optional) Proxy-specific settings
            - profile_id: (optional) Profile ID to use

    Returns:
        Configured Server instance.

    Example config (learn.yaml):
        llm:
          default: local
          backends:
            local:
              type: openai_compatible
              base_url: http://localhost:8000/v1
              model: qwen2.5-72b
        dbs:
          main:
            url: postgresql://user:pass@localhost/learn
        proxy:
          host: 0.0.0.0
          port: 8001
          profile_id: 1
          model_name: llm-learn-proxy
    """
    from appinfra import DotDict
    from appinfra.db.pg import PG
    from appinfra.log import LogConfig, LoggerFactory

    # Setup logging
    log_config = LogConfig.from_params(level="info")
    logger = LoggerFactory.create_root(log_config)

    # Setup database from dbs.main
    # Wrap in DotDict for attribute access (PG uses getattr for create_db, etc.)
    dbs = config.get("dbs", {})
    db_config = DotDict(**dbs.get("main", {}))
    pg = PG(logger, db_config)
    database = Database.from_pg(pg)
    database.migrate()  # Create database and tables if needed

    # Get proxy settings
    proxy_config = config.get("proxy", {})

    return create_server(
        llm_config=config.get("llm", {}),
        database=database,
        profile_id=proxy_config.get("profile_id", config.get("profile_id", 1)),
        model_name=proxy_config.get("model_name", "llm-learn-proxy"),
        host=proxy_config.get("host", config.get("host", "0.0.0.0")),
        port=proxy_config.get("port", config.get("port", 8001)),
    )
