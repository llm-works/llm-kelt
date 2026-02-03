"""FastAPI application factory for llm-learn proxy server using appinfra."""

from appinfra.app.fastapi import Server, ServerBuilder
from llm_infer.client import LLMClient

from ..client import LearnClient
from ..core.database import Database
from ..inference.context import ContextBuilder
from .routes import create_router


def create_server(
    llm_config: dict,
    database: Database,
    profile_id: str,
    model_name: str = "llm-learn-proxy",
    host: str = "0.0.0.0",
    port: int = 8001,
) -> Server:
    """Create server with learning-enhanced LLM proxy.

    Args:
        llm_config: LLM backend configuration (see LLMClient.from_config).
        database: Database instance for learning data.
        profile_id: Profile ID (32-char hash) to load facts from.
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
            profile_id="a3f8b2c1d4e5f6a7b8c9d0e1f2a3b4c5",
        )
        server.start()
    """
    # Initialize clients
    llm_client = LLMClient.from_config(llm_config)
    learn_client = LearnClient(profile_id=profile_id, database=database)
    context_builder = ContextBuilder(learn_client.facts)

    # Create router with dependencies
    router = create_router(
        model_name=model_name,
        llm_client=llm_client,
        context_builder=context_builder,
    )

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


def _setup_database_from_config(config: dict) -> Database:
    """Create and migrate database from config."""
    from appinfra import DotDict
    from appinfra.db.pg import PG
    from appinfra.log import LogConfig, LoggerFactory

    log_config = LogConfig.from_params(level="info")
    logger = LoggerFactory.create_root(log_config)

    dbs = config.get("dbs", {})
    db_config = DotDict(**dbs.get("main", {}))
    pg = PG(logger, db_config)
    database = Database.from_pg(pg)
    database.migrate()
    return database


def create_server_from_config(config: dict) -> Server:
    """Create server from llm-learn.yaml configuration.

    Args:
        config: Configuration dict matching llm-learn.yaml structure:
            - llm: LLM backend configuration
            - dbs.main: Database configuration
            - proxy: (optional) Proxy-specific settings
            - profile_id: (optional) Profile ID to use

    Returns:
        Configured Server instance.
    """
    database = _setup_database_from_config(config)
    proxy_config = config.get("proxy", {})

    profile_id = proxy_config.get("profile_id") or config.get("profile_id")
    if profile_id is None or not str(profile_id).strip():
        raise ValueError("profile_id is required in config (proxy.profile_id or profile_id)")

    return create_server(
        llm_config=config.get("llm", {}),
        database=database,
        profile_id=str(profile_id),
        model_name=proxy_config.get("model_name", "llm-learn-proxy"),
        host=proxy_config.get("host", config.get("host", "0.0.0.0")),
        port=proxy_config.get("port", config.get("port", 8001)),
    )
