"""FastAPI application factory for llm-learn proxy server using appinfra."""

from appinfra.app.fastapi import Server, ServerBuilder
from appinfra.dot_dict import DotDict
from appinfra.log import LogConfig, Logger, LoggerFactory

from ..client import LearnClient
from ..factory import LearnClientFactory
from .routes import create_router


def create_server(
    learn_client: LearnClient,
    model_name: str = "llm-learn-proxy",
    host: str = "0.0.0.0",
    port: int = 8001,
) -> Server:
    """Create server with learning-enhanced LLM proxy.

    Args:
        learn_client: Configured LearnClient instance.
        model_name: Model name to report in responses.
        host: Host to bind to.
        port: Port to bind to (default 8001, since llm-infer uses 8000).

    Returns:
        Configured Server instance.

    Example:
        from appinfra.config import Config
        from appinfra.log import LogConfig, LoggerFactory
        from llm_learn import LearnClientFactory
        from llm_learn.serving import create_server

        config = Config("etc/llm-learn.yaml")
        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))
        factory = LearnClientFactory(lg)
        learn_client = factory.create_from_config(profile_id="a3f8...", config=config)

        server = create_server(learn_client)
        server.start()
    """
    # Ensure LLM client is configured
    if learn_client.llm_client is None:
        raise ValueError("LearnClient must have llm_client configured for serving")

    # Create router with dependencies from learn_client
    router = create_router(
        model_name=model_name,
        llm_client=learn_client.llm_client,
        context_builder=learn_client.context_builder,
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


def create_server_from_config(config: dict, lg: Logger | None = None) -> Server:
    """Create server from llm-learn.yaml configuration.

    Args:
        config: Configuration dict matching llm-learn.yaml structure:
            - dbs.main: Database configuration
            - llm: LLM backend configuration
            - embedding: Embedding service configuration
            - learn: Learn-specific settings
            - proxy: Proxy-specific settings (host, port, profile_id, model_name)
        lg: Optional logger. If not provided, creates one.

    Returns:
        Configured Server instance.
    """
    # Create logger if not provided
    if lg is None:
        log_config = LogConfig.from_params(level="info")
        lg = LoggerFactory.create_root(log_config)

    proxy_config = config.get("proxy", {})

    profile_id = proxy_config.get("profile_id") or config.get("profile_id")
    if profile_id is None or not str(profile_id).strip():
        raise ValueError("profile_id is required in config (proxy.profile_id or profile_id)")

    # Use factory to create LearnClient from config
    factory = LearnClientFactory(lg)
    learn_client = factory.create_from_config(
        profile_id=str(profile_id),
        config=DotDict(**config),
    )

    return create_server(
        learn_client=learn_client,
        model_name=proxy_config.get("model_name", "llm-learn-proxy"),
        host=proxy_config.get("host", config.get("host", "0.0.0.0")),
        port=proxy_config.get("port", config.get("port", 8001)),
    )
