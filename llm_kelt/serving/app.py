"""FastAPI application factory for llm-kelt proxy server using appinfra."""

from appinfra.app.fastapi import Server, ServerBuilder
from appinfra.dot_dict import DotDict
from appinfra.log import LogConfig, Logger, LoggerFactory

from ..client import Client
from ..factory import ClientFactory
from .routes import create_router


def create_server(
    lg: Logger,
    kelt_client: Client,
    model_name: str = "llm-kelt-proxy",
    host: str = "0.0.0.0",
    port: int = 8001,
) -> Server:
    """Create server with kelt-enhanced LLM proxy.

    Args:
        lg: Logger instance.
        kelt_client: Configured Client instance.
        model_name: Model name to report in responses.
        host: Host to bind to.
        port: Port to bind to (default 8001, since llm-infer uses 8000).

    Returns:
        Configured Server instance.

    Example:
        from appinfra.config import Config
        from appinfra.log import LogConfig, LoggerFactory
        from llm_kelt import ClientContext, ClientFactory
        from llm_kelt.serving import create_server

        config = Config("etc/llm-kelt.yaml")
        lg = LoggerFactory.create_root(LogConfig.from_params(level="info"))
        factory = ClientFactory(lg)
        context = ClientContext(context_key="my-agent")
        kelt_client = factory.create_from_config(context=context, config=config)

        server = create_server(lg, kelt_client)
        server.start()
    """
    # Ensure LLM client is configured
    if kelt_client.llm_client is None:
        raise ValueError("Client must have llm_client configured for serving")

    # Create router with dependencies from kelt_client
    router = create_router(
        model_name=model_name,
        llm_client=kelt_client.llm_client,
        context_builder=kelt_client.context_builder,
    )

    return (
        ServerBuilder(lg, "llm-kelt-proxy")
        .with_host(host)
        .with_port(port)
        .with_title("llm-kelt Proxy")
        .with_description("OpenAI-compatible API with kelt context injection")
        .with_version("0.1.0")
        .routes.with_router(router)
        .done()
        .build()
    )


def create_server_from_config(config: dict, lg: Logger | None = None) -> Server:
    """Create server from llm-kelt.yaml configuration.

    Args:
        config: Configuration dict matching llm-kelt.yaml structure:
            - dbs.main: Database configuration
            - llm: LLM backend configuration
            - embedding: Embedding service configuration
            - kelt: Kelt-specific settings
            - proxy: Proxy-specific settings (host, port, context_key, model_name)
        lg: Optional logger. If not provided, creates one.

    Returns:
        Configured Server instance.
    """
    # Create logger if not provided
    if lg is None:
        log_config = LogConfig.from_params(level="info")
        lg = LoggerFactory.create_root(log_config)

    proxy_config = config.get("proxy", {})

    context_key = proxy_config.get("context_key") or config.get("context_key")
    if context_key is None or not str(context_key).strip():
        raise ValueError("context_key is required in config (proxy.context_key or context_key)")

    # Use factory to create Client from config
    from llm_kelt import ClientContext

    factory = ClientFactory(lg)
    context = ClientContext(context_key=str(context_key))
    kelt_client = factory.create_from_config(
        context=context,
        config=DotDict(**config),
    )

    return create_server(
        lg=lg,
        kelt_client=kelt_client,
        model_name=proxy_config.get("model_name", "llm-kelt-proxy"),
        host=proxy_config.get("host", config.get("host", "0.0.0.0")),
        port=proxy_config.get("port", config.get("port", 8001)),
    )
