"""Shared helpers for example scripts."""

from llm_learn import LearnClient

# Terminal formatting
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Semantic colors
H1 = f"{BOLD}{BLUE}"
H2 = f"{BOLD}{MAGENTA}"
OK = GREEN
WARN = YELLOW
INFO = CYAN
MUTED = DIM
CMD = YELLOW
LLM_Q = f"{BOLD}{WHITE}"
LLM_A = GREEN


def psql_cmd(learn: LearnClient) -> str:
    """Build psql command from database config."""
    url = learn.database.engine.url
    return f"psql -h {url.host} -p {url.port} -U {url.username} -d {url.database}"


def get_demo_context_key(name: str = "example") -> str:
    """Generate a demo context key.

    Args:
        name: Name for the demo context (default: "example")

    Returns:
        A context key string in format "demo:name"
    """
    return f"demo:{name}"
