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


def ensure_demo_profile(learn: LearnClient, profile_slug: str = "example") -> str:
    """Ensure demo workspace and profile exist, return profile_id.

    Args:
        learn: LearnClient instance (used for database access)
        profile_slug: Slug for the profile (default: "example")

    Returns:
        The profile_id (32-char hex hash) of the created/existing profile
    """
    from llm_learn.core.models import Profile, Workspace

    with learn.database.session() as session:
        workspace = session.query(Workspace).filter_by(slug="demo").first()
        if not workspace:
            workspace = Workspace(slug="demo", name="Demo Workspace")
            session.add(workspace)
            session.flush()

        profile = (
            session.query(Profile).filter_by(workspace_id=workspace.id, slug=profile_slug).first()
        )
        if not profile:
            profile = Profile(
                workspace_id=workspace.id,
                slug=profile_slug,
                name=f"{profile_slug.replace('-', ' ').title()} Profile",
            )
            session.add(profile)
            session.flush()

        session.commit()
        return profile.id
