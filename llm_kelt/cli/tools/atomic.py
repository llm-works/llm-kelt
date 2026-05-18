"""Atomic memory management CLI tools."""

from __future__ import annotations

from typing import Any

from appinfra.app.tools import Tool, ToolConfig


def _get_database(lg: Any, app_config: Any):
    """Create database connection from app config.

    Requires a database configured under dbs.default in the app config.
    """
    from llm_kelt.core.database import Database

    db_config = app_config.dbs.get("default")
    if db_config is None:
        raise RuntimeError(
            "No 'dbs.default' configuration found. "
            "The atomic CLI requires a database configured under 'dbs.default'."
        )

    from appinfra.db import PG

    pg = PG(lg, db_config)
    return Database(lg, pg)


def _get_embeddings_client(database):
    """Create embeddings client from database."""
    from llm_kelt.embedding import Config, QuantizationFormat, StoreClient

    config = Config(
        context_key="_cli",
        format=QuantizationFormat.F16,
        dimensions=384,
    )
    return StoreClient(config, database.session)


def _create_embedding_adapter(lg: Any, app_config: Any):
    """Create embedding adapter for vacuum operations."""
    from llm_kelt.memory.atomic import EmbeddingAdapter

    database = _get_database(lg, app_config)
    embeddings = _get_embeddings_client(database)
    return EmbeddingAdapter(
        session_factory=database.session,
        context_key=None,  # Vacuum all contexts
        embeddings=embeddings,
        embedder=None,
    )


class VacuumTool(Tool):
    """Clean up orphan embeddings from deleted facts."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent, ToolConfig(name="vacuum", help_text="Remove orphan embeddings"))

    def add_args(self, parser) -> None:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without deleting",
        )

    def run(self, **kwargs: Any) -> int:
        try:
            adapter = _create_embedding_adapter(self.lg, self.app.config)
        except Exception as e:
            self.lg.error("failed to connect to database", extra={"exception": e})
            print(f"Error: {e}")
            return 1

        return self._run_vacuum(adapter, self.args.dry_run)

    def _run_vacuum(self, adapter, dry_run: bool) -> int:
        """Execute vacuum and report results."""
        try:
            count = adapter.delete_orphans(dry_run=dry_run)
        except Exception as e:
            self.lg.error("vacuum failed", extra={"exception": e})
            print(f"Error: {e}")
            return 1

        if dry_run:
            print(f"Found {count} orphan embedding(s) (dry run, not deleted)")
        elif count > 0:
            print(f"Deleted {count} orphan embedding(s)")
        else:
            print("No orphan embeddings found")

        return 0


class AtomicTool(Tool):
    """Atomic memory management commands."""

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent, ToolConfig(name="atomic", help_text="Atomic memory commands"))
        self.add_tool(VacuumTool(self))

    def run(self, **kwargs: Any) -> int:
        # Delegate to subcommand group
        result: int = self.group.run(**kwargs)
        return result
