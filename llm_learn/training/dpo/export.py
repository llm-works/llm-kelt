"""DPO export functions.

Exports atomic preferences to DPO training format.
"""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.utils import utc_now
from ...memory.atomic.models import Fact, PreferenceDetails
from ...memory.isolation import build_context_filter
from ..export import ExportResult


def _build_preferences_query(
    context_key: str | None,
    category: str | None,
    min_margin: float | None,
    since: datetime | None,
    until: datetime | None,
):
    """Build SQLAlchemy query for preference export."""
    stmt = (
        select(Fact, PreferenceDetails)
        .join(PreferenceDetails, Fact.id == PreferenceDetails.fact_id)
        .where(Fact.type == "preference")
    )

    context_filter = build_context_filter(context_key, Fact.context_key)
    if context_filter is not None:
        stmt = stmt.where(context_filter)
    if category is not None:
        stmt = stmt.where(Fact.category == category)
    if min_margin is not None:
        stmt = stmt.where(PreferenceDetails.margin >= min_margin)
    if since is not None:
        stmt = stmt.where(Fact.created_at >= since)
    if until is not None:
        stmt = stmt.where(Fact.created_at <= until)

    return stmt.order_by(Fact.created_at)


def export_preferences(
    session_factory: Callable[[], AbstractContextManager[Session]],
    context_key: str | None,
    output_path: str | Path,
    *,
    category: str | None = None,
    min_margin: float | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> ExportResult:
    """Export preference pairs to DPO training format (JSONL).

    Args:
        session_factory: Database session factory
        context_key: Context key to scope export (None = all contexts)
        output_path: Path to write JSONL file
        category: Only export preferences from this category
        min_margin: Minimum margin threshold
        since: Only export preferences created after this time
        until: Only export preferences created before this time

    Returns:
        ExportResult with path, count, and metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stmt = _build_preferences_query(context_key, category, min_margin, since, until)

    with session_factory() as session:
        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for _, details in session.execute(stmt):
                record = {
                    "prompt": details.context,
                    "chosen": details.chosen,
                    "rejected": details.rejected,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    return ExportResult(
        path=output_path, count=count, context_key=context_key, format="dpo", exported_at=utc_now()
    )
