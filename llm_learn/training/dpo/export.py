"""DPO export functions.

Exports preference data for DPO training:
- export_run_pairs: Export pending pairs for a run to TRL DPO format
- export_preferences: Export from atomic preferences (legacy)
- generate_pairs: Generate pairs for Client.assign_pairs()
"""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session, aliased

from ...core.utils import utc_now
from ...memory.atomic.models import Fact, PreferenceDetails
from ...memory.isolation import build_context_filter
from ..export import ExportResult

# Type alias for DPO pairs: (chosen_fact_id, rejected_fact_id, prompt)
PairTuple = tuple[int, int, str]


def _dpo_record_from_row(row: tuple[Fact, PreferenceDetails]) -> dict[str, str]:
    """Convert preference pair to DPO training record."""
    fact, details = row
    return {"prompt": details.context, "chosen": details.chosen, "rejected": details.rejected}


def _build_preferences_query(
    context_key: str | None,
    category: str | None,
    since: datetime | None,
    until: datetime | None,
    min_margin: float | None,
):
    """Build SQLAlchemy query for preference pairs."""
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
    if since is not None:
        stmt = stmt.where(Fact.created_at >= since)
    if until is not None:
        stmt = stmt.where(Fact.created_at <= until)
    if min_margin is not None:
        stmt = stmt.where(PreferenceDetails.margin >= min_margin)
    return stmt.order_by(Fact.created_at)


def _write_jsonl(f, records_iter, record_builder) -> int:
    """Write records to JSONL file, return count. Skips None records."""
    count = 0
    for item in records_iter:
        record = record_builder(item)
        if record is not None:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_preferences(
    session_factory: Callable[[], AbstractContextManager[Session]],
    context_key: str | None,
    output_path: str | Path,
    *,
    category: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    min_margin: float | None = None,
) -> ExportResult:
    """Export preference pairs in TRL DPO format: {prompt, chosen, rejected}."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with session_factory() as session:
        stmt = _build_preferences_query(context_key, category, since, until, min_margin)
        with output_path.open("w", encoding="utf-8") as f:
            count = _write_jsonl(f, session.execute(stmt), _dpo_record_from_row)

    return ExportResult(
        path=output_path, count=count, context_key=context_key, format="dpo", exported_at=utc_now()
    )


def export_run_pairs(
    session_factory: Callable[[], AbstractContextManager[Session]],
    run_id: int,
    output_path: str | Path,
) -> ExportResult:
    """Export pending pairs for a run in TRL DPO format: {prompt, chosen, rejected}.

    Reads from dpo_pending_pairs and joins atomic_facts to get chosen/rejected content.
    """
    from .client import PendingPair  # Import here to avoid circular import

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Alias Fact for chosen and rejected joins
    chosen_fact = aliased(Fact)
    rejected_fact = aliased(Fact)

    with session_factory() as session:
        stmt = (
            select(PendingPair.prompt, chosen_fact.content, rejected_fact.content)
            .join(chosen_fact, PendingPair.chosen_fact_id == chosen_fact.id)
            .join(rejected_fact, PendingPair.rejected_fact_id == rejected_fact.id)
            .where(PendingPair.run_id == run_id)
        )

        count = 0
        with output_path.open("w", encoding="utf-8") as f:
            for prompt, chosen, rejected in session.execute(stmt):
                record = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1

    return ExportResult(
        path=output_path, count=count, context_key=None, format="dpo", exported_at=utc_now()
    )


def generate_pairs(
    session_factory: Callable[[], AbstractContextManager[Session]],
    context_key: str | None,
    *,
    category: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    min_margin: float | None = None,
    exclude_pairs: set[tuple[int, int]] | None = None,
) -> list[PairTuple]:
    """Generate DPO pair tuples for run assignment.

    Returns pairs as (fact_id, fact_id, prompt) tuples suitable for
    Client.assign_pairs(). Currently uses the preference fact ID
    for both chosen and rejected (since the preference contains both
    text fields in a single fact).

    Args:
        session_factory: Database session factory.
        context_key: Context key to scope query (None = all contexts).
        category: Filter by category.
        since: Only include preferences created after this time.
        until: Only include preferences created before this time.
        min_margin: Minimum margin threshold.
        exclude_pairs: Set of (chosen_id, rejected_id) pairs to exclude.

    Returns:
        List of (chosen_fact_id, rejected_fact_id, prompt) tuples.
    """
    with session_factory() as session:
        stmt = _build_preferences_query(context_key, category, since, until, min_margin)
        results = []

        for row in session.execute(stmt):
            fact, details = row
            # Use fact ID for both chosen/rejected (same preference fact)
            pair_key = (fact.id, fact.id)
            if exclude_pairs and pair_key in exclude_pairs:
                continue
            results.append((fact.id, fact.id, details.context))

        return results
