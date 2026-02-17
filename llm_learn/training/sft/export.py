"""SFT export functions.

Exports examples for SFT training:
- export_run_examples: Export pending examples for a run to TRL SFT format

Supports multiple fact types, dispatching to the appropriate details table:
- solution: problem -> instruction, answer_text -> output
- feedback: tags/comment -> instruction, content_text -> output
- interaction: query -> instruction, response -> output
"""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.models import Content
from ...core.utils import utc_now
from ...memory.atomic.models import Fact, FeedbackDetails, InteractionDetails, SolutionDetails
from ..export import ExportResult


def _export_solutions(session: Session, run_id: int) -> list[dict[str, str]]:
    """Export solution facts: problem -> instruction, answer (JSONB) -> output."""
    from .client import PendingExample

    stmt = (
        select(SolutionDetails)
        .join(PendingExample, PendingExample.fact_id == SolutionDetails.fact_id)
        .join(Fact, Fact.id == SolutionDetails.fact_id)
        .where(PendingExample.run_id == run_id)
        .where(Fact.type == "solution")
        .order_by(PendingExample.assigned_at)
    )
    records = []
    for (details,) in session.execute(stmt):
        if details.answer:
            # Use structured answer JSONB to preserve output format (e.g., {"text": ..., "style": ...})
            output = json.dumps(details.answer, ensure_ascii=False)
            records.append({"instruction": details.problem, "output": output})
    return records


def _export_feedback(session: Session, run_id: int) -> list[dict[str, str]]:
    """Export feedback facts: tags/comment -> instruction, content_text -> output."""
    from .client import PendingExample

    stmt = (
        select(FeedbackDetails, Content)
        .join(PendingExample, PendingExample.fact_id == FeedbackDetails.fact_id)
        .join(Fact, Fact.id == FeedbackDetails.fact_id)
        .join(Content, FeedbackDetails.content_id == Content.id)
        .where(PendingExample.run_id == run_id)
        .where(Fact.type == "feedback")
        .order_by(PendingExample.assigned_at)
    )
    records = []
    for details, content in session.execute(stmt):
        if content.content_text:
            parts = []
            if details.tags:
                parts.append(f"Tags: {', '.join(details.tags)}")
            if details.comment:
                parts.append(details.comment)
            instruction = " | ".join(parts) if parts else "Generate"
            records.append({"instruction": instruction, "output": content.content_text})
    return records


def _export_interactions(session: Session, run_id: int) -> list[dict[str, str]]:
    """Export interaction facts: query -> instruction, response -> output."""
    from .client import PendingExample

    stmt = (
        select(InteractionDetails)
        .join(PendingExample, PendingExample.fact_id == InteractionDetails.fact_id)
        .join(Fact, Fact.id == InteractionDetails.fact_id)
        .where(PendingExample.run_id == run_id)
        .where(Fact.type == "interaction")
        .order_by(PendingExample.assigned_at)
    )
    records = []
    for (details,) in session.execute(stmt):
        if details.response:
            records.append({"instruction": details.query, "output": details.response})
    return records


def export_run_examples(
    session_factory: Callable[[], AbstractContextManager[Session]],
    run_id: int,
    output_path: str | Path,
    include_context: bool = False,
) -> ExportResult:
    """Export pending examples for a run in SFT format: {instruction, output}.

    Reads from sft_pending_examples and dispatches to the appropriate details table
    based on fact type.

    Args:
        session_factory: Database session factory.
        run_id: The training run ID to export examples for.
        output_path: Path to write JSONL file.
        include_context: Reserved for future use.

    Returns:
        ExportResult with path, count, and metadata.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with session_factory() as session:
        # Collect records from all supported fact types
        records: list[dict[str, str]] = []
        records.extend(_export_solutions(session, run_id))
        records.extend(_export_feedback(session, run_id))
        records.extend(_export_interactions(session, run_id))

        # Write all records to JSONL
        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return ExportResult(
        path=output_path, count=len(records), context_key=None, format="sft", exported_at=utc_now()
    )
