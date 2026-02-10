"""Export functions for training data.

Exports collected data to formats suitable for training:
- DPO format for preference pairs (TRL DPOTrainer)
- SFT format for feedback (supervised fine-tuning)
"""

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.models import Content
from ..core.utils import utc_now
from ..memory.atomic.models import Fact, FeedbackDetails, PreferenceDetails
from ..memory.isolation import build_context_filter


@dataclass
class ExportResult:
    """Result of an export operation."""

    path: Path
    count: int
    context_key: str | None
    format: str
    exported_at: datetime


def _build_sft_instruction(fact: Fact, details: FeedbackDetails, content: Content) -> str:
    """Build instruction string from feedback tags, comment, and content title."""
    parts = []
    if details.tags:
        parts.append(f"Tags: {', '.join(details.tags)}")
    if details.comment:
        parts.append(details.comment)
    if content.title:
        parts.append(f"Topic: {content.title}")
    return " | ".join(parts) if parts else "Generate"


def _build_sft_record(
    fact: Fact, details: FeedbackDetails, content: Content, include_context: bool
) -> dict[str, str] | None:
    """Build a single SFT training record from feedback and content.

    Returns None if content has no text (record would be useless for training).
    """
    if not content.content_text:
        return None
    record: dict[str, str] = {
        "instruction": _build_sft_instruction(fact, details, content),
        "output": content.content_text,
    }
    if include_context and details.context:
        context_str = (
            json.dumps(details.context)
            if isinstance(details.context, dict)
            else str(details.context)
        )
        record["input"] = context_str
    return record


def _write_jsonl(f, records_iter, record_builder) -> int:
    """Write records to JSONL file, return count. Skips None records."""
    count = 0
    for item in records_iter:
        record = record_builder(item)
        if record is not None:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _dpo_record_from_row(row: tuple[Fact, PreferenceDetails]) -> dict[str, str]:
    """Convert preference pair to DPO training record."""
    fact, details = row
    return {"prompt": details.context, "chosen": details.chosen, "rejected": details.rejected}


def _classifier_record_from_row(
    row: tuple[Fact, FeedbackDetails, Content],
) -> dict[str, str | int] | None:
    """Convert feedback+content to classifier record. Returns None if no content text."""
    fact, details, content = row
    if not content.content_text:
        return None
    return {"text": content.content_text, "label": 1 if details.signal == "positive" else 0}


def _sft_record_builder(include_context: bool):
    """Return a function that builds SFT records with the given context setting."""

    def builder(row: tuple[Fact, FeedbackDetails, Content]) -> dict[str, str] | None:
        fact, details, content = row
        return _build_sft_record(fact, details, content, include_context)

    return builder


def _build_feedback_query(
    context_key: str | None,
    min_strength: float,
    since: datetime | None,
    until: datetime | None,
    signals: list[str] | None = None,
):
    """Build SQLAlchemy query for feedback with content."""
    stmt = (
        select(Fact, FeedbackDetails, Content)
        .join(FeedbackDetails, Fact.id == FeedbackDetails.fact_id)
        .join(Content, FeedbackDetails.content_id == Content.id)
        .where(
            Fact.type == "feedback",
            FeedbackDetails.strength >= min_strength,
        )
    )
    context_filter = build_context_filter(context_key, Fact.context_key)
    if context_filter is not None:
        stmt = stmt.where(context_filter)
    if signals is not None:
        stmt = stmt.where(FeedbackDetails.signal.in_(signals))
    if since is not None:
        stmt = stmt.where(Fact.created_at >= since)
    if until is not None:
        stmt = stmt.where(Fact.created_at <= until)
    return stmt.order_by(Fact.created_at)


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


def export_preferences_dpo(
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


def export_feedback_sft(
    session_factory: Callable[[], AbstractContextManager[Session]],
    context_key: str | None,
    output_path: str | Path,
    *,
    signal: str = "positive",
    min_strength: float = 0.5,
    since: datetime | None = None,
    until: datetime | None = None,
    include_context: bool = False,
) -> ExportResult:
    """
    Export feedback with content in SFT format.

    Output format (JSONL):
        {"instruction": str, "output": str}
        or with context:
        {"instruction": str, "input": str, "output": str}

    This exports feedback records that have associated content,
    formatted for supervised fine-tuning. The content text becomes
    the output, and any context/tags become the instruction.

    Note: Only feedback with associated content (content_id set) is exported.
    Feedback without content or with empty content_text is skipped.

    Args:
        session_factory: Database session factory
        context_key: Context key to scope export (None = all contexts)
        output_path: Path to write JSONL file
        signal: Feedback signal to export (default: "positive")
        min_strength: Minimum strength threshold (default: 0.5)
        since: Only export feedback created after this time
        until: Only export feedback created before this time
        include_context: Include context as "input" field in Alpaca format

    Returns:
        ExportResult with path, count, and metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with session_factory() as session:
        stmt = _build_feedback_query(context_key, min_strength, since, until, signals=[signal])

        with output_path.open("w", encoding="utf-8") as f:
            count = _write_jsonl(f, session.execute(stmt), _sft_record_builder(include_context))

    return ExportResult(
        path=output_path,
        count=count,
        context_key=context_key,
        format="sft",
        exported_at=utc_now(),
    )


def export_feedback_classifier(
    session_factory: Callable[[], AbstractContextManager[Session]],
    context_key: str | None,
    output_path: str | Path,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    min_strength: float = 0.5,
) -> ExportResult:
    """
    Export feedback for training a binary classifier.

    Output format (JSONL):
        {"text": str, "label": int}  # 1=positive, 0=negative

    Exports both positive and negative feedback for training
    a relevance/quality classifier.

    Note: Only feedback with associated content (content_id set) is exported.
    Feedback without content or with empty content_text is skipped.

    Args:
        session_factory: Database session factory
        context_key: Context key to scope export (None = all contexts)
        output_path: Path to write JSONL file
        since: Only export feedback created after this time
        until: Only export feedback created before this time
        min_strength: Minimum strength threshold (default: 0.5)

    Returns:
        ExportResult with path, count, and metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with session_factory() as session:
        stmt = _build_feedback_query(
            context_key, min_strength, since, until, signals=["positive", "negative"]
        )
        with output_path.open("w", encoding="utf-8") as f:
            count = _write_jsonl(f, session.execute(stmt), _classifier_record_from_row)

    return ExportResult(
        path=output_path,
        count=count,
        context_key=context_key,
        format="classifier",
        exported_at=utc_now(),
    )
