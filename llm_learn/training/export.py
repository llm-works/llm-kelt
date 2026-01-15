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

from ..core.models import Content, Feedback, PreferencePair
from ..core.utils import utc_now


@dataclass
class ExportResult:
    """Result of an export operation."""

    path: Path
    count: int
    profile_id: int
    format: str
    exported_at: datetime


def _build_sft_instruction(feedback: Feedback, content: Content) -> str:
    """Build instruction string from feedback tags, comment, and content title."""
    parts = []
    if feedback.tags:
        parts.append(f"Tags: {', '.join(feedback.tags)}")
    if feedback.comment:
        parts.append(feedback.comment)
    if content.title:
        parts.append(f"Topic: {content.title}")
    return " | ".join(parts) if parts else "Generate"


def _build_sft_record(
    feedback: Feedback, content: Content, include_context: bool
) -> dict[str, str] | None:
    """Build a single SFT training record from feedback and content.

    Returns None if content has no text (record would be useless for training).
    """
    if not content.content_text:
        return None
    record: dict[str, str] = {
        "instruction": _build_sft_instruction(feedback, content),
        "output": content.content_text,
    }
    if include_context and feedback.context:
        context_str = (
            json.dumps(feedback.context)
            if isinstance(feedback.context, dict)
            else str(feedback.context)
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


def _dpo_record_from_pair(pair: PreferencePair) -> dict[str, str]:
    """Convert preference pair to DPO training record."""
    return {"prompt": pair.context, "chosen": pair.chosen, "rejected": pair.rejected}


def _classifier_record_from_row(row: tuple[Feedback, Content]) -> dict[str, str | int] | None:
    """Convert feedback+content to classifier record. Returns None if no content text."""
    feedback, content = row
    if not content.content_text:
        return None
    return {"text": content.content_text, "label": 1 if feedback.signal == "positive" else 0}


def _sft_record_builder(include_context: bool):
    """Return a function that builds SFT records with the given context setting."""

    def builder(row: tuple[Feedback, Content]) -> dict[str, str] | None:
        return _build_sft_record(row[0], row[1], include_context)

    return builder


def _build_feedback_query(
    profile_id: int,
    min_strength: float,
    since: datetime | None,
    until: datetime | None,
    signals: list[str] | None = None,
):
    """Build SQLAlchemy query for feedback with content."""
    stmt = (
        select(Feedback, Content)
        .join(Content, Feedback.content_id == Content.id)
        .where(Feedback.profile_id == profile_id, Feedback.strength >= min_strength)
    )
    if signals is not None:
        stmt = stmt.where(Feedback.signal.in_(signals))
    if since is not None:
        stmt = stmt.where(Feedback.created_at >= since)
    if until is not None:
        stmt = stmt.where(Feedback.created_at <= until)
    return stmt.order_by(Feedback.created_at)


def export_preferences_dpo(
    session_factory: Callable[[], AbstractContextManager[Session]],
    profile_id: int,
    output_path: str | Path,
    *,
    domain: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    min_margin: float | None = None,
) -> ExportResult:
    """
    Export preference pairs in TRL DPO format.

    Output format (JSONL):
        {"prompt": str, "chosen": str, "rejected": str}

    Args:
        session_factory: Database session factory
        profile_id: Profile to export from
        output_path: Path to write JSONL file
        domain: Filter by domain (optional)
        since: Only export pairs created after this time
        until: Only export pairs created before this time
        min_margin: Only export pairs with margin >= this value

    Returns:
        ExportResult with path, count, and metadata
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with session_factory() as session:
        stmt = select(PreferencePair).where(PreferencePair.profile_id == profile_id)

        if domain is not None:
            stmt = stmt.where(PreferencePair.domain == domain)
        if since is not None:
            stmt = stmt.where(PreferencePair.created_at >= since)
        if until is not None:
            stmt = stmt.where(PreferencePair.created_at <= until)
        if min_margin is not None:
            stmt = stmt.where(PreferencePair.margin >= min_margin)

        stmt = stmt.order_by(PreferencePair.created_at)

        with output_path.open("w", encoding="utf-8") as f:
            count = _write_jsonl(f, session.scalars(stmt), _dpo_record_from_pair)

    return ExportResult(
        path=output_path,
        count=count,
        profile_id=profile_id,
        format="dpo",
        exported_at=utc_now(),
    )


def export_feedback_sft(
    session_factory: Callable[[], AbstractContextManager[Session]],
    profile_id: int,
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
        profile_id: Profile to export from
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
        stmt = _build_feedback_query(profile_id, min_strength, since, until, signals=[signal])

        with output_path.open("w", encoding="utf-8") as f:
            count = _write_jsonl(f, session.execute(stmt), _sft_record_builder(include_context))

    return ExportResult(
        path=output_path,
        count=count,
        profile_id=profile_id,
        format="sft",
        exported_at=utc_now(),
    )


def export_feedback_classifier(
    session_factory: Callable[[], AbstractContextManager[Session]],
    profile_id: int,
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
        profile_id: Profile to export from
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
            profile_id, min_strength, since, until, signals=["positive", "negative"]
        )
        with output_path.open("w", encoding="utf-8") as f:
            count = _write_jsonl(f, session.execute(stmt), _classifier_record_from_row)

    return ExportResult(
        path=output_path,
        count=count,
        profile_id=profile_id,
        format="classifier",
        exported_at=utc_now(),
    )
