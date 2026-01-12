"""JSONL export utilities for training data."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from sqlalchemy import select

from ..database import Database
from ..models import Content, Feedback, Prediction, PreferencePair


@contextmanager
def _open_output(output: str | Path | TextIO) -> Generator[TextIO, None, None]:
    """Context manager for handling both file paths and file-like objects."""
    if isinstance(output, (str, Path)):
        f = open(output, "w", encoding="utf-8")
        try:
            yield f
        finally:
            f.close()
    else:
        yield output


def export_feedback(
    database: Database,
    output: str | Path | TextIO,
    since: datetime | None = None,
    signals: list[str] | None = None,
) -> int:
    """
    Export feedback with content for training.

    Exports records in JSONL format suitable for classifier training:
    {"text": "...", "label": "positive|negative", "strength": 0.9, ...}

    Args:
        database: Database instance
        output: Output file path or file-like object
        since: Only export feedback created after this time
        signals: List of signals to include (default: positive, negative)

    Returns:
        Number of records exported
    """
    if signals is None:
        signals = ["positive", "negative"]

    count = 0
    with _open_output(output) as f, database.session() as session:
        stmt = (
            select(Feedback, Content)
            .outerjoin(Content, Feedback.content_id == Content.id)
            .where(Feedback.signal.in_(signals))
        )
        if since:
            stmt = stmt.where(Feedback.created_at >= since)
        stmt = stmt.order_by(Feedback.created_at.asc())

        for feedback, content in session.execute(stmt).all():
            record = {
                "id": feedback.id,
                "text": content.content_text if content else None,
                "title": content.title if content else None,
                "label": feedback.signal,
                "strength": feedback.strength,
                "source": content.source if content else None,
                "tags": feedback.tags,
                "created_at": feedback.created_at.isoformat() if feedback.created_at else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def export_preferences(
    database: Database,
    output: str | Path | TextIO,
    domain: str | None = None,
    since: datetime | None = None,
) -> int:
    """
    Export preference pairs for DPO training.

    Exports records in JSONL format suitable for DPO training:
    {"prompt": "...", "chosen": "...", "rejected": "...", "margin": 0.7}

    Args:
        database: Database instance
        output: Output file path or file-like object
        domain: Only export pairs from this domain
        since: Only export pairs created after this time

    Returns:
        Number of records exported
    """
    count = 0
    with _open_output(output) as f, database.session() as session:
        stmt = select(PreferencePair)
        if domain:
            stmt = stmt.where(PreferencePair.domain == domain)
        if since:
            stmt = stmt.where(PreferencePair.created_at >= since)
        stmt = stmt.order_by(PreferencePair.created_at.asc())

        for pair in session.scalars(stmt).all():
            record = {
                "id": pair.id,
                "prompt": pair.context,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "margin": pair.margin,
                "domain": pair.domain,
                "created_at": pair.created_at.isoformat() if pair.created_at else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def export_predictions(  # cq: max-lines=35
    database: Database,
    output: str | Path | TextIO,
    status: str | None = None,
    domain: str | None = None,
    since: datetime | None = None,
) -> int:
    """
    Export predictions for calibration analysis.

    Exports records in JSONL format:
    {"hypothesis": "...", "confidence": 0.7, "outcome": "correct", ...}

    Args:
        database: Database instance
        output: Output file path or file-like object
        status: Filter by status (pending, resolved)
        domain: Only export predictions from this domain
        since: Only export predictions created after this time

    Returns:
        Number of records exported
    """
    count = 0
    with _open_output(output) as f, database.session() as session:
        stmt = select(Prediction)
        if status:
            stmt = stmt.where(Prediction.status == status)
        if domain:
            stmt = stmt.where(Prediction.domain == domain)
        if since:
            stmt = stmt.where(Prediction.created_at >= since)
        stmt = stmt.order_by(Prediction.created_at.asc())

        for pred in session.scalars(stmt).all():
            record = {
                "id": pred.id,
                "hypothesis": pred.hypothesis,
                "confidence": pred.confidence,
                "confidence_reasoning": pred.confidence_reasoning,
                "status": pred.status,
                "outcome": pred.outcome,
                "actual_result": pred.actual_result,
                "domain": pred.domain,
                "tags": pred.tags,
                "resolution_date": pred.resolution_date.isoformat()
                if pred.resolution_date
                else None,
                "created_at": pred.created_at.isoformat() if pred.created_at else None,
                "resolved_at": pred.resolved_at.isoformat() if pred.resolved_at else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    Load JSONL file into list of records.

    Args:
        path: Path to JSONL file

    Returns:
        List of record dictionaries
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
