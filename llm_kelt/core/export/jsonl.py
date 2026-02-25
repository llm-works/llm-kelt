"""JSONL export utilities for training data."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from sqlalchemy import select

from ...memory.atomic.models import (
    Fact,
    FeedbackDetails,
    PredictionDetails,
    PreferenceDetails,
    SolutionDetails,
)
from ..database import Database
from ..models import Content


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


def _format_feedback_record(fact: Fact, details: FeedbackDetails, content: Content | None) -> dict:
    """Format a feedback record for export."""
    return {
        "id": fact.id,
        "text": content.content_text if content else None,
        "title": content.title if content else None,
        "label": details.signal,
        "strength": details.strength,
        "source": content.source if content else None,
        "tags": details.tags,
        "created_at": fact.created_at.isoformat() if fact.created_at else None,
    }


def export_feedback(
    database: Database,
    output: str | Path | TextIO,
    since: datetime | None = None,
    signals: list[str] | None = None,
) -> int:
    """Export feedback for classifier training. Format: {text, label, strength, ...}"""
    if signals is None:
        signals = ["positive", "negative"]

    count = 0
    with _open_output(output) as f, database.session() as session:
        stmt = (
            select(Fact, FeedbackDetails, Content)
            .join(FeedbackDetails, Fact.id == FeedbackDetails.fact_id)
            .outerjoin(Content, FeedbackDetails.content_id == Content.id)
            .where(Fact.type == "feedback", FeedbackDetails.signal.in_(signals))
        )
        if since:
            stmt = stmt.where(Fact.created_at >= since)
        for fact, details, content in session.execute(stmt.order_by(Fact.created_at.asc())).all():
            # Skip rows without usable text content
            if content is None or not content.content_text:
                continue
            f.write(
                json.dumps(_format_feedback_record(fact, details, content), ensure_ascii=False)
                + "\n"
            )
            count += 1
    return count


def export_preferences(
    database: Database,
    output: str | Path | TextIO,
    category: str | None = None,
    since: datetime | None = None,
) -> int:
    """
    Export preference pairs for DPO training.

    Exports records in JSONL format suitable for DPO training:
    {"prompt": "...", "chosen": "...", "rejected": "...", "margin": 0.7}

    Args:
        database: Database instance
        output: Output file path or file-like object
        category: Only export pairs from this category
        since: Only export pairs created after this time

    Returns:
        Number of records exported
    """
    count = 0
    with _open_output(output) as f, database.session() as session:
        stmt = (
            select(Fact, PreferenceDetails)
            .join(PreferenceDetails, Fact.id == PreferenceDetails.fact_id)
            .where(Fact.type == "preference")
        )
        if category:
            stmt = stmt.where(Fact.category == category)
        if since:
            stmt = stmt.where(Fact.created_at >= since)
        stmt = stmt.order_by(Fact.created_at.asc())

        for fact, details in session.execute(stmt).all():
            record = {
                "id": fact.id,
                "prompt": details.context,
                "chosen": details.chosen,
                "rejected": details.rejected,
                "margin": details.margin,
                "category": fact.category,
                "created_at": fact.created_at.isoformat() if fact.created_at else None,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count


def _format_prediction_record(fact: Fact, details: PredictionDetails) -> dict:
    """Format a prediction record for export."""
    return {
        "id": fact.id,
        "hypothesis": fact.content,
        "confidence": fact.confidence,
        "status": details.status,
        "outcome": details.outcome,
        "actual_result": details.actual_result,
        "category": fact.category,
        "tags": details.tags,
        "resolution_date": details.resolution_date.isoformat() if details.resolution_date else None,
        "created_at": fact.created_at.isoformat() if fact.created_at else None,
        "resolved_at": details.resolved_at.isoformat() if details.resolved_at else None,
    }


def export_predictions(
    database: Database,
    output: str | Path | TextIO,
    status: str | None = None,
    category: str | None = None,
    since: datetime | None = None,
) -> int:
    """Export predictions for calibration. Format: {hypothesis, confidence, outcome, ...}"""
    count = 0
    with _open_output(output) as f, database.session() as session:
        stmt = (
            select(Fact, PredictionDetails)
            .join(PredictionDetails, Fact.id == PredictionDetails.fact_id)
            .where(Fact.type == "prediction")
        )
        if status:
            stmt = stmt.where(PredictionDetails.status == status)
        if category:
            stmt = stmt.where(Fact.category == category)
        if since:
            stmt = stmt.where(Fact.created_at >= since)
        for fact, details in session.execute(stmt.order_by(Fact.created_at.asc())).all():
            f.write(json.dumps(_format_prediction_record(fact, details), ensure_ascii=False) + "\n")
            count += 1
    return count


def _format_solution_record(fact: Fact, details: SolutionDetails) -> dict:
    """Format a solution record for export."""
    return {
        "id": fact.id,
        "agent_name": details.agent_name,
        "problem": details.problem,
        "problem_context": details.problem_context,
        "answer": details.answer,
        "answer_text": details.answer_text,
        "tokens_used": details.tokens_used,
        "latency_ms": details.latency_ms,
        "tool_calls": details.tool_calls,
        "category": fact.category,
        "created_at": fact.created_at.isoformat() if fact.created_at else None,
    }


def export_solutions(
    database: Database,
    output: str | Path | TextIO,
    agent_name: str | None = None,
    category: str | None = None,
    since: datetime | None = None,
) -> int:
    """Export solutions for SFT training. Format: {problem, answer, agent_name, ...}"""
    count = 0
    with _open_output(output) as f, database.session() as session:
        stmt = (
            select(Fact, SolutionDetails)
            .join(SolutionDetails, Fact.id == SolutionDetails.fact_id)
            .where(Fact.type == "solution")
        )
        if agent_name:
            stmt = stmt.where(SolutionDetails.agent_name == agent_name)
        if category:
            stmt = stmt.where(Fact.category == category)
        if since:
            stmt = stmt.where(Fact.created_at >= since)
        for fact, details in session.execute(stmt.order_by(Fact.created_at.asc())).all():
            f.write(json.dumps(_format_solution_record(fact, details), ensure_ascii=False) + "\n")
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
