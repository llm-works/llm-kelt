# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for JSONL export utilities."""

from datetime import UTC, datetime
from unittest.mock import MagicMock

from llm_kelt.core.export.jsonl import (
    _format_feedback_record,
    _format_prediction_record,
    _format_solution_record,
)


class TestFormatFeedbackRecord:
    """Tests for _format_feedback_record helper."""

    def test_formats_complete_record(self):
        """Formats record with all fields present."""
        fact = MagicMock()
        fact.id = 123
        fact.created_at = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        details = MagicMock()
        details.signal = "positive"
        details.strength = 0.9
        details.tags = ["interesting", "relevant"]

        content = MagicMock()
        content.content_text = "This is the content"
        content.title = "Test Title"
        content.source = "test_source"

        result = _format_feedback_record(fact, details, content)

        assert result["id"] == 123
        assert result["text"] == "This is the content"
        assert result["title"] == "Test Title"
        assert result["label"] == "positive"
        assert result["strength"] == 0.9
        assert result["source"] == "test_source"
        assert result["tags"] == ["interesting", "relevant"]
        assert result["created_at"] == "2024-01-15T10:30:00+00:00"

    def test_handles_none_content(self):
        """Handles None content gracefully."""
        fact = MagicMock()
        fact.id = 456
        fact.created_at = None

        details = MagicMock()
        details.signal = "negative"
        details.strength = 0.5
        details.tags = []

        result = _format_feedback_record(fact, details, None)

        assert result["id"] == 456
        assert result["text"] is None
        assert result["title"] is None
        assert result["source"] is None
        assert result["created_at"] is None


class TestFormatPredictionRecord:
    """Tests for _format_prediction_record helper."""

    def test_formats_complete_record(self):
        """Formats prediction with all fields."""
        fact = MagicMock()
        fact.id = 789
        fact.content = "The market will rise"
        fact.confidence = 0.75
        fact.category = "markets"
        fact.created_at = datetime(2024, 2, 20, 14, 0, 0, tzinfo=UTC)

        details = MagicMock()
        details.status = "resolved"
        details.outcome = "correct"
        details.actual_result = "Market rose 5%"
        details.tags = ["finance"]
        details.resolution_date = datetime(2024, 3, 1).date()
        details.resolved_at = datetime(2024, 3, 1, 9, 0, 0, tzinfo=UTC)

        result = _format_prediction_record(fact, details)

        assert result["id"] == 789
        assert result["hypothesis"] == "The market will rise"
        assert result["confidence"] == 0.75
        assert result["status"] == "resolved"
        assert result["outcome"] == "correct"
        assert result["actual_result"] == "Market rose 5%"
        assert result["category"] == "markets"
        assert result["tags"] == ["finance"]
        assert result["resolution_date"] == "2024-03-01"
        assert result["resolved_at"] == "2024-03-01T09:00:00+00:00"

    def test_handles_none_dates(self):
        """Handles None dates gracefully."""
        fact = MagicMock()
        fact.id = 100
        fact.content = "Test"
        fact.confidence = 0.5
        fact.category = None
        fact.created_at = None

        details = MagicMock()
        details.status = "pending"
        details.outcome = None
        details.actual_result = None
        details.tags = None
        details.resolution_date = None
        details.resolved_at = None

        result = _format_prediction_record(fact, details)

        assert result["resolution_date"] is None
        assert result["resolved_at"] is None
        assert result["created_at"] is None


class TestFormatSolutionRecord:
    """Tests for _format_solution_record helper."""

    def test_formats_complete_record(self):
        """Formats solution with all fields."""
        fact = MagicMock()
        fact.id = 555
        fact.category = "code_review"
        fact.created_at = datetime(2024, 4, 10, 8, 0, 0, tzinfo=UTC)

        details = MagicMock()
        details.agent_name = "reviewer"
        details.problem = "Review this PR"
        details.problem_context = {"repo": "test/repo", "pr": 123}
        details.answer = {"approved": True}
        details.answer_text = "LGTM"
        details.tokens_used = 1500
        details.latency_ms = 2500
        details.tool_calls = [{"name": "read_file", "args": {"path": "main.py"}}]

        result = _format_solution_record(fact, details)

        assert result["id"] == 555
        assert result["agent_name"] == "reviewer"
        assert result["problem"] == "Review this PR"
        assert result["problem_context"] == {"repo": "test/repo", "pr": 123}
        assert result["answer"] == {"approved": True}
        assert result["answer_text"] == "LGTM"
        assert result["tokens_used"] == 1500
        assert result["latency_ms"] == 2500
        assert result["tool_calls"] == [{"name": "read_file", "args": {"path": "main.py"}}]
        assert result["category"] == "code_review"
        assert result["created_at"] == "2024-04-10T08:00:00+00:00"
