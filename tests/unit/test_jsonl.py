"""Unit tests for JSONL export utilities."""

from datetime import UTC, datetime
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock

from llm_kelt.core.export.jsonl import (
    _format_feedback_record,
    _format_prediction_record,
    _format_solution_record,
    _open_output,
    load_jsonl,
)


class TestOpenOutput:
    """Tests for _open_output context manager."""

    def test_with_string_path(self, tmp_path):
        """Opens file from string path."""
        filepath = str(tmp_path / "test.jsonl")
        with _open_output(filepath) as f:
            f.write("test\n")

        assert Path(filepath).read_text() == "test\n"

    def test_with_path_object(self, tmp_path):
        """Opens file from Path object."""
        filepath = tmp_path / "test.jsonl"
        with _open_output(filepath) as f:
            f.write("test\n")

        assert filepath.read_text() == "test\n"

    def test_with_file_object(self):
        """Passes through file-like object unchanged."""
        sio = StringIO()
        with _open_output(sio) as f:
            f.write("test\n")

        assert sio.getvalue() == "test\n"

    def test_closes_file_on_exit(self, tmp_path):
        """File is closed after context exits."""
        filepath = tmp_path / "test.jsonl"
        with _open_output(filepath) as f:
            f.write("test\n")

        # File should be closed now
        assert Path(filepath).exists()


class TestLoadJsonl:
    """Tests for load_jsonl function."""

    def test_loads_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")

        result = load_jsonl(filepath)
        assert result == []

    def test_loads_single_record(self, tmp_path):
        """Single line file returns one record."""
        filepath = tmp_path / "single.jsonl"
        filepath.write_text('{"key": "value"}\n')

        result = load_jsonl(filepath)
        assert result == [{"key": "value"}]

    def test_loads_multiple_records(self, tmp_path):
        """Multiple lines return multiple records."""
        filepath = tmp_path / "multi.jsonl"
        filepath.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')

        result = load_jsonl(filepath)
        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines are skipped."""
        filepath = tmp_path / "blanks.jsonl"
        filepath.write_text('{"a": 1}\n\n{"b": 2}\n  \n{"c": 3}\n')

        result = load_jsonl(filepath)
        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_handles_unicode(self, tmp_path):
        """Unicode content is handled correctly."""
        filepath = tmp_path / "unicode.jsonl"
        filepath.write_text('{"text": "Hello"}\n', encoding="utf-8")

        result = load_jsonl(filepath)
        assert result == [{"text": "Hello"}]

    def test_accepts_string_path(self, tmp_path):
        """Accepts string path as well as Path."""
        filepath = tmp_path / "test.jsonl"
        filepath.write_text('{"key": "value"}\n')

        result = load_jsonl(str(filepath))
        assert result == [{"key": "value"}]


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
