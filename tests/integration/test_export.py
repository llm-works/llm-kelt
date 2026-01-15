"""Tests for training data export functions."""

import json
import tempfile
from pathlib import Path

from llm_learn.training import (
    export_feedback_classifier,
    export_feedback_sft,
    export_preferences_dpo,
)


class TestExportPreferencesDPO:
    """Test export_preferences_dpo function."""

    def test_export_basic(self, learn_client, database, clean_tables):
        """Test basic DPO export."""
        # Create test data
        learn_client.preferences.record(
            context="Summarize this article",
            chosen="Concise summary",
            rejected="Verbose essay",
            domain="synthesis",
        )
        learn_client.preferences.record(
            context="Explain quantum computing",
            chosen="Simple explanation",
            rejected="Overly complex explanation",
            domain="explanation",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dpo.jsonl"
            result = export_preferences_dpo(
                database.session,
                learn_client.profile_id,
                output_path,
            )

            assert result.count == 2
            assert result.format == "dpo"
            assert result.path == output_path
            assert output_path.exists()

            # Verify content
            lines = output_path.read_text().strip().split("\n")
            assert len(lines) == 2

            record = json.loads(lines[0])
            assert "prompt" in record
            assert "chosen" in record
            assert "rejected" in record
            assert record["prompt"] == "Summarize this article"
            assert record["chosen"] == "Concise summary"
            assert record["rejected"] == "Verbose essay"

    def test_export_filter_by_domain(self, learn_client, database, clean_tables):
        """Test export filtering by domain."""
        learn_client.preferences.record(context="A", chosen="G", rejected="B", domain="synthesis")
        learn_client.preferences.record(context="B", chosen="G", rejected="B", domain="analysis")
        learn_client.preferences.record(context="C", chosen="G", rejected="B", domain="synthesis")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dpo.jsonl"
            result = export_preferences_dpo(
                database.session,
                learn_client.profile_id,
                output_path,
                domain="synthesis",
            )

            assert result.count == 2

    def test_export_filter_by_margin(self, learn_client, database, clean_tables):
        """Test export filtering by minimum margin."""
        learn_client.preferences.record(context="A", chosen="G", rejected="B", margin=0.9)
        learn_client.preferences.record(context="B", chosen="G", rejected="B", margin=0.5)
        learn_client.preferences.record(context="C", chosen="G", rejected="B", margin=0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dpo.jsonl"
            result = export_preferences_dpo(
                database.session,
                learn_client.profile_id,
                output_path,
                min_margin=0.5,
            )

            assert result.count == 2

    def test_export_empty(self, learn_client, database, clean_tables):
        """Test export with no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dpo.jsonl"
            result = export_preferences_dpo(
                database.session,
                learn_client.profile_id,
                output_path,
            )

            assert result.count == 0
            assert output_path.exists()
            assert output_path.read_text() == ""

    def test_export_creates_parent_dirs(self, learn_client, database, clean_tables):
        """Test that export creates parent directories."""
        learn_client.preferences.record(context="A", chosen="G", rejected="B")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "dpo.jsonl"
            result = export_preferences_dpo(
                database.session,
                learn_client.profile_id,
                output_path,
            )

            assert result.count == 1
            assert output_path.exists()


class TestExportFeedbackSFT:
    """Test export_feedback_sft function."""

    def test_export_basic(self, learn_client, database, clean_tables):
        """Test basic SFT export."""
        # Create feedback with content
        learn_client.feedback.record(
            content_text="This is a great article about AI.",
            signal="positive",
            strength=0.9,
            tags=["interesting", "ai"],
            comment="Very informative",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sft.jsonl"
            result = export_feedback_sft(
                database.session,
                learn_client.profile_id,
                output_path,
            )

            assert result.count == 1
            assert result.format == "sft"

            lines = output_path.read_text().strip().split("\n")
            record = json.loads(lines[0])

            assert "instruction" in record
            assert "output" in record
            assert record["output"] == "This is a great article about AI."
            assert "interesting" in record["instruction"]

    def test_export_filters_by_signal(self, learn_client, database, clean_tables):
        """Test that export filters by signal type."""
        learn_client.feedback.record(
            content_text="Good content",
            signal="positive",
            strength=0.9,
        )
        learn_client.feedback.record(
            content_text="Bad content",
            signal="negative",
            strength=0.9,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sft.jsonl"
            result = export_feedback_sft(
                database.session,
                learn_client.profile_id,
                output_path,
                signal="positive",
            )

            assert result.count == 1

    def test_export_filters_by_strength(self, learn_client, database, clean_tables):
        """Test that export filters by minimum strength."""
        learn_client.feedback.record(
            content_text="Strong positive",
            signal="positive",
            strength=0.9,
        )
        learn_client.feedback.record(
            content_text="Weak positive",
            signal="positive",
            strength=0.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sft.jsonl"
            result = export_feedback_sft(
                database.session,
                learn_client.profile_id,
                output_path,
                min_strength=0.5,
            )

            assert result.count == 1

    def test_export_with_context(self, learn_client, database, clean_tables):
        """Test export with context in Alpaca format."""
        learn_client.feedback.record(
            content_text="Response text",
            signal="positive",
            strength=0.9,
            context={"query": "What is AI?"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sft.jsonl"
            result = export_feedback_sft(
                database.session,
                learn_client.profile_id,
                output_path,
                include_context=True,
            )

            assert result.count == 1

            lines = output_path.read_text().strip().split("\n")
            record = json.loads(lines[0])

            assert "input" in record
            assert "What is AI?" in record["input"]


class TestExportFeedbackClassifier:
    """Test export_feedback_classifier function."""

    def test_export_basic(self, learn_client, database, clean_tables):
        """Test basic classifier export."""
        learn_client.feedback.record(
            content_text="Good content",
            signal="positive",
            strength=0.9,
        )
        learn_client.feedback.record(
            content_text="Bad content",
            signal="negative",
            strength=0.9,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "classifier.jsonl"
            result = export_feedback_classifier(
                database.session,
                learn_client.profile_id,
                output_path,
            )

            assert result.count == 2
            assert result.format == "classifier"

            lines = output_path.read_text().strip().split("\n")
            records = [json.loads(line) for line in lines]

            # Check labels
            labels = {r["text"]: r["label"] for r in records}
            assert labels["Good content"] == 1
            assert labels["Bad content"] == 0

    def test_export_excludes_dismiss(self, learn_client, database, clean_tables):
        """Test that dismiss signals are excluded."""
        learn_client.feedback.record(
            content_text="Positive",
            signal="positive",
            strength=0.9,
        )
        learn_client.feedback.record(
            content_text="Dismissed",
            signal="dismiss",
            strength=0.9,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "classifier.jsonl"
            result = export_feedback_classifier(
                database.session,
                learn_client.profile_id,
                output_path,
            )

            assert result.count == 1

    def test_export_filters_by_strength(self, learn_client, database, clean_tables):
        """Test that export filters by minimum strength."""
        learn_client.feedback.record(
            content_text="Strong",
            signal="positive",
            strength=0.9,
        )
        learn_client.feedback.record(
            content_text="Weak",
            signal="positive",
            strength=0.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "classifier.jsonl"
            result = export_feedback_classifier(
                database.session,
                learn_client.profile_id,
                output_path,
                min_strength=0.5,
            )

            assert result.count == 1
