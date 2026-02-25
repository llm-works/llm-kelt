"""Tests for training data export functions."""

import json
import tempfile
from pathlib import Path

from llm_kelt.training import export_feedback_classifier, export_feedback_sft
from llm_kelt.training.dpo import export_preferences


class TestExportPreferencesDPO:
    """Test export_preferences function."""

    def test_export_basic(self, kelt_client, database, clean_tables):
        """Test basic DPO export."""
        # Create test data
        kelt_client.atomic.preferences.record(
            context="Summarize this article",
            chosen="Concise summary",
            rejected="Verbose essay",
            category="synthesis",
        )
        kelt_client.atomic.preferences.record(
            context="Explain quantum computing",
            chosen="Simple explanation",
            rejected="Overly complex explanation",
            category="explanation",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dpo.jsonl"
            result = export_preferences(
                database.session,
                kelt_client.context.context_key,
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

    def test_export_filter_by_category(self, kelt_client, database, clean_tables):
        """Test export filtering by category."""
        kelt_client.atomic.preferences.record(
            context="A", chosen="G", rejected="B", category="synthesis"
        )
        kelt_client.atomic.preferences.record(
            context="B", chosen="G", rejected="B", category="analysis"
        )
        kelt_client.atomic.preferences.record(
            context="C", chosen="G", rejected="B", category="synthesis"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dpo.jsonl"
            result = export_preferences(
                database.session,
                kelt_client.context.context_key,
                output_path,
                category="synthesis",
            )

            assert result.count == 2

    def test_export_filter_by_margin(self, kelt_client, database, clean_tables):
        """Test export filtering by minimum margin."""
        kelt_client.atomic.preferences.record(context="A", chosen="G", rejected="B", margin=0.9)
        kelt_client.atomic.preferences.record(context="B", chosen="G", rejected="B", margin=0.5)
        kelt_client.atomic.preferences.record(context="C", chosen="G", rejected="B", margin=0.3)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dpo.jsonl"
            result = export_preferences(
                database.session,
                kelt_client.context.context_key,
                output_path,
                min_margin=0.5,
            )

            assert result.count == 2

    def test_export_empty(self, kelt_client, database, clean_tables):
        """Test export with no data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "dpo.jsonl"
            result = export_preferences(
                database.session,
                kelt_client.context.context_key,
                output_path,
            )

            assert result.count == 0
            assert output_path.exists()
            assert output_path.read_text() == ""

    def test_export_creates_parent_dirs(self, kelt_client, database, clean_tables):
        """Test that export creates parent directories."""
        kelt_client.atomic.preferences.record(context="A", chosen="G", rejected="B")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "dpo.jsonl"
            result = export_preferences(
                database.session,
                kelt_client.context.context_key,
                output_path,
            )

            assert result.count == 1
            assert output_path.exists()


class TestExportFeedbackSFT:
    """Test export_feedback_sft function."""

    def test_export_basic(self, kelt_client, database, clean_tables):
        """Test basic SFT export."""
        # Create content first
        content_id = kelt_client.content.create(
            content_text="This is a great article about AI.",
            source="test",
            title="AI Overview",
        )

        # Record feedback on that content
        kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=content_id,
            strength=0.9,
            tags=["interesting", "ai"],
            comment="Very informative",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sft.jsonl"
            result = export_feedback_sft(
                database.session,
                kelt_client.context.context_key,
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

    def test_export_filters_by_signal(self, kelt_client, database, clean_tables):
        """Test that export filters by signal type."""
        # Create content
        good_id = kelt_client.content.create(
            content_text="Good content",
            source="test",
        )
        bad_id = kelt_client.content.create(
            content_text="Bad content",
            source="test",
        )

        kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=good_id,
            strength=0.9,
        )
        kelt_client.atomic.feedback.record(
            signal="negative",
            content_id=bad_id,
            strength=0.9,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sft.jsonl"
            result = export_feedback_sft(
                database.session,
                kelt_client.context.context_key,
                output_path,
                signal="positive",
            )

            assert result.count == 1

    def test_export_filters_by_strength(self, kelt_client, database, clean_tables):
        """Test that export filters by minimum strength."""
        # Create content
        strong_id = kelt_client.content.create(
            content_text="Strong positive",
            source="test",
        )
        weak_id = kelt_client.content.create(
            content_text="Weak positive",
            source="test",
        )

        kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=strong_id,
            strength=0.9,
        )
        kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=weak_id,
            strength=0.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sft.jsonl"
            result = export_feedback_sft(
                database.session,
                kelt_client.context.context_key,
                output_path,
                min_strength=0.5,
            )

            assert result.count == 1

    def test_export_with_context(self, kelt_client, database, clean_tables):
        """Test export with context in Alpaca format."""
        # Create content
        content_id = kelt_client.content.create(
            content_text="Response text",
            source="test",
        )

        kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=content_id,
            strength=0.9,
            context={"query": "What is AI?"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "sft.jsonl"
            result = export_feedback_sft(
                database.session,
                kelt_client.context.context_key,
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

    def test_export_basic(self, kelt_client, database, clean_tables):
        """Test basic classifier export."""
        # Create content
        good_id = kelt_client.content.create(
            content_text="Good content",
            source="test",
        )
        bad_id = kelt_client.content.create(
            content_text="Bad content",
            source="test",
        )

        kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=good_id,
            strength=0.9,
        )
        kelt_client.atomic.feedback.record(
            signal="negative",
            content_id=bad_id,
            strength=0.9,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "classifier.jsonl"
            result = export_feedback_classifier(
                database.session,
                kelt_client.context.context_key,
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

    def test_export_excludes_dismiss(self, kelt_client, database, clean_tables):
        """Test that dismiss signals are excluded."""
        # Create content
        good_id = kelt_client.content.create(
            content_text="Positive",
            source="test",
        )
        dismiss_id = kelt_client.content.create(
            content_text="Dismissed",
            source="test",
        )

        kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=good_id,
            strength=0.9,
        )
        kelt_client.atomic.feedback.record(
            signal="dismiss",
            content_id=dismiss_id,
            strength=0.9,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "classifier.jsonl"
            result = export_feedback_classifier(
                database.session,
                kelt_client.context.context_key,
                output_path,
            )

            assert result.count == 1

    def test_export_filters_by_strength(self, kelt_client, database, clean_tables):
        """Test that export filters by minimum strength."""
        # Create content
        strong_id = kelt_client.content.create(
            content_text="Strong",
            source="test",
        )
        weak_id = kelt_client.content.create(
            content_text="Weak",
            source="test",
        )

        kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=strong_id,
            strength=0.9,
        )
        kelt_client.atomic.feedback.record(
            signal="positive",
            content_id=weak_id,
            strength=0.3,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "classifier.jsonl"
            result = export_feedback_classifier(
                database.session,
                kelt_client.context.context_key,
                output_path,
                min_strength=0.5,
            )

            assert result.count == 1
