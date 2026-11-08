# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for training manifest module."""

from __future__ import annotations

from llm_kelt.training.manifest import Deployment, Manifest

# Import private functions to test internal parsing logic directly
from llm_kelt.training.manifest.loader import (
    _build_output_result,
    _to_number,
    validate_manifest,
)
from llm_kelt.training.manifest.schema import Data, Source


class TestManifestSchema:
    """Test Manifest schema (DotDict-based, validation via validate_manifest)."""

    def test_valid_manifest(self):
        """Test creating a valid manifest."""
        manifest = Manifest(
            adapter="my-adapter",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
        )

        assert manifest.adapter == "my-adapter"
        assert manifest.method == "sft"
        assert manifest.data.format == "inline"
        assert len(manifest.data.records) == 1

    def test_empty_adapter_detected_by_validation(self):
        """Test that empty adapter is caught by validate_manifest."""
        manifest = Manifest(
            adapter="",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test"}]),
        )
        errors = validate_manifest(manifest)
        assert "adapter is required" in errors

    def test_invalid_method_detected_by_validation(self):
        """Test that invalid method is caught by validate_manifest."""
        manifest = Manifest(
            adapter="my-adapter",
            method="invalid",
            data=Data(format="inline", records=[{"prompt": "test"}]),
        )
        errors = validate_manifest(manifest)
        assert any("method must be" in e for e in errors)

    def test_path_traversal_slash_detected_by_validation(self):
        """Test that adapter with slash is caught by validate_manifest (path traversal)."""
        manifest = Manifest(
            adapter="../evil",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test"}]),
        )
        errors = validate_manifest(manifest)
        assert any("Invalid adapter" in e for e in errors)

    def test_path_traversal_backslash_detected_by_validation(self):
        """Test that adapter with backslash is caught by validate_manifest (path traversal)."""
        manifest = Manifest(
            adapter="evil\\path",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test"}]),
        )
        errors = validate_manifest(manifest)
        assert any("Invalid adapter" in e for e in errors)

    def test_path_traversal_dotdot_detected_by_validation(self):
        """Test that adapter with .. is caught by validate_manifest (path traversal)."""
        manifest = Manifest(
            adapter="..adapter",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test"}]),
        )
        errors = validate_manifest(manifest)
        assert any("Invalid adapter" in e for e in errors)

    def test_valid_adapter_with_hyphen_and_numbers(self):
        """Test that adapter with hyphens and numbers is valid."""
        manifest = Manifest(
            adapter="my-agent-v2",
            method="dpo",
            data=Data(format="inline", records=[{"prompt": "p", "chosen": "c", "rejected": "r"}]),
        )
        assert manifest.adapter == "my-agent-v2"
        errors = validate_manifest(manifest)
        assert errors == []


class TestDataSchema:
    """Test Data schema (DotDict-based, validation via validate_manifest)."""

    def test_inline_data_valid(self):
        """Test valid inline data."""
        data = Data(format="inline", records=[{"prompt": "test"}])
        assert data.format == "inline"
        assert len(data.records) == 1

    def test_inline_data_empty_records_detected_by_validation(self):
        """Test that inline data with empty records is caught by validate_manifest."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="inline", records=[]),
        )
        errors = validate_manifest(manifest)
        assert any("Inline data requires non-empty records" in e for e in errors)

    def test_external_data_valid(self):
        """Test valid external data."""
        data = Data(format="external", path="data/train.jsonl")
        assert data.format == "external"
        assert data.path == "data/train.jsonl"

    def test_external_data_no_path_detected_by_validation(self):
        """Test that external data without path is caught by validate_manifest."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="external"),
        )
        errors = validate_manifest(manifest)
        assert any("External data requires a path" in e for e in errors)


class TestSourceSchema:
    """Test Source schema (DotDict-based)."""

    def test_empty_source(self):
        """Test creating empty source."""
        source = Source()
        assert source.get("context_key") is None
        assert source.get("description") is None

    def test_source_with_values(self):
        """Test creating source with values."""
        source = Source(context_key="my-agent", description="Training for coding tasks")
        assert source.context_key == "my-agent"
        assert source.description == "Training for coding tasks"


class TestDeploymentSchema:
    """Test Deployment schema (DotDict-based, validation via validate_manifest)."""

    def test_empty_deployment(self):
        """Test creating empty deployment (uses default policy)."""
        deployment = Deployment()
        assert deployment.policy == "replace"  # Default policy

    def test_deployment_with_policy(self):
        """Test creating deployment with explicit policy."""
        deployment = Deployment(policy="add")
        assert deployment.policy == "add"

    def test_deployment_policy_skip(self):
        """Test deployment policy skip is valid."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
            deployment=Deployment(policy="skip"),
        )
        errors = validate_manifest(manifest)
        assert errors == []

    def test_deployment_policy_add(self):
        """Test deployment policy add is valid."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
            deployment=Deployment(policy="add"),
        )
        errors = validate_manifest(manifest)
        assert errors == []

    def test_deployment_policy_replace(self):
        """Test deployment policy replace is valid."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
            deployment=Deployment(policy="replace"),
        )
        errors = validate_manifest(manifest)
        assert errors == []

    def test_deployment_invalid_policy_detected_by_validation(self):
        """Test that invalid deployment policy is caught by validate_manifest."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
            deployment=Deployment(policy="invalid"),
        )
        errors = validate_manifest(manifest)
        assert any("deployment.policy must be" in e for e in errors)


class TestValidateManifest:
    """Test validate_manifest function."""

    def test_valid_manifest_returns_empty(self):
        """Test that valid manifest returns no errors."""
        manifest = Manifest(
            adapter="valid",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
        )

        errors = validate_manifest(manifest)
        assert errors == []

    def test_missing_adapter_returns_error(self):
        """Test that missing adapter is caught."""
        manifest = Manifest(
            adapter="",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test"}]),
            training={},
            lora={},
        )

        errors = validate_manifest(manifest)
        assert "adapter is required" in errors

    def test_invalid_method_returns_error(self):
        """Test that invalid method is caught."""
        manifest = Manifest(
            adapter="test",
            method="invalid",
            data=Data(format="inline", records=[{"prompt": "test"}]),
            training={},
            lora={},
        )

        errors = validate_manifest(manifest)
        assert any("method must be" in e for e in errors)

    def test_negative_epochs_returns_error(self):
        """Test that negative epochs is caught."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
            training={"num_epochs": -1},
        )

        errors = validate_manifest(manifest)
        assert "num_epochs must be positive" in errors

    def test_negative_batch_size_returns_error(self):
        """Test that negative batch size is caught."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
            training={"batch_size": 0},
        )

        errors = validate_manifest(manifest)
        assert "batch_size must be positive" in errors

    def test_string_number_validation(self):
        """Test that string numbers are coerced and validated correctly."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
            training={"learning_rate": "2e-4", "num_epochs": "3"},
        )

        errors = validate_manifest(manifest)
        assert errors == []  # String numbers should be coerced correctly


class TestToNumber:
    """Test _to_number helper function."""

    def test_int_passthrough(self):
        """Test that int passes through."""
        assert _to_number(42, 0) == 42

    def test_float_passthrough(self):
        """Test that float passes through."""
        assert _to_number(3.14, 0) == 3.14

    def test_string_int(self):
        """Test that string int is converted."""
        assert _to_number("42", 0) == 42.0

    def test_string_float(self):
        """Test that string float is converted."""
        assert _to_number("3.14", 0) == 3.14

    def test_scientific_notation(self):
        """Test that scientific notation is converted."""
        assert _to_number("2e-4", 0) == 2e-4
        assert _to_number("1e3", 0) == 1000.0

    def test_none_returns_default(self):
        """Test that None returns default."""
        assert _to_number(None, 99) == 99

    def test_invalid_string_returns_default(self):
        """Test that invalid string returns default."""
        assert _to_number("not a number", 99) == 99


class TestBuildOutputResult:
    """Test _build_output_result function."""

    def test_missing_status_preserved(self):
        """Test that missing status field is preserved as None (DotDict behavior)."""
        data = {
            "base_model": "test-model",
            "method": "sft",
        }
        result = _build_output_result(data)

        assert result is not None
        assert result.get("status") is None  # DotDict.get() returns None for missing keys

    def test_complete_output(self):
        """Test parsing complete output data."""
        data = {
            "status": "completed",
            "adapter": {"md5": "abc123", "mtime": "2024-01-01T12:00:00", "path": "/path"},
            "base_model": "Qwen/Qwen2.5-7B",
            "method": "sft",
            "metrics": {"loss": 0.5},
            "config": {"lora": {"r": 16}},
            "started_at": "2024-01-01T10:00:00",
            "completed_at": "2024-01-01T11:00:00",
            "samples_trained": 1000,
        }
        result = _build_output_result(data)

        assert result is not None
        assert result.status == "completed"
        assert result.adapter is not None
        assert result.adapter.md5 == "abc123"
        assert result.base_model == "Qwen/Qwen2.5-7B"
        assert result.samples_trained == 1000

    def test_missing_timestamps_default_to_now(self):
        """Test that missing timestamps default to current UTC time."""
        data = {"status": "completed"}
        result = _build_output_result(data)

        assert result is not None
        # Should have timezone-aware datetime
        assert result.started_at.tzinfo is not None
        assert result.completed_at.tzinfo is not None
