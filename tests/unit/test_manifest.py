"""Unit tests for training manifest module."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_learn.training.manifest import Client, Deployment, Manifest
from llm_learn.training.manifest.errors import CorruptedManifestError

# Import private functions to test internal parsing logic directly
from llm_learn.training.manifest.loader import (
    _build_output_result,
    _to_number,
    load_manifest,
    save_manifest,
    validate_manifest,
)
from llm_learn.training.manifest.schema import Data, Source
from llm_learn.training.schema import Adapter


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


class TestLoadManifest:
    """Test load_manifest function."""

    def test_load_valid_manifest(self, tmp_path: Path):
        """Test loading a valid manifest file."""
        manifest_path = tmp_path / "test.yaml"
        manifest_path.write_text("""
adapter: my-adapter
method: sft
data:
  format: inline
  records:
    - prompt: "What is 2+2?"
      completion: "4"
""")

        manifest = load_manifest(manifest_path)

        assert manifest.adapter == "my-adapter"
        assert manifest.method == "sft"
        assert manifest.data.format == "inline"
        assert len(manifest.data.records) == 1
        assert manifest.data.records[0]["prompt"] == "What is 2+2?"
        # source_path should be set for resolving relative external data paths
        assert manifest.source_path == manifest_path.resolve()

    def test_load_dpo_manifest(self, tmp_path: Path):
        """Test loading a DPO manifest."""
        manifest_path = tmp_path / "dpo.yaml"
        manifest_path.write_text("""
adapter: dpo-adapter
method: dpo
data:
  format: inline
  records:
    - prompt: "Explain AI"
      chosen: "AI is..."
      rejected: "I don't know"
dpo:
  beta: 0.1
""")

        manifest = load_manifest(manifest_path)

        assert manifest.method == "dpo"
        assert manifest.method_config.get("beta") == 0.1

    def test_load_manifest_with_deployment(self, tmp_path: Path):
        """Test loading manifest with deployment section."""
        manifest_path = tmp_path / "test.yaml"
        manifest_path.write_text("""
adapter: test-adapter
method: sft
deployment:
  policy: add
data:
  format: inline
  records:
    - prompt: test
      completion: response
""")

        manifest = load_manifest(manifest_path)

        assert manifest.deployment is not None
        assert manifest.deployment.policy == "add"

    def test_load_manifest_with_parent(self, tmp_path: Path):
        """Test loading manifest with parent adapter."""
        manifest_path = tmp_path / "test.yaml"
        manifest_path.write_text("""
adapter: child-adapter
method: sft
parent:
  md5: abc123def456
  mtime: "2024-01-01T12:00:00"
  path: /path/to/parent
data:
  format: inline
  records:
    - prompt: test
      completion: response
""")

        manifest = load_manifest(manifest_path)

        assert manifest.parent is not None
        assert manifest.parent.md5 == "abc123def456"
        assert manifest.parent.path == "/path/to/parent"

    def test_load_missing_file_raises(self, tmp_path: Path):
        """Test that loading missing file raises FileNotFoundError."""
        manifest_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            load_manifest(manifest_path)

    def test_load_empty_file_raises(self, tmp_path: Path):
        """Test that loading empty file raises CorruptedManifestError."""
        manifest_path = tmp_path / "empty.yaml"
        manifest_path.touch()

        with pytest.raises(CorruptedManifestError, match="file is empty"):
            load_manifest(manifest_path)

    def test_load_invalid_yaml_raises(self, tmp_path: Path):
        """Test that invalid YAML structure raises error."""
        manifest_path = tmp_path / "invalid.yaml"
        # Use a list which YAML parses successfully but is not a dict
        manifest_path.write_text("- item1\n- item2\n")

        with pytest.raises(CorruptedManifestError, match="expected dict"):
            load_manifest(manifest_path)

    def test_load_missing_required_field_raises(self, tmp_path: Path):
        """Test that missing required field raises ValueError."""
        manifest_path = tmp_path / "incomplete.yaml"
        manifest_path.write_text("""
adapter: my-adapter
# missing method and data
""")

        with pytest.raises(ValueError, match="missing required field: method"):
            load_manifest(manifest_path)

    def test_load_gzipped_manifest(self, tmp_path: Path):
        """Test loading gzipped manifest."""
        import gzip

        manifest_path = tmp_path / "test.yaml.gz"
        content = """
adapter: gzip-adapter
method: sft
data:
  format: inline
  records:
    - prompt: test
      completion: response
"""
        with gzip.open(manifest_path, "wt", encoding="utf-8") as f:
            f.write(content)

        manifest = load_manifest(manifest_path)
        assert manifest.adapter == "gzip-adapter"


class TestSaveManifest:
    """Test save_manifest function."""

    def test_save_and_load_roundtrip(self, tmp_path: Path):
        """Test that save→load preserves manifest data."""
        original = Manifest(
            adapter="test-adapter",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
        )

        manifest_path = tmp_path / "manifest.yaml"
        save_manifest(original, manifest_path)

        loaded = load_manifest(manifest_path)

        assert loaded.adapter == original.adapter
        assert loaded.method == original.method
        assert loaded.data.format == original.data.format
        assert loaded.data.records == original.data.records

    def test_save_gzipped(self, tmp_path: Path):
        """Test saving gzipped manifest."""
        manifest = Manifest(
            adapter="test",
            method="sft",
            data=Data(format="inline", records=[{"prompt": "test", "completion": "response"}]),
        )

        manifest_path = tmp_path / "manifest.yaml"
        save_manifest(manifest, manifest_path, compress=True)

        # Should create .gz file
        gz_path = tmp_path / "manifest.yaml.gz"
        assert gz_path.exists()

        # Should be loadable
        loaded = load_manifest(gz_path)
        assert loaded.adapter == "test"


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

    def test_missing_timestamps_use_utc_now(self):
        """Test that missing timestamps use utc_now()."""
        data = {"status": "completed"}
        result = _build_output_result(data)

        assert result is not None
        # Should have timezone-aware datetime
        assert result.started_at.tzinfo is not None
        assert result.completed_at.tzinfo is not None


class TestManifestClient:
    """Test manifest Client class."""

    @pytest.fixture
    def lg(self):
        """Create a logger for testing."""
        from appinfra.log import LogConfig, LoggerFactory

        log_config = LogConfig.from_params(level="warning")
        return LoggerFactory.create_root(log_config)

    @pytest.fixture
    def storage(self, lg, tmp_path: Path):
        """Create FileStorage with temp directory."""
        from llm_learn.training.storage import FileStorage

        return FileStorage(lg, tmp_path)

    @pytest.fixture
    def client(self, lg, storage):
        """Create a manifest client with temp registry."""
        return Client(lg=lg, storage=storage)

    def test_create_sft_manifest(self, client: Client):
        """Test creating an SFT manifest."""
        manifest = client.create(
            adapter="test-sft",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
            model="Qwen/Qwen2.5-7B-Instruct",
        )

        assert manifest.adapter == "test-sft"
        assert manifest.method == "sft"
        assert manifest.training.get("requested_model") == "Qwen/Qwen2.5-7B-Instruct"
        assert len(manifest.data.records) == 1

    def test_create_dpo_manifest(self, client: Client):
        """Test creating a DPO manifest."""
        manifest = client.create(
            adapter="test-dpo",
            method="dpo",
            data=[{"prompt": "test", "chosen": "good", "rejected": "bad"}],
            config={"beta": 0.05},
        )

        assert manifest.adapter == "test-dpo"
        assert manifest.method == "dpo"
        assert manifest.method_config.get("beta") == 0.05

    def test_create_with_parent(self, client: Client):
        """Test creating manifest with parent adapter."""
        parent = Adapter(md5="abc123", mtime="2024-01-01T12:00:00", path="/path/to/parent")
        manifest = client.create(
            adapter="child",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
            parent=parent,
        )

        assert manifest.parent is not None
        assert manifest.parent.md5 == "abc123"

    def test_submit_and_list_pending(self, client: Client, tmp_path: Path):
        """Test submitting manifest to queue and listing pending."""
        manifest = client.create(
            adapter="queued",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
        )

        client.submit(manifest)

        pending = client.list_pending()
        assert len(pending) == 1
        assert pending[0].adapter == "queued"

    def test_submit_duplicate_raises(self, client: Client):
        """Test that submitting duplicate adapter key raises."""
        manifest = client.create(
            adapter="duplicate",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
        )

        client.submit(manifest)

        with pytest.raises(ValueError, match="already pending"):
            client.submit(manifest)

    def test_get_pending(self, client: Client):
        """Test getting a pending manifest by key."""
        manifest = client.create(
            adapter="lookup-test",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
        )
        client.submit(manifest)

        retrieved = client.get_pending("lookup-test")
        assert retrieved is not None
        assert retrieved.adapter == "lookup-test"

    def test_get_pending_not_found(self, client: Client):
        """Test getting non-existent pending manifest returns None."""
        result = client.get_pending("nonexistent")
        assert result is None

    def test_remove_pending(self, client: Client):
        """Test removing a pending manifest."""
        manifest = client.create(
            adapter="to-remove",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
        )
        client.submit(manifest)

        client.remove_pending("to-remove")

        assert client.get_pending("to-remove") is None

    def test_remove_pending_not_found_raises(self, client: Client):
        """Test that removing non-existent manifest raises."""
        with pytest.raises(FileNotFoundError, match="not found"):
            client.remove_pending("nonexistent")

    def test_default_profiles_applied(self, lg, tmp_path: Path):
        """Test that default profiles are merged into manifest config."""
        from llm_learn.training.storage import FileStorage

        default_profiles = {
            "sft": {"epochs": 5, "batch_size": 8, "learning_rate": 1e-4},
            "dpo": {"beta": 0.2, "epochs": 3},
        }
        storage = FileStorage(lg, tmp_path)
        client = Client(lg=lg, storage=storage, default_profiles=default_profiles)

        manifest = client.create(
            adapter="with-defaults",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
        )

        # epochs -> num_epochs mapping
        assert manifest.training.get("num_epochs") == 5
        assert manifest.training.get("batch_size") == 8
        assert manifest.training.get("learning_rate") == 1e-4

    def test_config_overrides_defaults(self, lg, tmp_path: Path):
        """Test that explicit config overrides default profile."""
        from llm_learn.training.storage import FileStorage

        default_profiles = {"sft": {"epochs": 5, "batch_size": 8}}
        storage = FileStorage(lg, tmp_path)
        client = Client(lg=lg, storage=storage, default_profiles=default_profiles)

        manifest = client.create(
            adapter="with-overrides",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
            config={"epochs": 10},  # Override default
        )

        assert manifest.training.get("num_epochs") == 10
        assert manifest.training.get("batch_size") == 8  # Default preserved

    def test_create_with_deployment_policy_skip(self, client: Client):
        """Test creating manifest with deployment_policy=skip."""
        manifest = client.create(
            adapter="test",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
            deployment_policy="skip",
        )

        assert manifest.deployment is not None
        assert manifest.deployment.policy == "skip"

    def test_create_with_deployment_policy_add(self, client: Client):
        """Test creating manifest with deployment_policy=add."""
        manifest = client.create(
            adapter="test",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
            deployment_policy="add",
        )

        assert manifest.deployment is not None
        assert manifest.deployment.policy == "add"

    def test_create_with_deployment_policy_replace(self, client: Client):
        """Test creating manifest with deployment_policy=replace."""
        manifest = client.create(
            adapter="test",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
            deployment_policy="replace",
        )

        assert manifest.deployment is not None
        assert manifest.deployment.policy == "replace"

    def test_create_without_deployment_policy(self, client: Client):
        """Test creating manifest without deployment_policy defaults to replace."""
        manifest = client.create(
            adapter="test",
            method="sft",
            data=[{"prompt": "test", "completion": "response"}],
        )

        # deployment should default to "replace" for visibility in YAML
        assert manifest.deployment is not None
        assert manifest.deployment.policy == "replace"
