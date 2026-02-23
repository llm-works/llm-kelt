"""Unit tests for FileStorage class."""

from __future__ import annotations

from pathlib import Path

import pytest
from appinfra.log import LogConfig, LoggerFactory

from llm_learn.training.storage import FileStorage


@pytest.fixture
def lg():
    """Create a logger for testing."""
    log_config = LogConfig.from_params(level="warning")
    return LoggerFactory.create_root(log_config)


@pytest.fixture
def storage(tmp_path: Path, lg):
    """Create FileStorage with temp directory."""
    return FileStorage(lg, tmp_path)


class TestDirectoryManagement:
    """Test directory creation and cleanup methods."""

    def test_ensure_directories_creates_all(self, storage: FileStorage):
        """Test that ensure_directories creates all required dirs."""
        storage.ensure_directories()

        assert storage.pending_path.exists()
        assert storage.completed_path.exists()
        assert storage.adapters_path.exists()
        assert storage.deployed_path.exists()
        assert storage.work_path.exists()

    def test_ensure_directories_idempotent(self, storage: FileStorage):
        """Test that ensure_directories can be called multiple times."""
        storage.ensure_directories()
        storage.ensure_directories()  # Should not raise

        assert storage.pending_path.exists()

    def test_ensure_work_dir_creates_directory(self, storage: FileStorage):
        """Test creating work directory for adapter."""
        work_dir = storage.ensure_work_dir("my-adapter")

        assert work_dir.exists()
        assert work_dir == storage.work_path / "my-adapter"

    def test_ensure_work_dir_cleans_existing(self, storage: FileStorage):
        """Test that ensure_work_dir removes existing content when clean=True."""
        work_dir = storage.ensure_work_dir("my-adapter")
        (work_dir / "old_file.txt").write_text("old content")

        work_dir = storage.ensure_work_dir("my-adapter", clean=True)

        assert work_dir.exists()
        assert not (work_dir / "old_file.txt").exists()

    def test_ensure_work_dir_preserves_when_clean_false(self, storage: FileStorage):
        """Test that ensure_work_dir preserves existing content when clean=False."""
        work_dir = storage.ensure_work_dir("my-adapter")
        (work_dir / "keep_me.txt").write_text("content")

        work_dir = storage.ensure_work_dir("my-adapter", clean=False)

        assert (work_dir / "keep_me.txt").exists()

    def test_clean_work_dir_removes_directory(self, storage: FileStorage):
        """Test removing work directory."""
        work_dir = storage.ensure_work_dir("my-adapter")
        assert work_dir.exists()

        storage.clean_work_dir("my-adapter")

        assert not work_dir.exists()

    def test_clean_work_dir_nonexistent_ok(self, storage: FileStorage):
        """Test that cleaning non-existent work dir doesn't raise."""
        storage.clean_work_dir("nonexistent")  # Should not raise


class TestManifestOperations:
    """Test manifest file operations."""

    def test_list_pending_empty(self, storage: FileStorage):
        """Test listing pending manifests when none exist."""
        result = storage.list_pending()
        assert result == []

    def test_list_pending_returns_yaml_files(self, storage: FileStorage):
        """Test listing pending manifests."""
        storage.ensure_directories()
        (storage.pending_path / "adapter1.yaml").write_text("adapter: adapter1")
        (storage.pending_path / "adapter2.yaml").write_text("adapter: adapter2")

        result = storage.list_pending()

        assert len(result) == 2
        assert result[0].name == "adapter1.yaml"
        assert result[1].name == "adapter2.yaml"

    def test_list_completed_empty(self, storage: FileStorage):
        """Test listing completed manifests when none exist."""
        result = storage.list_completed()
        assert result == []

    def test_list_completed_includes_gzip(self, storage: FileStorage):
        """Test listing completed manifests includes .yaml.gz files."""
        import gzip

        storage.ensure_directories()
        (storage.completed_path / "a.yaml").write_text("adapter: a")
        with gzip.open(storage.completed_path / "b.yaml.gz", "wt") as f:
            f.write("adapter: b")

        result = storage.list_completed()

        assert len(result) == 2
        names = [p.name for p in result]
        assert "a.yaml" in names
        assert "b.yaml.gz" in names

    def test_read_manifest_yaml(self, storage: FileStorage):
        """Test reading YAML manifest."""
        storage.ensure_directories()
        path = storage.pending_path / "test.yaml"
        path.write_text("adapter: test\nmethod: sft\n")

        data = storage.read_manifest(path)

        assert data["adapter"] == "test"
        assert data["method"] == "sft"

    def test_read_manifest_gzip(self, storage: FileStorage):
        """Test reading gzipped manifest."""
        import gzip

        storage.ensure_directories()
        path = storage.completed_path / "test.yaml.gz"
        with gzip.open(path, "wt") as f:
            f.write("adapter: gzipped\n")

        data = storage.read_manifest(path)

        assert data["adapter"] == "gzipped"

    def test_read_manifest_not_found(self, storage: FileStorage):
        """Test reading non-existent manifest raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Manifest not found"):
            storage.read_manifest(storage.pending_path / "nonexistent.yaml")

    def test_write_manifest_yaml(self, storage: FileStorage):
        """Test writing YAML manifest."""
        path = storage.base_path / "test.yaml"
        data = {"adapter": "test", "method": "sft"}

        result = storage.write_manifest(path, data)

        assert result == path
        assert path.exists()
        content = path.read_text()
        assert "adapter: test" in content

    def test_write_manifest_gzip(self, storage: FileStorage):
        """Test writing gzipped manifest."""
        path = storage.base_path / "test.yaml"
        data = {"adapter": "compressed"}

        result = storage.write_manifest(path, data, compress=True)

        assert result.suffix == ".gz"
        assert result.exists()

        # Verify readable
        loaded = storage.read_manifest(result)
        assert loaded["adapter"] == "compressed"

    def test_get_pending_path(self, storage: FileStorage):
        """Test getting pending manifest path."""
        path = storage.get_pending_path("my-adapter")
        assert path == storage.pending_path / "my-adapter.yaml"

    def test_pending_exists(self, storage: FileStorage):
        """Test checking if pending manifest exists."""
        storage.ensure_directories()
        assert not storage.pending_exists("test")

        (storage.pending_path / "test.yaml").write_text("adapter: test")
        assert storage.pending_exists("test")

    def test_remove_pending(self, storage: FileStorage):
        """Test removing pending manifest."""
        storage.ensure_directories()
        path = storage.pending_path / "to-remove.yaml"
        path.write_text("adapter: to-remove")

        storage.remove_pending("to-remove")

        assert not path.exists()

    def test_remove_pending_not_found(self, storage: FileStorage):
        """Test removing non-existent pending manifest raises."""
        storage.ensure_directories()
        with pytest.raises(FileNotFoundError, match="not in queue"):
            storage.remove_pending("nonexistent")

    def test_move_to_completed(self, storage: FileStorage):
        """Test moving manifest to completed directory."""
        storage.ensure_directories()
        source = storage.pending_path / "test.yaml"
        source.write_text("adapter: test\nmethod: sft\n")

        result = storage.move_to_completed(source, "test", "abc123")

        assert result.exists()
        assert "test-abc123" in result.name
        assert result.suffix == ".gz"  # Default compress=True
        assert not source.exists()  # Original deleted

    def test_move_to_completed_uncompressed(self, storage: FileStorage):
        """Test moving manifest without compression."""
        storage.ensure_directories()
        source = storage.pending_path / "test.yaml"
        source.write_text("adapter: test\n")

        result = storage.move_to_completed(source, "test", "abc123", compress=False)

        assert result.suffix == ".yaml"
        assert result.exists()


class TestAdapterStorage:
    """Test adapter storage operations."""

    def test_store_adapter(self, storage: FileStorage, tmp_path: Path):
        """Test storing adapter to registry."""
        # Create source adapter
        source = tmp_path / "source_adapter"
        source.mkdir()
        (source / "adapter_model.safetensors").write_text("weights")
        (source / "config.yaml").write_text("enabled: true")

        result = storage.store_adapter(source, "my-key", "20240101-120000-abc123")

        assert result.exists()
        assert (result / "adapter_model.safetensors").exists()
        assert result == storage.adapters_path / "my-key" / "20240101-120000-abc123"

    def test_store_adapter_overwrites_existing_version(self, storage: FileStorage, tmp_path: Path):
        """Test that storing adapter replaces existing version."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("v1")

        storage.store_adapter(source, "key", "v1")

        # Update source and re-store
        (source / "file.txt").write_text("v2")
        result = storage.store_adapter(source, "key", "v1")

        assert (result / "file.txt").read_text() == "v2"

    def test_store_adapter_source_not_found(self, storage: FileStorage, tmp_path: Path):
        """Test storing from non-existent source raises."""
        with pytest.raises(FileNotFoundError, match="not found"):
            storage.store_adapter(tmp_path / "nonexistent", "key", "v1")

    def test_remove_adapter_all_versions(self, storage: FileStorage, tmp_path: Path):
        """Test removing adapter removes all versions."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("data")

        storage.store_adapter(source, "key", "v1")
        storage.store_adapter(source, "key", "v2")

        storage.remove_adapter("key")

        assert not (storage.adapters_path / "key").exists()

    def test_remove_adapter_specific_version(self, storage: FileStorage, tmp_path: Path):
        """Test removing specific version keeps others."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "file.txt").write_text("data")

        storage.store_adapter(source, "key", "v1")
        storage.store_adapter(source, "key", "v2")

        storage.remove_adapter("key", "v1")

        assert not (storage.adapters_path / "key" / "v1").exists()
        assert (storage.adapters_path / "key" / "v2").exists()

    def test_remove_adapter_not_found(self, storage: FileStorage):
        """Test removing non-existent adapter raises."""
        with pytest.raises(ValueError, match="not found"):
            storage.remove_adapter("nonexistent")

    def test_remove_adapter_version_not_found(self, storage: FileStorage, tmp_path: Path):
        """Test removing non-existent version raises."""
        source = tmp_path / "source"
        source.mkdir()
        storage.store_adapter(source, "key", "v1")

        with pytest.raises(ValueError, match="Version.*not found"):
            storage.remove_adapter("key", "v99")

    def test_get_adapter_path(self, storage: FileStorage):
        """Test getting adapter path."""
        path = storage.get_adapter_path("my-key", "v1")
        assert path == storage.adapters_path / "my-key" / "v1"

    def test_list_adapters_empty(self, storage: FileStorage):
        """Test listing adapters when none exist."""
        assert storage.list_adapters() == []

    def test_list_adapters(self, storage: FileStorage, tmp_path: Path):
        """Test listing adapter keys."""
        source = tmp_path / "source"
        source.mkdir()
        storage.store_adapter(source, "adapter-a", "v1")
        storage.store_adapter(source, "adapter-b", "v1")

        result = storage.list_adapters()

        assert set(result) == {"adapter-a", "adapter-b"}

    def test_list_versions_empty(self, storage: FileStorage):
        """Test listing versions for non-existent adapter."""
        assert storage.list_versions("nonexistent") == []

    def test_list_versions(self, storage: FileStorage, tmp_path: Path):
        """Test listing versions for adapter."""
        source = tmp_path / "source"
        source.mkdir()
        storage.store_adapter(source, "key", "20240101-100000-aaa")
        storage.store_adapter(source, "key", "20240102-100000-bbb")

        result = storage.list_versions("key")

        assert result == ["20240101-100000-aaa", "20240102-100000-bbb"]

    def test_get_latest_version(self, storage: FileStorage, tmp_path: Path):
        """Test getting latest version (requires config.yaml)."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")

        storage.store_adapter(source, "key", "20240101-100000-aaa")
        storage.store_adapter(source, "key", "20240102-100000-bbb")

        result = storage.get_latest_version("key")

        assert result == "20240102-100000-bbb"

    def test_get_latest_version_no_versions(self, storage: FileStorage):
        """Test getting latest version when none exist."""
        storage.ensure_directories()
        (storage.adapters_path / "empty-key").mkdir(parents=True)

        assert storage.get_latest_version("empty-key") is None

    def test_adapter_exists(self, storage: FileStorage, tmp_path: Path):
        """Test checking if adapter exists."""
        source = tmp_path / "source"
        source.mkdir()

        assert not storage.adapter_exists("key")

        storage.store_adapter(source, "key", "v1")

        assert storage.adapter_exists("key")

    def test_version_exists(self, storage: FileStorage, tmp_path: Path):
        """Test checking if version exists."""
        source = tmp_path / "source"
        source.mkdir()
        storage.store_adapter(source, "key", "v1")

        assert storage.version_exists("key", "v1")
        assert not storage.version_exists("key", "v2")


class TestAdapterConfig:
    """Test adapter config read/write."""

    def test_read_adapter_config(self, storage: FileStorage, tmp_path: Path):
        """Test reading adapter config."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true\ndescription: Test")
        storage.store_adapter(source, "key", "v1")

        config = storage.read_adapter_config("key", "v1")

        assert config["enabled"] is True
        assert config["description"] == "Test"

    def test_read_adapter_config_not_found(self, storage: FileStorage, tmp_path: Path):
        """Test reading non-existent config returns empty dict."""
        source = tmp_path / "source"
        source.mkdir()
        storage.store_adapter(source, "key", "v1")

        config = storage.read_adapter_config("key", "v1")

        assert config == {}

    def test_write_adapter_config(self, storage: FileStorage, tmp_path: Path):
        """Test writing adapter config."""
        source = tmp_path / "source"
        source.mkdir()
        storage.store_adapter(source, "key", "v1")

        storage.write_adapter_config("key", "v1", {"enabled": True, "description": "New"})

        config = storage.read_adapter_config("key", "v1")
        assert config["enabled"] is True
        assert config["description"] == "New"


class TestDeployment:
    """Test deployment symlink operations."""

    def test_deploy_creates_versioned_symlink(self, storage: FileStorage, tmp_path: Path):
        """Test deploying adapter creates versioned symlink ({key}-{md5})."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true\nmd5: abc123def456")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "abc123def456"})

        storage.deploy("key", "v1")

        # New style: deployed/{key}-{md5}
        symlink = storage.deployed_path / "key-abc123def456"
        assert symlink.is_symlink()
        assert symlink.resolve() == storage.adapters_path / "key" / "v1"
        # Old style should not exist
        assert not (storage.deployed_path / "key").exists()

    def test_deploy_uses_latest_if_version_not_specified(
        self, storage: FileStorage, tmp_path: Path
    ):
        """Test deploy uses latest version when not specified."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "aaa111"})
        storage.store_adapter(source, "key", "v2")
        storage.write_adapter_config("key", "v2", {"md5": "bbb222"})

        storage.deploy("key")  # No version specified

        assert storage.get_deployed_version("key") == "v2"

    def test_deploy_replace_policy_removes_existing(self, storage: FileStorage, tmp_path: Path):
        """Test deploying with replace policy removes existing symlinks."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "aaa111"})
        storage.store_adapter(source, "key", "v2")
        storage.write_adapter_config("key", "v2", {"md5": "bbb222"})

        storage.deploy("key", "v1", policy="replace")
        assert storage.is_deployed("key", "aaa111")

        storage.deploy("key", "v2", policy="replace")
        assert storage.is_deployed("key", "bbb222")
        assert not storage.is_deployed("key", "aaa111")  # Old version removed

    def test_deploy_add_policy_keeps_existing(self, storage: FileStorage, tmp_path: Path):
        """Test deploying with add policy keeps existing symlinks."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "aaa111"})
        storage.store_adapter(source, "key", "v2")
        storage.write_adapter_config("key", "v2", {"md5": "bbb222"})

        storage.deploy("key", "v1", policy="add")
        storage.deploy("key", "v2", policy="add")

        # Both versions should be deployed
        assert storage.is_deployed("key", "aaa111")
        assert storage.is_deployed("key", "bbb222")

    def test_deploy_no_versions_raises(self, storage: FileStorage):
        """Test deploying adapter with no versions raises."""
        storage.ensure_directories()
        (storage.adapters_path / "empty").mkdir(parents=True)

        with pytest.raises(ValueError, match="No versions found"):
            storage.deploy("empty")

    def test_undeploy_removes_all_symlinks(self, storage: FileStorage, tmp_path: Path):
        """Test undeploying without md5 removes all versions."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "aaa111"})
        storage.store_adapter(source, "key", "v2")
        storage.write_adapter_config("key", "v2", {"md5": "bbb222"})

        storage.deploy("key", "v1", policy="add")
        storage.deploy("key", "v2", policy="add")

        storage.undeploy("key")

        assert not storage.is_deployed("key")
        assert not storage.is_deployed("key", "aaa111")
        assert not storage.is_deployed("key", "bbb222")

    def test_undeploy_specific_md5(self, storage: FileStorage, tmp_path: Path):
        """Test undeploying specific md5 version."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "aaa111"})
        storage.store_adapter(source, "key", "v2")
        storage.write_adapter_config("key", "v2", {"md5": "bbb222"})

        storage.deploy("key", "v1", policy="add")
        storage.deploy("key", "v2", policy="add")

        storage.undeploy("key", md5="aaa111")

        assert not storage.is_deployed("key", "aaa111")
        assert storage.is_deployed("key", "bbb222")

    def test_undeploy_nonexistent_ok(self, storage: FileStorage):
        """Test undeploying non-deployed adapter doesn't raise."""
        storage.ensure_directories()
        storage.undeploy("nonexistent")  # Should not raise

    def test_is_deployed_any_version(self, storage: FileStorage, tmp_path: Path):
        """Test is_deployed without md5 checks any version."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "abc123"})

        assert not storage.is_deployed("key")

        storage.deploy("key", "v1")

        assert storage.is_deployed("key")

    def test_is_deployed_specific_md5(self, storage: FileStorage, tmp_path: Path):
        """Test is_deployed with md5 checks specific version."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "abc123"})

        storage.deploy("key", "v1")

        assert storage.is_deployed("key", "abc123")
        assert not storage.is_deployed("key", "xyz789")

    def test_list_deployed_empty(self, storage: FileStorage):
        """Test list_deployed when nothing is deployed."""
        storage.ensure_directories()
        assert storage.list_deployed() == []

    def test_list_deployed_all(self, storage: FileStorage, tmp_path: Path):
        """Test list_deployed returns all deployed adapters."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "adapter-a", "v1")
        storage.write_adapter_config("adapter-a", "v1", {"md5": "aaa111"})
        storage.store_adapter(source, "adapter-b", "v1")
        storage.write_adapter_config("adapter-b", "v1", {"md5": "bbb222"})

        storage.deploy("adapter-a", "v1")
        storage.deploy("adapter-b", "v1")

        deployed = storage.list_deployed()
        assert ("adapter-a", "aaa111") in deployed
        assert ("adapter-b", "bbb222") in deployed

    def test_list_deployed_filtered_by_key(self, storage: FileStorage, tmp_path: Path):
        """Test list_deployed filtered by adapter key."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "adapter-a", "v1")
        storage.write_adapter_config("adapter-a", "v1", {"md5": "aaa111"})
        storage.store_adapter(source, "adapter-b", "v1")
        storage.write_adapter_config("adapter-b", "v1", {"md5": "bbb222"})

        storage.deploy("adapter-a", "v1")
        storage.deploy("adapter-b", "v1")

        deployed = storage.list_deployed("adapter-a")
        assert deployed == [("adapter-a", "aaa111")]

    def test_migrate_old_symlink(self, storage: FileStorage, tmp_path: Path):
        """Test migration of old-style symlink to versioned format."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "abc123"})

        # Create old-style symlink manually
        storage.deployed_path.mkdir(parents=True, exist_ok=True)
        old_symlink = storage.deployed_path / "key"
        old_symlink.symlink_to(Path("..") / "adapters" / "key" / "v1")

        # Deploy should migrate the old symlink
        storage.deploy("key", "v1")

        # Old symlink should be removed
        assert not old_symlink.exists()
        # New versioned symlink should exist
        assert storage.is_deployed("key", "abc123")

    def test_get_deployed_version(self, storage: FileStorage, tmp_path: Path):
        """Test getting deployed version ID."""
        source = tmp_path / "source"
        source.mkdir()
        (source / "config.yaml").write_text("enabled: true")
        storage.store_adapter(source, "key", "v1")
        storage.write_adapter_config("key", "v1", {"md5": "aaa111"})
        storage.store_adapter(source, "key", "v2")
        storage.write_adapter_config("key", "v2", {"md5": "bbb222"})

        storage.deploy("key", "v1")
        assert storage.get_deployed_version("key") == "v1"

        storage.deploy("key", "v2")
        assert storage.get_deployed_version("key") == "v2"

    def test_get_deployed_version_not_deployed(self, storage: FileStorage):
        """Test getting deployed version when not deployed returns None."""
        assert storage.get_deployed_version("nonexistent") is None


class TestDataFileOperations:
    """Test data file operations."""

    def test_write_data_file(self, storage: FileStorage):
        """Test writing JSONL data file."""
        work_dir = storage.ensure_work_dir("adapter")
        records = [
            {"prompt": "Hello", "response": "Hi"},
            {"prompt": "Bye", "response": "Goodbye"},
        ]

        path = storage.write_data_file(work_dir, "train.jsonl", records)

        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert '"prompt": "Hello"' in lines[0]

    def test_resolve_external_data_absolute(self, storage: FileStorage, tmp_path: Path):
        """Test resolving absolute external data path."""
        data_file = tmp_path / "data.jsonl"
        data_file.write_text('{"prompt": "test"}\n')

        result = storage.resolve_external_data(str(data_file))

        assert result == data_file

    def test_resolve_external_data_relative(self, storage: FileStorage, tmp_path: Path):
        """Test resolving relative external data path."""
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        data_file = manifest_dir / "data.jsonl"
        data_file.write_text('{"prompt": "test"}\n')

        result = storage.resolve_external_data("data.jsonl", manifest_dir)

        assert result == data_file

    def test_resolve_external_data_not_found(self, storage: FileStorage, tmp_path: Path):
        """Test resolving non-existent data file raises."""
        with pytest.raises(FileNotFoundError, match="Data file not found"):
            storage.resolve_external_data(str(tmp_path / "nonexistent.jsonl"))


class TestValidation:
    """Test key validation."""

    def test_validate_key_valid(self, storage: FileStorage):
        """Test that valid keys pass validation."""
        # Should not raise
        storage.validate_key("my-adapter")
        storage.validate_key("adapter_v2")
        storage.validate_key("123-test")

    def test_validate_key_slash(self, storage: FileStorage):
        """Test that forward slash is rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            storage.validate_key("../evil")

    def test_validate_key_backslash(self, storage: FileStorage):
        """Test that backslash is rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            storage.validate_key("evil\\path")

    def test_validate_key_dotdot(self, storage: FileStorage):
        """Test that .. is rejected."""
        with pytest.raises(ValueError, match="path traversal"):
            storage.validate_key("..adapter")


class TestIterationHelpers:
    """Test iteration helper methods."""

    def test_iter_adapter_keys(self, storage: FileStorage, tmp_path: Path):
        """Test iterating adapter keys with paths."""
        source = tmp_path / "source"
        source.mkdir()
        storage.store_adapter(source, "a", "v1")
        storage.store_adapter(source, "b", "v1")

        result = storage.iter_adapter_keys()

        keys = {k for k, _ in result}
        assert keys == {"a", "b"}

    def test_iter_adapter_keys_empty(self, storage: FileStorage):
        """Test iterating when no adapters exist."""
        result = storage.iter_adapter_keys()
        assert result == []
