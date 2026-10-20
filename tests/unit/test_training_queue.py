# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for the training queue: submit, claim, complete, fail, status, run_pending."""

from __future__ import annotations

import gzip
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml
from appinfra import DotDict

from llm_kelt.training.manifest import Client
from llm_kelt.training.manifest.loader import load_manifest
from llm_kelt.training.manifest.schema import Data, Manifest
from llm_kelt.training.runner import Runner
from llm_kelt.training.schema import Adapter, RunResult
from llm_kelt.training.storage import FileStorage


@pytest.fixture
def storage(lg, tmp_path: Path) -> FileStorage:
    return FileStorage(lg, tmp_path / "registry")


def _manifest(adapter: str = "foo", **data: object) -> Manifest:
    data = data or {"format": "inline", "records": [{"instruction": "i", "output": "o"}]}
    return Manifest(
        adapter=adapter,
        method="sft",
        data=Data(data),
        training=DotDict(requested_model="org/model-a"),
    )


def _result(status: str = "completed", md5: str = "abc123def456", error: str | None = None):
    now = datetime.now(UTC)
    adapter = Adapter(md5=md5, mtime="", path="/adapters/x") if status == "completed" else None
    return RunResult(status=status, started_at=now, completed_at=now, adapter=adapter, error=error)


def _complete(storage: FileStorage, adapter: str = "foo") -> None:
    manifest = storage.claim_pending_manifest(adapter)
    manifest.output = _result()
    storage.complete_manifest(manifest)


class TestSubmit:
    def test_writes_pending_without_leftover_temp_file(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        assert [p.name for p in storage.pending_path.iterdir()] == ["foo.yaml"]
        assert load_manifest(storage.pending_path / "foo.yaml").adapter == "foo"

    def test_second_submit_for_pending_key_raises(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        with pytest.raises(ValueError, match="already pending"):
            storage.submit_manifest(_manifest())

    def test_concurrent_submit_does_not_overwrite(self, storage: FileStorage, monkeypatch):
        """A pending file appearing between the check and the link is not overwritten."""
        real_link = os.link

        def racing_link(src, dst):
            Path(dst).write_text("other submission")
            return real_link(src, dst)

        monkeypatch.setattr("llm_kelt.training.storage.file.os.link", racing_link)
        with pytest.raises(ValueError, match="already pending"):
            storage.submit_manifest(_manifest())
        assert (storage.pending_path / "foo.yaml").read_text() == "other submission"
        assert [p.name for p in storage.pending_path.iterdir()] == ["foo.yaml"]


class TestListPendingKeys:
    def test_oldest_first_and_ignores_temp_files(self, storage: FileStorage):
        for key in ("b", "a"):
            storage.submit_manifest(_manifest(key))
        os.utime(storage.pending_path / "b.yaml", (1000, 1000))
        os.utime(storage.pending_path / "a.yaml", (2000, 2000))
        (storage.pending_path / ".c.123.tmp").write_text("partial")
        assert storage.list_pending_keys() == ["b", "a"]

    def test_empty(self, storage: FileStorage):
        assert storage.list_pending_keys() == []


class TestClaim:
    def test_moves_pending_to_running(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        manifest = storage.claim_pending_manifest("foo")
        assert manifest.adapter == "foo"
        assert not (storage.pending_path / "foo.yaml").exists()
        assert (storage.running_path / "foo.yaml").exists()

    def test_not_pending_raises_file_not_found(self, storage: FileStorage):
        with pytest.raises(FileNotFoundError):
            storage.claim_pending_manifest("foo")

    def test_already_running_raises_file_exists(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        storage.claim_pending_manifest("foo")
        storage.submit_manifest(_manifest())
        with pytest.raises(FileExistsError):
            storage.claim_pending_manifest("foo")
        assert (storage.pending_path / "foo.yaml").exists()

    def test_relative_external_path_anchored_to_pending(self, storage: FileStorage):
        storage.submit_manifest(_manifest(format="external", path="data/train.jsonl"))
        manifest = storage.claim_pending_manifest("foo")
        assert manifest.data.path == str(storage.pending_path / "data/train.jsonl")

    def test_absolute_external_path_unchanged(self, storage: FileStorage):
        storage.submit_manifest(_manifest(format="external", path="/data/train.jsonl"))
        assert storage.claim_pending_manifest("foo").data.path == "/data/train.jsonl"


class TestComplete:
    def test_removes_running_file_only(self, storage: FileStorage):
        """A newer submission for the same key, pending while this one ran, stays queued."""
        storage.submit_manifest(_manifest())
        manifest = storage.claim_pending_manifest("foo")
        storage.submit_manifest(_manifest())
        manifest.output = _result()
        storage.complete_manifest(manifest)

        assert not (storage.running_path / "foo.yaml").exists()
        assert (storage.pending_path / "foo.yaml").exists()
        assert (storage.completed_path / "foo-abc123def456.yaml.gz").exists()

    def test_direct_run_of_pending_file_removes_it(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        manifest = load_manifest(storage.pending_path / "foo.yaml")
        manifest.output = _result()
        storage.complete_manifest(manifest)
        assert not (storage.pending_path / "foo.yaml").exists()


class TestFail:
    def test_moves_to_failed_with_error_in_output(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        storage.claim_pending_manifest("foo")
        failed = storage.fail_running_manifest("foo", _result("failed", error="RuntimeError: x"))

        assert not (storage.running_path / "foo.yaml").exists()
        assert failed.parent == storage.failed_path
        assert failed.name.endswith(".yaml.gz")
        loaded = load_manifest(failed)
        assert loaded.output.status == "failed"
        assert loaded.output.error == "RuntimeError: x"

    def test_unreadable_manifest_moved_unchanged(self, storage: FileStorage):
        storage.running_path.mkdir(parents=True)
        (storage.running_path / "foo.yaml").write_text("not: [valid")
        failed = storage.fail_running_manifest("foo", _result("failed", error="e"))
        assert failed.read_text() == "not: [valid"
        assert not (storage.running_path / "foo.yaml").exists()


class TestQueueStatus:
    def test_never_submitted(self, storage: FileStorage):
        assert storage.get_queue_status("foo") is None

    def test_pending_then_running(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        assert storage.get_queue_status("foo").state == "pending"
        storage.claim_pending_manifest("foo")
        assert storage.get_queue_status("foo").state == "running"

    def test_completed_carries_adapter(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        _complete(storage)
        status = storage.get_queue_status("foo")
        assert status.state == "completed"
        assert status.output.adapter.md5 == "abc123def456"
        assert status.created_at is not None

    def test_failed_carries_error(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        storage.claim_pending_manifest("foo")
        storage.fail_running_manifest("foo", _result("failed", error="ValueError: bad"))
        status = storage.get_queue_status("foo")
        assert status.state == "failed"
        assert status.output.error == "ValueError: bad"

    def test_pending_resubmission_wins_over_finished(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        _complete(storage)
        storage.submit_manifest(_manifest())
        assert storage.get_queue_status("foo").state == "pending"

    def test_newest_finished_wins(self, storage: FileStorage):
        storage.submit_manifest(_manifest())
        _complete(storage)
        storage.submit_manifest(_manifest())
        storage.claim_pending_manifest("foo")
        failed = storage.fail_running_manifest("foo", _result("failed", error="e"))
        completed = storage.completed_path / "foo-abc123def456.yaml.gz"
        os.utime(completed, (1000, 1000))
        os.utime(failed, (2000, 2000))
        assert storage.get_queue_status("foo").state == "failed"
        os.utime(completed, (3000, 3000))
        assert storage.get_queue_status("foo").state == "completed"

    def test_key_prefix_of_another_key_not_confused(self, storage: FileStorage):
        storage.submit_manifest(_manifest("foo-bar"))
        _complete(storage, "foo-bar")
        assert storage.get_queue_status("foo") is None
        assert storage.get_queue_status("foo-bar").state == "completed"

    def test_reads_metadata_only(self, storage: FileStorage, monkeypatch):
        """Status never loads the training data section."""
        storage.submit_manifest(_manifest())
        _complete(storage)
        monkeypatch.setattr(
            "llm_kelt.training.manifest.loader._read_yaml_file",
            lambda path: pytest.fail("full manifest load"),
        )
        assert storage.get_queue_status("foo").state == "completed"

    def test_client_status(self, lg, storage: FileStorage):
        client = Client(lg=lg, storage=storage)
        client.submit(_manifest())
        assert client.status("foo").state == "pending"


class TestRunPending:
    @pytest.fixture
    def runner(self, lg, storage: FileStorage, monkeypatch) -> Runner:
        runner = Runner(lg, storage)
        monkeypatch.setattr(runner, "_enrich_adapter", lambda result: result)
        return runner

    def _dispatch(self, monkeypatch, runner: Runner, failing: set[str]) -> None:
        def dispatch(manifest, skip_registration):
            if manifest.adapter in failing:
                raise RuntimeError(f"boom {manifest.adapter}")
            return _result(md5=f"{manifest.adapter}0000000000"[:12])

        monkeypatch.setattr(runner, "_dispatch_training", dispatch)

    def test_failure_does_not_stop_batch(self, runner: Runner, storage: FileStorage, monkeypatch):
        for key in ("a", "b"):
            storage.submit_manifest(_manifest(key))
        os.utime(storage.pending_path / "a.yaml", (1000, 1000))
        self._dispatch(monkeypatch, runner, failing={"a"})

        results = runner.run_pending()

        assert [(k, r.status) for k, r in results] == [("a", "failed"), ("b", "completed")]
        assert results[0][1].error == "RuntimeError: boom a"
        assert list(storage.running_path.iterdir()) == []
        assert list(storage.pending_path.glob("*.yaml")) == []
        assert storage.get_queue_status("a").state == "failed"
        assert storage.get_queue_status("b").state == "completed"

    def test_invalid_manifest_fails_with_reason(self, runner: Runner, storage: FileStorage):
        storage.submit_manifest(_manifest(format="inline", records=[], templates={"x": "y"}))
        [(key, result)] = runner.run_pending()
        assert result.status == "failed"
        assert "Invalid manifest" in result.error

    def test_unreadable_pending_manifest_fails(self, runner: Runner, storage: FileStorage):
        storage.pending_path.mkdir(parents=True)
        (storage.pending_path / "foo.yaml").write_text("not: [valid")
        [(key, result)] = runner.run_pending()
        assert key == "foo"
        assert result.status == "failed"
        assert list(storage.running_path.iterdir()) == []
        assert [p.read_text() for p in storage.failed_path.iterdir()] == ["not: [valid"]

    def test_already_running_key_skipped(self, runner: Runner, storage: FileStorage, monkeypatch):
        storage.submit_manifest(_manifest())
        storage.claim_pending_manifest("foo")
        storage.submit_manifest(_manifest())
        self._dispatch(monkeypatch, runner, failing=set())
        assert runner.run_pending() == []
        assert (storage.pending_path / "foo.yaml").exists()

    def test_nothing_pending(self, runner: Runner):
        assert runner.run_pending() == []

    def test_completed_manifest_is_gzipped_yaml(self, runner, storage: FileStorage, monkeypatch):
        storage.submit_manifest(_manifest())
        self._dispatch(monkeypatch, runner, failing=set())
        runner.run_pending()
        [completed] = storage.completed_path.iterdir()
        with gzip.open(completed, "rt", encoding="utf-8") as f:
            assert yaml.safe_load(f)["output"]["status"] == "completed"
