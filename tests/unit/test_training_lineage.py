# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for training lineage: series of sets, and series builds through the queue."""

from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest
from appinfra import DotDict
from appinfra.cas import MemoryStore

from llm_kelt.training.lineage import Lineage, series_root
from llm_kelt.training.manifest import Client
from llm_kelt.training.manifest.loader import load_manifest
from llm_kelt.training.manifest.schema import Data, Deployment, Manifest
from llm_kelt.training.runner import Runner
from llm_kelt.training.schema import Adapter, RunResult
from llm_kelt.training.storage import FileStorage

ROOT = series_root("org/model-a", {"method": "sft"})


def _rec(i: int) -> dict:
    return {"instruction": f"q{i}", "output": f"a{i}"}


@pytest.fixture
def lineage(lg, tmp_path: Path) -> Lineage:
    return Lineage(lg, MemoryStore(lg), tmp_path / "lineage")


class TestLineage:
    def test_resubmitted_set_returns_its_commit(self, lineage: Lineage):
        day1, created = lineage.submit("foo", "main", [_rec(1), _rec(2)], {}, ROOT)
        lineage.submit("foo", "main", [_rec(3)], {}, ROOT)
        assert created
        assert lineage.submit("foo", "main", [_rec(2), _rec(1)], {}, ROOT) == (day1, False)

    def test_build_input_is_union_oldest_first(self, lineage: Lineage):
        lineage.submit("foo", "main", [_rec(1)], {"t1": "x"}, ROOT)
        tip, _ = lineage.submit("foo", "main", [_rec(2)], {"t2": "y"}, ROOT)
        records, templates, root = lineage.build_input("foo", tip)
        assert records == [_rec(1), _rec(2)]
        assert templates == {"t1": "x", "t2": "y"}
        assert root == ROOT

    @pytest.mark.parametrize(
        ("root", "what"),
        [
            (series_root("org/model-b", {"method": "sft"}), "base model"),
            (series_root("org/model-a", {"method": "dpo"}), "recipe"),
        ],
    )
    def test_other_base_or_recipe_rejected(self, lineage: Lineage, root: dict, what: str):
        lineage.submit("foo", "main", [_rec(1)], {}, ROOT)
        with pytest.raises(ValueError, match=f"another {what}"):
            lineage.submit("foo", "main", [_rec(2)], {}, root)

    def test_concurrent_submits_all_land(self, lineage: Lineage):
        threads = [
            threading.Thread(target=lineage.submit, args=("foo", "main", [_rec(i)], {}, ROOT))
            for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        records, _, _ = lineage.build_input("foo", lineage.tip("foo", "main"))
        assert sorted(r["output"] for r in records) == [f"a{i}" for i in range(8)]

    def test_series_info_and_current_build(self, lineage: Lineage):
        day1, _ = lineage.submit("foo", "main", [_rec(1)], {}, ROOT)
        day2, _ = lineage.submit("foo", "main", [_rec(2)], {}, ROOT)
        lineage.record_build("foo", day1, "completed", md5="abc123def456")
        lineage.record_build("foo", day2, "failed", error="RuntimeError: x")

        info = lineage.series("foo", "main")
        assert info.tip == day2
        assert info.base_model == "org/model-a"
        assert [e.commit for e in info.sets] == [day1, day2]
        assert info.sets[0].submitted_at is not None
        assert info.current_build.commit == day1
        assert lineage.series("foo", "other") is None

    def test_fork_shares_history_then_diverges(self, lineage: Lineage):
        day1, _ = lineage.submit("foo", "main", [_rec(1)], {}, ROOT)
        day2, _ = lineage.submit("foo", "main", [_rec(2)], {}, ROOT)
        lineage.record_build("foo", day1, "completed", md5="abc123def456")

        assert lineage.fork("foo", "main", "trial", day1) == day1
        assert lineage.series("foo", "trial").current_build.commit == day1
        trial2, _ = lineage.submit("foo", "trial", [_rec(3)], {}, ROOT)
        assert lineage.tip("foo", "main") == day2
        records, _, _ = lineage.build_input("foo", trial2)
        assert records == [_rec(1), _rec(3)]

    def test_fork_rejects_unknown_commit_and_existing_series(self, lineage: Lineage):
        lineage.submit("foo", "main", [_rec(1)], {}, ROOT)
        other, _ = lineage.submit("foo", "other", [_rec(2)], {}, ROOT)
        with pytest.raises(ValueError, match="not in series"):
            lineage.fork("foo", "main", "trial", other)
        with pytest.raises(ValueError, match="already exists"):
            lineage.fork("foo", "main", "other")

    def test_find_set(self, lineage: Lineage):
        lineage.submit("foo", "main", [_rec(1)], {}, ROOT)
        lineage.submit("foo", "trial", [_rec(1)], {}, ROOT)
        lineage.submit("bar", "main", [_rec(2)], {}, ROOT)
        set_id = lineage.series("foo", "main").sets[0].set_id
        assert lineage.find_set(set_id) == [("foo", "main"), ("foo", "trial")]


@pytest.fixture
def storage(lg, tmp_path: Path) -> FileStorage:
    return FileStorage(lg, tmp_path / "registry")


def _manifest(*records: dict, series: str = "main", policy: str = "replace") -> Manifest:
    return Manifest(
        adapter="foo",
        method="sft",
        series=series,
        data=Data(format="inline", records=list(records)),
        deployment=Deployment(policy=policy),
        training=DotDict(requested_model="org/model-a", num_epochs=1),
    )


def _complete(storage: FileStorage, md5: str = "abc123def456") -> Manifest:
    manifest = storage.claim_pending_manifest("foo")
    now = datetime.now(UTC)
    adapter = Adapter(md5=md5, mtime="", path="/adapters/x")
    manifest.output = RunResult(
        status="completed", started_at=now, completed_at=now, adapter=adapter
    )
    storage.complete_manifest(manifest)
    return manifest


class TestSeriesQueue:
    def test_submits_while_pending_queue_one_build_of_the_tip(self, storage: FileStorage):
        storage.submit_manifest(_manifest(_rec(1)))
        result = storage.submit_manifest(_manifest(_rec(2)))
        assert [p.name for p in storage.pending_path.iterdir()] == ["foo.yaml"]

        manifest = storage.claim_pending_manifest("foo")
        assert manifest.commit == result.commit
        assert manifest.data.records == [_rec(1), _rec(2)]
        assert load_manifest(storage.running_path / "foo.yaml").commit == result.commit

    def test_completed_build_records_commit_and_keeps_sets_out_of_manifest(self, storage):
        result = storage.submit_manifest(_manifest(_rec(1)))
        _complete(storage)

        build = storage.lineage.build("foo", result.commit)
        assert (build.state, build.md5) == ("completed", "abc123def456")
        [completed] = storage.completed_path.iterdir()
        assert load_manifest(completed).data.records == []
        assert load_manifest(completed).commit == result.commit

    def test_resubmit_of_built_set_queues_nothing(self, storage: FileStorage):
        first = storage.submit_manifest(_manifest(_rec(1)))
        _complete(storage)
        again = storage.submit_manifest(_manifest(_rec(1)))
        assert (again.commit, again.created) == (first.commit, False)
        assert list(storage.pending_path.glob("*.yaml")) == []

    def test_failed_build_is_recorded_and_resubmit_requeues_it(self, storage: FileStorage):
        result = storage.submit_manifest(_manifest(_rec(1)))
        storage.claim_pending_manifest("foo")
        now = datetime.now(UTC)
        failed = RunResult(status="failed", started_at=now, completed_at=now, error="boom")
        storage.fail_running_manifest("foo", failed)

        assert storage.lineage.build("foo", result.commit).state == "failed"
        storage.submit_manifest(_manifest(_rec(1)))
        assert (storage.pending_path / "foo.yaml").exists()

    def test_pending_one_off_manifest_blocks_series_submit(self, storage: FileStorage):
        one_off = _manifest(_rec(1))
        one_off.series = None
        storage.submit_manifest(one_off)
        with pytest.raises(ValueError, match="already pending"):
            storage.submit_manifest(_manifest(_rec(2)))

    def test_series_of_one_adapter_queue_separately(self, lg, storage: FileStorage):
        storage.submit_manifest(_manifest(_rec(1)))
        storage.submit_manifest(_manifest(_rec(2), series="trial", policy="skip"))
        assert sorted(storage.list_pending_keys()) == ["foo", "foo+trial"]

        trial = storage.claim_pending_manifest("foo+trial")
        assert (trial.series, trial.data.records) == ("trial", [_rec(2)])
        assert storage.claim_pending_manifest("foo").data.records == [_rec(1)]
        assert Client(lg, storage).status("foo", series="trial").state == "running"

    def test_only_main_series_is_deployed(self, storage: FileStorage):
        with pytest.raises(ValueError, match="deployment policy 'skip'"):
            storage.submit_manifest(_manifest(_rec(1), series="trial"))
        storage.submit_manifest(_manifest(_rec(1), series="trial", policy="skip"))

    def test_latest_version_skips_other_series(self, storage: FileStorage, tmp_path: Path):
        """Deploying without a version never picks a build of a series other than main."""
        source = tmp_path / "adapter"
        source.mkdir()
        (source / "adapter_model.safetensors").write_text("weights")
        storage.store_adapter_files(source, "foo", "20000101-000000-aaa111")
        storage.write_adapter_config("foo", "20000101-000000-aaa111", {"md5": "aaa111"})
        now = datetime.now(UTC)
        trial = RunResult(
            started_at=now,
            completed_at=now,
            adapter=Adapter(md5="bbb222", mtime="", path=str(source)),
            series="trial",
        )
        storage.store_adapter(trial, "foo", "trial build", deploy=False)

        storage.deploy_adapter("foo")
        assert storage.get_deployed_version("foo") == "20000101-000000-aaa111"
        assert storage.get_adapter("foo").md5 == "aaa111"

    def test_model_override_fails_series_build(self, lg, storage: FileStorage):
        storage.submit_manifest(_manifest(_rec(1)))
        [(_, result)] = Runner(lg, storage).run_pending(model_override="org/model-b")
        assert result.status == "failed"
        assert "overrides are not allowed" in result.error
