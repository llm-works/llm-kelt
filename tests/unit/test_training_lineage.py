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

from llm_kelt.training.lineage import MAX_ANNOTATIONS_BYTES, Lineage, Submission, series_root
from llm_kelt.training.manifest import Client
from llm_kelt.training.manifest.loader import load_manifest, save_manifest
from llm_kelt.training.manifest.schema import Data, Deployment, Manifest
from llm_kelt.training.runner import Runner
from llm_kelt.training.schema import Adapter, RunResult
from llm_kelt.training.storage import FileStorage

ROOT = series_root("org/model-a", {"method": "sft"})


def _rec(i: int) -> dict:
    return {"instruction": f"q{i}", "output": f"a{i}"}


def _submit(
    lineage: Lineage,
    series: str,
    *records: dict,
    adapter: str = "foo",
    templates: dict | None = None,
    annotations: dict | None = None,
    root: dict = ROOT,
) -> Submission:
    return lineage.submit(adapter, series, list(records), templates or {}, root, annotations or {})


@pytest.fixture
def lineage(lg, tmp_path: Path) -> Lineage:
    return Lineage(lg, MemoryStore(lg), tmp_path / "lineage")


class TestLineage:
    def test_resubmitted_set_returns_its_commit(self, lineage: Lineage):
        day1 = _submit(lineage, "main", _rec(1), _rec(2))
        _submit(lineage, "main", _rec(3))
        assert (day1.created, day1.root) == (True, True)
        again = _submit(lineage, "main", _rec(2), _rec(1))
        assert again == Submission(day1.commit, created=False, root=False)

    def test_only_the_first_submit_starts_the_series(self, lineage: Lineage):
        assert _submit(lineage, "main", _rec(1)).root
        assert not _submit(lineage, "main", _rec(2)).root
        assert _submit(lineage, "other", _rec(2)).root

    def test_build_input_is_union_oldest_first(self, lineage: Lineage):
        _submit(lineage, "main", _rec(1), templates={"t1": "x"})
        tip = _submit(lineage, "main", _rec(2), templates={"t2": "y"}).commit
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
        _submit(lineage, "main", _rec(1))
        with pytest.raises(ValueError, match=f"another {what}"):
            _submit(lineage, "main", _rec(2), root=root)

    def test_concurrent_submits_all_land(self, lineage: Lineage):
        threads = [
            threading.Thread(target=_submit, args=(lineage, "main", _rec(i))) for i in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        records, _, _ = lineage.build_input("foo", lineage.tip("foo", "main"))
        assert sorted(r["output"] for r in records) == [f"a{i}" for i in range(8)]

    def test_series_info_and_current_build(self, lineage: Lineage):
        day1 = _submit(lineage, "main", _rec(1)).commit
        day2 = _submit(lineage, "main", _rec(2)).commit
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
        day1 = _submit(lineage, "main", _rec(1)).commit
        day2 = _submit(lineage, "main", _rec(2)).commit
        lineage.record_build("foo", day1, "completed", md5="abc123def456")

        assert lineage.fork("foo", "main", "trial", day1) == day1
        assert lineage.series("foo", "trial").current_build.commit == day1
        trial2 = _submit(lineage, "trial", _rec(3)).commit
        assert lineage.tip("foo", "main") == day2
        records, _, _ = lineage.build_input("foo", trial2)
        assert records == [_rec(1), _rec(3)]

    def test_fork_rejects_unknown_commit_and_existing_series(self, lineage: Lineage):
        _submit(lineage, "main", _rec(1))
        other = _submit(lineage, "other", _rec(2)).commit
        with pytest.raises(ValueError, match="not in series"):
            lineage.fork("foo", "main", "trial", other)
        with pytest.raises(ValueError, match="already exists"):
            lineage.fork("foo", "main", "other")

    def test_find_set(self, lineage: Lineage):
        _submit(lineage, "main", _rec(1))
        _submit(lineage, "trial", _rec(1))
        _submit(lineage, "bar", _rec(2), adapter="bar")
        set_id = lineage.series("foo", "main").sets[0].set_id
        assert lineage.find_set(set_id) == [("foo", "main"), ("foo", "trial")]


class TestAnnotations:
    def test_first_submit_of_a_set_annotates_it(self, lineage: Lineage):
        _submit(lineage, "main", _rec(1), annotations={"source": "batch 1"})
        _submit(lineage, "trial", _rec(1), annotations={"source": "other"})
        [entry] = lineage.series("foo", "trial").sets
        assert entry.annotations == {"source": "batch 1"}

    def test_annotate_set_merges(self, lineage: Lineage):
        _submit(lineage, "main", _rec(1), annotations={"source": "batch 1"})
        set_id = lineage.series("foo", "main").sets[0].set_id
        lineage.annotate_set("foo", set_id, {"note": "use with caution"})
        assert lineage.set_annotations("foo", set_id) == {
            "source": "batch 1",
            "note": "use with caution",
        }
        with pytest.raises(ValueError, match="not found"):
            lineage.annotate_set("bar", set_id, {"note": "x"})

    def test_build_annotations_merge_survive_pruning_and_reset_on_rebuild(self, lineage: Lineage):
        commit = _submit(lineage, "main", _rec(1)).commit
        with pytest.raises(ValueError, match="No build"):
            lineage.annotate_build("foo", commit, {"gate": "passed"})
        lineage.record_build("foo", commit, "completed", md5="abc123def456")
        lineage.annotate_build("foo", commit, {"gate": "passed"})
        lineage.annotate_build("foo", commit, {"eval_loss": 0.4})
        lineage.mark_pruned("foo", commit)

        build = lineage.build("foo", commit)
        assert (build.state, build.md5) == ("pruned", "abc123def456")
        assert build.annotations == {"gate": "passed", "eval_loss": 0.4}
        lineage.record_build("foo", commit, "completed", md5="abc123def456")
        assert lineage.build("foo", commit).annotations == {}

    @pytest.mark.parametrize(
        "annotations",
        [{"at": datetime.now(UTC)}, {"x": float("nan")}, {"note": "x" * MAX_ANNOTATIONS_BYTES}],
    )
    def test_annotations_must_be_small_json(self, lineage: Lineage, annotations: dict):
        with pytest.raises(ValueError, match="JSON|limit"):
            _submit(lineage, "main", _rec(1), annotations=annotations)

    def test_merging_a_non_object_raises_value_error(self, lineage: Lineage):
        commit = _submit(lineage, "main", _rec(1), annotations={"n": 1}).commit
        lineage.record_build("foo", commit, "completed", md5="abc123def456")
        set_id = lineage.series("foo", "main").sets[0].set_id
        with pytest.raises(ValueError, match="JSON object"):
            lineage.annotate_build("foo", commit, [1])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="JSON object"):
            lineage.annotate_set("foo", set_id, [1])  # type: ignore[arg-type]


class TestDropAndWipe:
    def test_drop_removes_only_what_no_other_series_holds(self, lineage: Lineage):
        shared = _submit(lineage, "main", _rec(1), annotations={"n": 1}).commit
        lineage.record_build("foo", shared, "completed", md5="aaa111aaa111")
        lineage.fork("foo", "main", "trial")
        own = _submit(lineage, "trial", _rec(2), annotations={"n": 2}).commit
        lineage.record_build("foo", own, "completed", md5="bbb222bbb222")
        own_set = lineage.series("foo", "trial").sets[1].set_id

        assert lineage.exclusive_commits("foo", "trial") == [own]
        assert lineage.drop_series("foo", "trial") == [own]
        assert lineage.series("foo", "trial") is None
        assert lineage.build("foo", own) is None
        assert lineage.set_annotations("foo", own_set) == {}
        [kept] = lineage.series("foo", "main").sets
        assert (kept.build.md5, kept.annotations) == ("aaa111aaa111", {"n": 1})

    def test_wipe_removes_every_series(self, lineage: Lineage):
        commit = _submit(lineage, "main", _rec(1), annotations={"n": 1}).commit
        lineage.record_build("foo", commit, "completed", md5="aaa111aaa111")
        _submit(lineage, "trial", _rec(2))
        lineage.wipe("foo")
        assert lineage.series("foo", "main") is None
        assert lineage.series("foo", "trial") is None
        assert lineage.build("foo", commit) is None


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


def _build(storage: FileStorage, tmp_path: Path, series: str, md5: str) -> str:
    """Build the series' tip as a training client would: store the version, record the build."""
    commit = storage.lineage.tip("foo", series)
    source = tmp_path / f"weights-{md5}"
    source.mkdir()
    (source / "adapter_model.safetensors").write_text(md5)
    now = datetime.now(UTC)
    result = RunResult(
        started_at=now,
        completed_at=now,
        adapter=Adapter(md5=md5, mtime="", path=str(source)),
        series=series,
        commit=commit,
    )
    storage.store_adapter(result, "foo", f"{series} build", deploy=False)
    storage.lineage.record_build("foo", commit, "completed", md5=md5)
    return commit


class TestSeriesQueue:
    def test_submits_while_pending_queue_one_build_of_the_tip(self, storage: FileStorage):
        storage.submit_manifest(_manifest(_rec(1)))
        result = storage.submit_manifest(_manifest(_rec(2)))
        assert [p.name for p in storage.pending_path.iterdir()] == ["foo.yaml"]

        manifest = storage.claim_pending_manifest("foo")
        assert manifest.commit == result.commit
        assert manifest.data.records == [_rec(1), _rec(2)]
        assert load_manifest(storage.running_path / "foo.yaml").commit == result.commit

    def test_submit_result_says_whether_it_started_the_series(self, storage: FileStorage):
        assert storage.submit_manifest(_manifest(_rec(1))).root
        assert not storage.submit_manifest(_manifest(_rec(2))).root

    def test_manifest_annotations_annotate_the_set_not_the_queued_build(
        self, storage: FileStorage, tmp_path: Path
    ):
        manifest = _manifest(_rec(1))
        manifest.annotations = {"prompt": "v2"}
        path = tmp_path / "set.yaml"
        save_manifest(manifest, path)
        storage.submit_manifest(load_manifest(path))

        [entry] = storage.lineage.series("foo", "main").sets
        assert entry.annotations == {"prompt": "v2"}
        assert load_manifest(storage.pending_path / "foo.yaml").annotations == {}

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

    def test_only_main_series_deploys_on_completion(self, storage: FileStorage):
        with pytest.raises(ValueError, match="deployment policy 'skip'"):
            storage.submit_manifest(_manifest(_rec(1), series="trial"))
        storage.submit_manifest(_manifest(_rec(1), series="trial", policy="skip"))

    def test_latest_version_skips_other_series(self, storage: FileStorage, tmp_path: Path):
        """Deploying without a version never picks a build of a series the key does not serve."""
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
            commit="c" * 64,
        )
        info = storage.store_adapter(trial, "foo", "trial build", deploy=False)
        config = storage.read_adapter_config("foo", info.version_id)
        assert (config["series"], config["commit"]) == ("trial", "c" * 64)

        storage.deploy_adapter("foo")
        assert storage.get_deployed_version("foo") == "20000101-000000-aaa111"
        assert storage.get_adapter("foo").md5 == "aaa111"

    def test_model_override_fails_series_build(self, lg, storage: FileStorage):
        storage.submit_manifest(_manifest(_rec(1)))
        [(_, result)] = Runner(lg, storage).run_pending(model_override="org/model-b")
        assert result.status == "failed"
        assert "overrides are not allowed" in result.error

    def test_concurrent_series_submits_queue_one_build(self, storage: FileStorage):
        """Concurrent submits all land in the series and queue at most one build."""
        results: list[tuple[str, bool]] = []

        def submit(i: int) -> None:
            r = storage.submit_manifest(_manifest(_rec(i)))
            results.append((r.commit, r.created))

        threads = [threading.Thread(target=submit, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All sets land in the series
        records, _, _ = storage.lineage.build_input("foo", storage.lineage.tip("foo", "main"))
        assert sorted(r["output"] for r in records) == [f"a{i}" for i in range(8)]
        # Only one pending file (no duplicates)
        assert [p.name for p in storage.pending_path.iterdir()] == ["foo.yaml"]


class TestSeriesDeployment:
    def test_deploying_a_new_series_switches_the_key_to_it(
        self, storage: FileStorage, tmp_path: Path
    ):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        v1 = _build(storage, tmp_path, "v1", "aaa111aaa111")
        storage.deploy_series("foo", "v1")
        assert storage.active_series("foo").series == "v1"

        storage.submit_manifest(_manifest(_rec(2), series="v2", policy="skip"))
        v2 = _build(storage, tmp_path, "v2", "bbb222bbb222")
        version_id = storage.deploy_series("foo", "v2", v2)

        active = storage.active_series("foo")
        assert (active.series, active.commit, active.version_id) == ("v2", v2, version_id)
        assert storage.list_deployed("foo") == [("foo", "bbb222bbb222")]
        assert storage.get_adapter("foo").md5 == "bbb222bbb222"
        assert v1 != v2

    def test_a_fork_deployed_at_a_shared_build_is_active(
        self, storage: FileStorage, tmp_path: Path
    ):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        shared = _build(storage, tmp_path, "v1", "aaa111aaa111")
        storage.lineage.fork("foo", "v1", "trial")
        storage.deploy_series("foo", "trial", shared)
        assert storage.active_series("foo").series == "trial"
        # A deploy that bypasses deploy_series leaves the version's own series in charge
        storage.deploy_adapter("foo", storage.active_series("foo").version_id)
        storage.deploy_series("foo", "v1")
        assert storage.active_series("foo").series == "v1"

    def test_concurrent_deploys_record_the_deployed_commit(
        self, storage: FileStorage, tmp_path: Path
    ):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        _build(storage, tmp_path, "v1", "aaa111aaa111")
        storage.submit_manifest(_manifest(_rec(2), series="v2", policy="skip"))
        _build(storage, tmp_path, "v2", "bbb222bbb222")

        threads = [
            threading.Thread(target=storage.deploy_series, args=("foo", f"v{1 + i % 2}"))
            for i in range(16)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        active = storage.active_series("foo")
        assert storage.lineage.deployed("foo") == (active.series, active.commit)
        assert len(storage.list_deployed("foo")) == 1

    def test_redeploy_keeps_the_link(self, storage: FileStorage, tmp_path: Path):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        _build(storage, tmp_path, "v1", "aaa111aaa111")
        storage.deploy_series("foo", "v1")
        storage.deploy_series("foo", "v1")
        assert [p.name for p in storage.deployed_path.iterdir()] == ["foo-aaa111aaa111"]

    def test_unbuilt_or_pruned_commit_is_not_deployed(self, storage: FileStorage, tmp_path):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        with pytest.raises(ValueError, match="no completed build"):
            storage.deploy_series("foo", "v1")
        commit = _build(storage, tmp_path, "v1", "aaa111aaa111")
        storage.prune_series_adapters("foo", "v1")
        with pytest.raises(ValueError, match="is pruned"):
            storage.deploy_series("foo", "v1", commit)
        with pytest.raises(ValueError, match="not in series"):
            storage.deploy_series("foo", "v1", "c" * 64)


class TestSeriesRetention:
    def test_prune_keeps_sets_and_the_deployed_build(self, storage: FileStorage, tmp_path):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        old = _build(storage, tmp_path, "v1", "aaa111aaa111")
        storage.submit_manifest(_manifest(_rec(2), series="v1", policy="skip"))
        new = _build(storage, tmp_path, "v1", "bbb222bbb222")
        storage.deploy_series("foo", "v1", new)

        assert storage.prune_series_adapters("foo", "v1") == [old]
        assert storage.lineage.build("foo", old).state == "pruned"
        assert storage.lineage.build("foo", new).state == "completed"
        assert [v.split("-")[-1] for v in storage.list_versions("foo")] == ["bbb222bbb222"]
        assert len(storage.lineage.series("foo", "v1").sets) == 2

    def test_drop_refuses_the_active_series(self, storage: FileStorage, tmp_path: Path):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        _build(storage, tmp_path, "v1", "aaa111aaa111")
        storage.deploy_series("foo", "v1")
        with pytest.raises(ValueError, match="is deployed"):
            storage.drop_series("foo", "v1")

    def test_drop_removes_the_series_weights_and_queued_build(
        self, storage: FileStorage, tmp_path: Path
    ):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        _build(storage, tmp_path, "v1", "aaa111aaa111")
        storage.deploy_series("foo", "v1")
        storage.submit_manifest(_manifest(_rec(2), series="v2", policy="skip"))
        built = _build(storage, tmp_path, "v2", "bbb222bbb222")
        storage.submit_manifest(_manifest(_rec(3), series="v2", policy="skip"))
        assert (storage.pending_path / "foo+v2.yaml").exists()

        assert len(storage.drop_series("foo", "v2")) == 2
        assert storage.lineage.series("foo", "v2") is None
        assert storage.lineage.build("foo", built) is None
        assert not (storage.pending_path / "foo+v2.yaml").exists()
        assert [v.split("-")[-1] for v in storage.list_versions("foo")] == ["aaa111aaa111"]

    def test_wipe_removes_everything_of_the_key(self, storage: FileStorage, tmp_path: Path):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        _build(storage, tmp_path, "v1", "aaa111aaa111")
        storage.deploy_series("foo", "v1")
        storage.submit_manifest(_manifest(_rec(2), series="v2", policy="skip"))

        storage.wipe_adapter("foo")
        assert storage.lineage.series("foo", "v1") is None
        assert storage.list_deployed("foo") == []
        assert not storage.adapter_exists("foo")
        assert storage.list_pending_keys() == []

    def test_wipe_refuses_while_a_build_runs(self, storage: FileStorage):
        storage.submit_manifest(_manifest(_rec(1), series="v1", policy="skip"))
        storage.claim_pending_manifest("foo+v1")
        with pytest.raises(ValueError, match="running"):
            storage.wipe_adapter("foo")
