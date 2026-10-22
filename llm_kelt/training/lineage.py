# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Training lineage: series of training sets, each build retrained from the base model.

Every adapter key has its own repository in an ``appinfra.cas`` store, and a series is a
branch of it (ref ``series/<name>``). Each submit adds the submitted training set to the
series as a commit, unless the set is already in the series' history. A build of a commit
trains from the base model on every set from the root commit to that commit, so no
weights carry over between builds.

- Set: a tree of a ``records`` blob and a ``templates`` blob, both canonical JSON. Records
  are sorted by their canonical bytes, so their order does not change the set id.
- Commit: the set's tree and the parent commit, with no timestamp. The root commit's
  payload holds the series' base model and recipe; later submits must match it.
- Submit dates and builds are kept outside the store, per (adapter, commit):
  ``<records>/<adapter>/submitted/<commit>`` and ``<records>/<adapter>/builds/<commit>.yaml``.
  Series of one adapter that share a commit share its build.
- Annotations are small mutable JSON objects outside the store, so they never change an
  id: a set's (``<records>/<adapter>/sets/<set>.yaml``) start as those of its first submit;
  a build's live in its build record, survive pruning its weights and are reset by a
  rebuild.
- Every write holds the registry's write lock (see locks.py); reads do not lock.
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from appinfra.cas import (
    Blob,
    Commit,
    RefConflictError,
    Repo,
    Store,
    Tree,
    TreeEntry,
    canonical_json,
    is_object_id,
)
from appinfra.log import Logger

from .locks import RegistryLock

DEFAULT_SERIES = "main"

_REF_PREFIX = "series/"

# Objects a dropped series leaves unreachable are deleted once they are this old (seconds),
# so a concurrent submit's objects, stored but not yet referenced, are never collected.
_PRUNE_GRACE = 3600.0

# Annotations are notes and small metrics, read with every series lookup; the cap is on
# their JSON encoding, per set or build.
MAX_ANNOTATIONS_BYTES = 64 * 1024

BuildState = Literal["completed", "failed", "pruned"]


@dataclass(frozen=True)
class Build:
    """Outcome of training one commit."""

    # "pruned": completed, then its adapter weights were deleted; md5 names them
    state: BuildState
    at: datetime
    # Adapter weights md5, for a completed or pruned build
    md5: str | None = None
    # Error message, for a failed build
    error: str | None = None
    # Mutable metadata, e.g. a training gate's verdict and metrics
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Submission:
    """Outcome of adding a set to a series."""

    # Commit holding the set
    commit: str
    # Whether this submit added the set; False when it was already in the series' history
    created: bool
    # Whether this submit started the series (its commit is the root)
    root: bool


@dataclass(frozen=True)
class SetEntry:
    """One set of a series, at the commit that added it."""

    commit: str
    set_id: str
    submitted_at: datetime | None
    build: Build | None
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SeriesInfo:
    """State of a series: its tip, base, recipe and sets, oldest first."""

    adapter: str
    series: str
    tip: str
    base_model: str
    recipe: dict[str, Any]
    sets: list[SetEntry]

    @property
    def current_build(self) -> SetEntry | None:
        """The newest set whose commit has a completed build, or None."""
        for entry in reversed(self.sets):
            if entry.build is not None and entry.build.state == "completed":
                return entry
        return None


def series_root(base_model: str, recipe: dict[str, Any]) -> dict[str, Any]:
    """Return the root commit payload of a series: its base model and recipe."""
    return {"base": {"model": base_model}, "recipe": recipe}


class Lineage:
    """Series of training sets per adapter key, kept in a content-addressed store."""

    def __init__(self, lg: Logger, store: Store, records_path: Path, lock: RegistryLock) -> None:
        """Initialize lineage.

        Args:
            lg: Logger instance.
            store: Content-addressed store; one repository per adapter key.
            records_path: Directory for submit dates, build records and annotations.
            lock: The registry's write lock, shared with every other writer of the
                registry.
        """
        self._lg = lg
        self._store = store
        self._records_path = records_path
        self._lock = lock

    def submit(
        self,
        adapter: str,
        series: str,
        records: list[dict[str, Any]],
        templates: dict[str, str],
        root: dict[str, Any],
        annotations: dict[str, Any],
    ) -> Submission:
        """Add a training set to a series.

        Args:
            adapter: Adapter key.
            series: Series name.
            records: Training records of the set.
            templates: Shared prompt templates the records reference.
            root: Root payload (see ``series_root``); the series' first submit stores it,
                later submits must match it.
            annotations: The set's annotations (a JSON object); stored unless the set
                already has some.

        Returns:
            The commit that holds the set. A set already in the series' history returns
            its existing commit, with ``created`` False.

        Raises:
            ValueError: The series was started with another base model or recipe, or the
                annotations are not a JSON object within MAX_ANNOTATIONS_BYTES.
        """
        annotations = _json_object(annotations)
        with self._lock.hold():
            return self._submit(adapter, series, records, templates, root, annotations)

    def _submit(
        self,
        adapter: str,
        series: str,
        records: list[dict[str, Any]],
        templates: dict[str, str],
        root: dict[str, Any],
        annotations: dict[str, Any],
    ) -> Submission:
        """Body of submit(); the caller holds the registry's lock."""
        repo = self._store.repo(adapter)
        set_id = _put_set(repo, records, templates)
        if annotations:
            self._write_once(self._set_path(adapter, set_id), yaml.safe_dump(annotations))
        payload = canonical_json(root)
        ref = _REF_PREFIX + series
        while True:
            tip = repo.get_ref(ref)
            if tip is not None:
                history = list(repo.first_parents(tip))
                _check_root(adapter, series, history[-1].payload, payload)
                existing = next((c.id for c in history if c.tree == set_id), None)
                if existing is not None:
                    return Submission(existing, created=False, root=False)
            try:
                commit = repo.commit(ref, set_id, payload if tip is None else b"")
            except RefConflictError:
                continue  # another submit moved the series; check its history again
            self._write_once(self._submitted_path(adapter, commit), datetime.now(UTC).isoformat())
            self._lg.info(
                "added set to series",
                extra={"adapter": adapter, "series": series, "set": set_id, "commit": commit},
            )
            # The ref was created by this compare-and-set, so no other submit started it
            return Submission(commit, created=True, root=tip is None)

    def fork(self, adapter: str, series: str, new_series: str, commit: str | None = None) -> str:
        """Start ``new_series`` at a commit of ``series`` (default: its tip); return it.

        Raises:
            ValueError: ``series`` does not exist, ``commit`` is not in its history, or
                ``new_series`` exists.
        """
        repo = self._store.repo(adapter)
        with self._lock.hold():
            tip = self.tip(adapter, series)
            if tip is None:
                raise ValueError(f"Series not found: {adapter}/{series}")
            if commit is None:
                commit = tip
            elif all(c.id != commit for c in repo.first_parents(tip)):
                raise ValueError(f"Commit {commit} is not in series {adapter}/{series}")
            if not repo.set_ref(_REF_PREFIX + new_series, commit, None):
                raise ValueError(f"Series already exists: {adapter}/{new_series}")
        self._lg.info(
            "forked series",
            extra={"adapter": adapter, "series": series, "new": new_series, "commit": commit},
        )
        return commit

    def tip(self, adapter: str, series: str) -> str | None:
        """Return the newest commit of a series, or None if the series does not exist."""
        return self._store.repo(adapter).get_ref(_REF_PREFIX + series)

    def build_input(
        self, adapter: str, commit: str
    ) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, Any]]:
        """Return what a build of ``commit`` trains on: records, templates and root payload.

        Records are those of every set from the root commit to ``commit``, oldest set
        first; templates are the union of the sets' templates.
        """
        repo = self._store.repo(adapter)
        history = list(repo.first_parents(commit))
        records: list[dict[str, Any]] = []
        templates: dict[str, str] = {}
        for c in reversed(history):
            set_records, set_templates = _read_set(repo, c.tree)
            records.extend(set_records)
            templates.update(set_templates)
        return records, templates, json.loads(history[-1].payload)

    def record_build(
        self,
        adapter: str,
        commit: str,
        state: Literal["completed", "failed"],
        *,
        md5: str | None = None,
        error: str | None = None,
    ) -> None:
        """Record the outcome of a build of ``commit``, replacing an earlier one.

        The earlier build's annotations go with it: they describe weights this build
        replaces. A failed build does not replace a completed or pruned one: those weights
        exist (or existed) and are still the commit's build.
        """
        build = {"state": state, "at": datetime.now(UTC).isoformat(), "md5": md5, "error": error}
        with self._lock.hold():
            earlier = self._read_build(adapter, commit)
            if state == "failed" and earlier and earlier["state"] in ("completed", "pruned"):
                self._lg.warning(
                    "failed rebuild kept the earlier build",
                    extra={"adapter": adapter, "commit": commit, "error": error},
                )
                return
            _write_atomic(self._build_path(adapter, commit), yaml.safe_dump(build))

    def build(self, adapter: str, commit: str) -> Build | None:
        """Return the recorded build of ``commit``, or None if it was never built."""
        data = self._read_build(adapter, commit)
        if data is None:
            return None
        return Build(
            state=data["state"],
            at=datetime.fromisoformat(data["at"]),
            md5=data.get("md5"),
            error=data.get("error"),
            annotations=data.get("annotations") or {},
        )

    def annotate_build(self, adapter: str, commit: str, annotations: dict[str, Any]) -> None:
        """Merge ``annotations`` into the annotations of ``commit``'s build, key by key.

        Raises:
            ValueError: ``commit`` was never built, or the merged annotations are not a
                JSON object within MAX_ANNOTATIONS_BYTES.
        """
        new = _json_object(annotations)  # before merging: a non-dict would raise TypeError
        with self._lock.hold():
            data = self._read_build(adapter, commit)
            if data is None:
                raise ValueError(f"No build of commit {commit} for adapter {adapter}")
            data["annotations"] = _json_object({**(data.get("annotations") or {}), **new})
            _write_atomic(self._build_path(adapter, commit), yaml.safe_dump(data))

    def annotate_set(self, adapter: str, set_id: str, annotations: dict[str, Any]) -> None:
        """Merge ``annotations`` into a set's annotations, key by key.

        Raises:
            ValueError: No series of the adapter holds the set, or the merged annotations
                are not a JSON object within MAX_ANNOTATIONS_BYTES.
        """
        new = _json_object(annotations)  # before merging: a non-dict would raise TypeError
        repo = self._store.repo(adapter)
        with self._lock.hold():
            refs = repo.refs(_REF_PREFIX).values()
            if set_id not in (c.tree for tip in refs for c in repo.first_parents(tip)):
                raise ValueError(f"Training set not found: {adapter}/{set_id}")
            merged = _json_object({**self.set_annotations(adapter, set_id), **new})
            _write_atomic(self._set_path(adapter, set_id), yaml.safe_dump(merged))

    def mark_pruned(self, adapter: str, commit: str) -> None:
        """Mark a completed build pruned: its weights are gone, its record stays."""
        with self._lock.hold():
            data = self._read_build(adapter, commit)
            if data is None or data["state"] != "completed":
                raise ValueError(f"No completed build of commit {commit} for adapter {adapter}")
            data["state"] = "pruned"
            _write_atomic(self._build_path(adapter, commit), yaml.safe_dump(data))

    def set_annotations(self, adapter: str, set_id: str) -> dict[str, Any]:
        """Return a set's annotations, empty if it has none."""
        try:
            text = self._set_path(adapter, set_id).read_text(encoding="utf-8")
        except FileNotFoundError:
            return {}
        return yaml.safe_load(text) or {}

    def series(self, adapter: str, series: str) -> SeriesInfo | None:
        """Return the state of a series, or None if it does not exist."""
        tip = self.tip(adapter, series)
        if tip is None:
            return None
        history = list(self._store.repo(adapter).first_parents(tip))
        root = json.loads(history[-1].payload)
        sets = [
            SetEntry(
                commit=c.id,
                set_id=c.tree,
                submitted_at=self._read_submitted(adapter, c.id),
                build=self.build(adapter, c.id),
                annotations=self.set_annotations(adapter, c.tree),
            )
            for c in reversed(history)
        ]
        return SeriesInfo(
            adapter=adapter,
            series=series,
            tip=tip,
            base_model=root["base"]["model"],
            recipe=root["recipe"],
            sets=sets,
        )

    def find_set(self, set_id: str) -> list[tuple[str, str]]:
        """Return (adapter, series) of every series whose history contains the set."""
        if not is_object_id(set_id):
            raise ValueError(f"Invalid set id: {set_id}")
        found: list[tuple[str, str]] = []
        for adapter in self._store.list_repos():
            repo = self._store.repo(adapter)
            for ref, tip in repo.refs(_REF_PREFIX).items():
                if any(c.tree == set_id for c in repo.first_parents(tip)):
                    found.append((adapter, ref.removeprefix(_REF_PREFIX)))
        return found

    def exclusive_commits(self, adapter: str, series: str) -> list[str]:
        """Commits of a series that no other series of the adapter reaches, oldest first."""
        repo = self._store.repo(adapter)
        tip = self.tip(adapter, series)
        if tip is None:
            raise ValueError(f"Series not found: {adapter}/{series}")
        shared = self._reachable(repo, exclude=_REF_PREFIX + series)
        return [c.id for c in reversed(list(repo.first_parents(tip))) if c.id not in shared]

    def drop_series(self, adapter: str, series: str) -> list[str]:
        """Delete a series and what only it reaches: commits, sets and their records.

        Sets and builds shared with another series stay. Returns the dropped commits.

        Raises:
            ValueError: The series does not exist.
        """
        repo = self._store.repo(adapter)
        ref = _REF_PREFIX + series
        with self._lock.hold():
            tip = repo.get_ref(ref)
            if tip is None:
                raise ValueError(f"Series not found: {adapter}/{series}")
            history = list(repo.first_parents(tip))
            repo.delete_ref(ref, tip)  # cannot fail: every ref writer holds the lock
            shared = self._reachable(repo, exclude=None)
            shared_sets = {self._commit_tree(repo, c) for c in shared}
            dropped = [c for c in history if c.id not in shared]
            for c in dropped:
                self._build_path(adapter, c.id).unlink(missing_ok=True)
                self._submitted_path(adapter, c.id).unlink(missing_ok=True)
                if c.tree not in shared_sets:
                    self._set_path(adapter, c.tree).unlink(missing_ok=True)
            repo.prune(_PRUNE_GRACE)
        self._lg.info("dropped series", extra={"adapter": adapter, "series": series})
        return [c.id for c in reversed(dropped)]

    def record_deployed(self, adapter: str, series: str, commit: str) -> None:
        """Record which series a deployment of ``commit`` serves.

        Needed because series that share a commit share its build: the deployed version
        alone names the series it was built for, not the one it was deployed for.
        """
        record = yaml.safe_dump({"series": series, "commit": commit})
        with self._lock.hold():
            _write_atomic(self._records_path / adapter / "deployed.yaml", record)

    def deployed(self, adapter: str) -> tuple[str, str] | None:
        """Return the (series, commit) last recorded by ``record_deployed``, or None."""
        try:
            text = (self._records_path / adapter / "deployed.yaml").read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        data = yaml.safe_load(text)
        if not data or "series" not in data or "commit" not in data:
            return None
        return data["series"], data["commit"]

    def wipe(self, adapter: str) -> None:
        """Delete every series of an adapter with all its sets, builds and annotations."""
        with self._lock.hold():
            self._store.delete_repo(adapter)
            shutil.rmtree(self._records_path / adapter, ignore_errors=True)
        self._lg.info("wiped lineage", extra={"adapter": adapter})

    def _reachable(self, repo: Repo, exclude: str | None) -> set[str]:
        """Ids of the commits any series ref reaches, except the ref ``exclude``."""
        reachable: set[str] = set()
        for ref, tip in repo.refs(_REF_PREFIX).items():
            if ref != exclude:
                reachable.update(c.id for c in repo.first_parents(tip))
        return reachable

    def _commit_tree(self, repo: Repo, commit: str) -> str:
        found = repo.get(commit, Commit)
        if found is None:
            raise ValueError(f"Commit not found: {commit}")
        return found.tree

    def _read_build(self, adapter: str, commit: str) -> dict[str, Any] | None:
        try:
            text = self._build_path(adapter, commit).read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        data: dict[str, Any] = yaml.safe_load(text)
        return data

    def _build_path(self, adapter: str, commit: str) -> Path:
        return self._records_path / adapter / "builds" / f"{commit}.yaml"

    def _submitted_path(self, adapter: str, commit: str) -> Path:
        return self._records_path / adapter / "submitted" / commit

    def _set_path(self, adapter: str, set_id: str) -> Path:
        return self._records_path / adapter / "sets" / f"{set_id}.yaml"

    def _write_once(self, path: Path, text: str) -> None:
        """Write a record unless it exists; the first writer wins."""
        if not path.exists():
            _write_atomic(path, text)

    def _read_submitted(self, adapter: str, commit: str) -> datetime | None:
        try:
            return datetime.fromisoformat(self._submitted_path(adapter, commit).read_text())
        except FileNotFoundError:
            return None


def _put_set(repo: Repo, records: list[dict[str, Any]], templates: dict[str, str]) -> str:
    """Store a training set and return its id (the id of its tree)."""
    ordered = sorted(canonical_json(r) for r in records)
    records_blob = repo.put(Blob(b"[" + b",".join(ordered) + b"]"))
    templates_blob = repo.put(Blob(canonical_json(templates)))
    return repo.put(
        Tree(
            (
                TreeEntry("records", "blob", records_blob),
                TreeEntry("templates", "blob", templates_blob),
            )
        )
    )


def _read_set(repo: Repo, set_id: str) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Return the records and templates of a stored training set."""
    tree = repo.get(set_id, Tree)
    if tree is None:
        raise ValueError(f"Training set not found: {set_id}")
    parts = {}
    for name in ("records", "templates"):
        entry = tree.get(name)
        blob = repo.get(entry.id, Blob) if entry is not None else None
        if blob is None:
            raise ValueError(f"Training set {set_id} has no {name}")
        parts[name] = json.loads(blob.data)
    return parts["records"], parts["templates"]


def _json_object(annotations: dict[str, Any]) -> dict[str, Any]:
    """Return annotations as plain JSON values, if they are a JSON object within the cap.

    Raises:
        ValueError: Not a JSON object, or its encoding exceeds MAX_ANNOTATIONS_BYTES.
    """
    if not isinstance(annotations, dict):
        raise ValueError("Annotations must be a JSON object")
    try:
        encoded = json.dumps(annotations, allow_nan=False)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Annotations are not JSON: {e}") from None
    size = len(encoded.encode())
    if size > MAX_ANNOTATIONS_BYTES:
        raise ValueError(f"Annotations take {size} bytes; the limit is {MAX_ANNOTATIONS_BYTES}")
    plain: dict[str, Any] = json.loads(encoded)
    return plain


def _check_root(adapter: str, series: str, stored: bytes, submitted: bytes) -> None:
    """Raise ValueError unless a submit's base model and recipe match the series' root."""
    if stored == submitted:
        return
    old, new = json.loads(stored), json.loads(submitted)
    what = "base model" if old["base"] != new["base"] else "recipe"
    raise ValueError(
        f"Series {adapter}/{series} was started with another {what}; "
        f"submit to a new series to change it"
    )


def _write_atomic(path: Path, text: str) -> None:
    """Write ``text`` to a temp file next to ``path`` and rename it into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
