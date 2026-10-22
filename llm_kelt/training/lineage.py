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
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from appinfra.cas import (
    Blob,
    RefConflictError,
    Repo,
    Store,
    Tree,
    TreeEntry,
    canonical_json,
    is_object_id,
)
from appinfra.log import Logger

DEFAULT_SERIES = "main"

_REF_PREFIX = "series/"


@dataclass(frozen=True)
class Build:
    """Outcome of training one commit."""

    state: Literal["completed", "failed"]
    at: datetime
    # Adapter weights md5, for a completed build
    md5: str | None = None
    # Error message, for a failed build
    error: str | None = None


@dataclass(frozen=True)
class SetEntry:
    """One set of a series, at the commit that added it."""

    commit: str
    set_id: str
    submitted_at: datetime | None
    build: Build | None


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

    def __init__(self, lg: Logger, store: Store, records_path: Path) -> None:
        """Initialize lineage.

        Args:
            lg: Logger instance.
            store: Content-addressed store; one repository per adapter key.
            records_path: Directory for submit dates and build records.
        """
        self._lg = lg
        self._store = store
        self._records_path = records_path

    def submit(
        self,
        adapter: str,
        series: str,
        records: list[dict[str, Any]],
        templates: dict[str, str],
        root: dict[str, Any],
    ) -> tuple[str, bool]:
        """Add a training set to a series.

        Args:
            adapter: Adapter key.
            series: Series name.
            records: Training records of the set.
            templates: Shared prompt templates the records reference.
            root: Root payload (see ``series_root``); the series' first submit stores it,
                later submits must match it.

        Returns:
            (commit, created): the commit that holds the set, and whether this submit
            added it. A set already in the series' history returns its existing commit.

        Raises:
            ValueError: The series was started with another base model or recipe.
        """
        repo = self._store.repo(adapter)
        set_id = _put_set(repo, records, templates)
        payload = canonical_json(root)
        ref = _REF_PREFIX + series
        while True:
            tip = repo.get_ref(ref)
            if tip is not None:
                history = list(repo.first_parents(tip))
                _check_root(adapter, series, history[-1].payload, payload)
                existing = next((c.id for c in history if c.tree == set_id), None)
                if existing is not None:
                    return existing, False
            try:
                commit = repo.commit(ref, set_id, payload if tip is None else b"")
            except RefConflictError:
                continue  # another submit moved the series; check its history again
            self._write_submitted(adapter, commit)
            self._lg.info(
                "added set to series",
                extra={"adapter": adapter, "series": series, "set": set_id, "commit": commit},
            )
            return commit, True

    def fork(self, adapter: str, series: str, new_series: str, commit: str | None = None) -> str:
        """Start ``new_series`` at a commit of ``series`` (default: its tip); return it.

        Raises:
            ValueError: ``series`` does not exist, ``commit`` is not in its history, or
                ``new_series`` exists.
        """
        repo = self._store.repo(adapter)
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
        """Record the outcome of a build of ``commit``, replacing an earlier one."""
        build = {"state": state, "at": datetime.now(UTC).isoformat(), "md5": md5, "error": error}
        _write_atomic(self._build_path(adapter, commit), yaml.safe_dump(build))

    def build(self, adapter: str, commit: str) -> Build | None:
        """Return the recorded build of ``commit``, or None if it was never built."""
        try:
            data = yaml.safe_load(self._build_path(adapter, commit).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        return Build(
            state=data["state"],
            at=datetime.fromisoformat(data["at"]),
            md5=data.get("md5"),
            error=data.get("error"),
        )

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

    def _build_path(self, adapter: str, commit: str) -> Path:
        return self._records_path / adapter / "builds" / f"{commit}.yaml"

    def _submitted_path(self, adapter: str, commit: str) -> Path:
        return self._records_path / adapter / "submitted" / commit

    def _write_submitted(self, adapter: str, commit: str) -> None:
        """Record when a commit was added; the first submit of a commit wins."""
        path = self._submitted_path(adapter, commit)
        if not path.exists():
            _write_atomic(path, datetime.now(UTC).isoformat())

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
