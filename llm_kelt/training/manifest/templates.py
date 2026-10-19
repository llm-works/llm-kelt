# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Shared prompt templates for SFT records.

Prompt text shared by many records is stored once in the manifest, as a template with
named ``{{slot}}`` placeholders. A templated record references its template by id and
carries only its own slot values:

    templates: {<id>: "Summarize the text below.\\n\\n{{text}}\\n\\nAnswer in one line."}
    records:   [{"template": <id>, "slots": {"text": "..."}, "output": "..."}]

Rendering replaces ``template`` and ``slots`` with the plain ``instruction`` the trainer
reads, so a templated record trains exactly like its expanded form. Slot values are
inserted verbatim and never scanned for placeholders themselves.

The template id is the SHA-256 hex digest of the template text (UTF-8), so a changed
template gets a new id.
"""

import hashlib
import re
from typing import Any

_SLOT_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")

# Rendering errors are collected per record; cap how many are reported in one message.
_MAX_REPORTED_ERRORS = 10


def template_id(text: str) -> str:
    """Return the id of a template: the SHA-256 hex digest of its UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slot_names(text: str) -> set[str]:
    """Return the slot names a template's ``{{slot}}`` placeholders reference."""
    return set(_SLOT_RE.findall(text))


def validate_templates(templates: Any) -> list[str]:
    """Check that templates is a mapping of id to text and every id matches its text."""
    if not isinstance(templates, dict):
        return [f"templates must be a mapping of id to text, got {type(templates).__name__}"]

    errors: list[str] = []
    for tid, text in templates.items():
        if not isinstance(text, str):
            errors.append(f"template {tid}: text must be a string")
        elif tid != template_id(text):
            errors.append(f"template {tid}: id does not match the SHA-256 of its text")
    return errors


def _check_slots(text: str, slots: Any) -> None:
    """Raise ValueError unless slots provide exactly the template's slots, as strings."""
    if not isinstance(slots, dict):
        raise ValueError(f"slots must be a mapping, got {type(slots).__name__}")

    expected = slot_names(text)
    missing = expected - slots.keys()
    extra = slots.keys() - expected
    if missing:
        raise ValueError(f"missing slots {sorted(missing)}")
    if extra:
        raise ValueError(f"unknown slots {sorted(extra)}")

    non_str = sorted(k for k, v in slots.items() if not isinstance(v, str))
    if non_str:
        raise ValueError(f"slot values must be strings: {non_str}")


def render_record(templates: dict[str, str], record: dict[str, Any]) -> dict[str, Any]:
    """Render a templated record into a plain record with ``instruction``.

    Fields other than ``template`` and ``slots`` are kept as they are.

    Raises:
        ValueError: Unknown template id, slot mismatch, or the record also sets
            ``instruction`` or ``prompt``.
    """
    tid = record["template"]
    if tid not in templates:
        raise ValueError(f"unknown template {tid!r}")
    if "instruction" in record or "prompt" in record:
        raise ValueError("a templated record must not also set 'instruction' or 'prompt'")

    text = templates[tid]
    slots = record.get("slots", {})
    _check_slots(text, slots)

    rendered = {k: v for k, v in record.items() if k not in ("template", "slots")}
    rendered["instruction"] = _SLOT_RE.sub(lambda m: slots[m.group(1)], text)
    return rendered


def render_records(
    templates: dict[str, str], records: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Render every templated record; records without ``template`` pass through unchanged.

    Raises:
        ValueError: Listing the first failing records by index.
    """
    rendered: list[dict[str, Any]] = []
    errors: list[str] = []
    for i, record in enumerate(records):
        if "template" not in record:
            rendered.append(record)
            continue
        try:
            rendered.append(render_record(templates, record))
        except ValueError as e:
            errors.append(f"record {i}: {e}")

    if errors:
        shown = "; ".join(errors[:_MAX_REPORTED_ERRORS])
        more = len(errors) - _MAX_REPORTED_ERRORS
        suffix = f" (and {more} more)" if more > 0 else ""
        raise ValueError(f"Invalid templated SFT records: {shown}{suffix}")
    return rendered
