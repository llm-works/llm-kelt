# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Keys that partition stored knowledge, and the patterns that select them.

A key is an opaque string; the only structure assumed is ':' between its parts.
Reads take a key or a pattern, writes take a key only.

Patterns:
    *       any run of characters, ':' included
    ?       exactly one character
    K:**    K itself and every key below it (K:...)

Every key or pattern becomes SQL through parse_key(), so the forms above are
defined in one place.
"""

from dataclasses import dataclass
from typing import Any

from sqlalchemy import ColumnElement, SQLColumnExpression, or_

from .errors import ValidationError

SUBTREE_SUFFIX = ":**"
_WILDCARDS = frozenset("*?")


def is_key_pattern(key: str) -> bool:
    """Whether the key selects stored keys (has * or ?) instead of naming one."""
    return not _WILDCARDS.isdisjoint(key)


def require_exact_key(key: str) -> str:
    """Return the key unchanged, or raise ValidationError if it is a pattern."""
    if is_key_pattern(key):
        raise ValidationError(f"key pattern selects stored keys and cannot be written: {key!r}")
    return key


@dataclass(frozen=True)
class KeyMatch:
    """A key or pattern in SQL terms: a value matches if any of the terms does."""

    equals: str | None = None
    likes: tuple[str, ...] = ()


def _to_like(text: str) -> str:
    """Translate glob text to a LIKE pattern, escaping LIKE's own wildcards (escape: \\)."""
    escaped = text.replace("\\", "\\\\").replace("%", r"\%").replace("_", r"\_")
    return escaped.replace("*", "%").replace("?", "_")


def parse_key(key: str) -> KeyMatch:
    """Reduce a key or pattern to SQL terms.

    Examples:
        "acme:prod"     -> equals "acme:prod"
        "acme:*"        -> like "acme:%"
        "acme:prod:**"  -> equals "acme:prod", or like "acme:prod:%"
    """
    if key.endswith(SUBTREE_SUFFIX):
        base = key[: -len(SUBTREE_SUFFIX)]
        own = parse_key(base)
        return KeyMatch(equals=own.equals, likes=(*own.likes, _to_like(base) + ":%"))
    if is_key_pattern(key):
        return KeyMatch(likes=(_to_like(key),))
    return KeyMatch(equals=key)


def key_filter(key: str, column: SQLColumnExpression[Any]) -> ColumnElement[bool]:
    """SQLAlchemy condition selecting the rows whose column matches the key or pattern."""
    match = parse_key(key)
    terms: list[ColumnElement[bool]] = [column.like(p, escape="\\") for p in match.likes]
    if match.equals is not None:
        terms.insert(0, column == match.equals)
    return terms[0] if len(terms) == 1 else or_(*terms)


def key_sql(key: str, column: str, param: str) -> tuple[str, dict[str, str]]:
    """Raw SQL condition and bind parameters selecting rows whose column matches the key.

    Args:
        key: Key or pattern.
        column: Column reference as written in the SQL (e.g. "r.context_key").
        param: Prefix for the bind parameter names.
    """
    match = parse_key(key)
    terms: list[str] = []
    params: dict[str, str] = {}
    if match.equals is not None:
        params[param] = match.equals
        terms.append(f"{column} = :{param}")
    for i, pattern in enumerate(match.likes):
        name = f"{param}_{i}"
        params[name] = pattern
        terms.append(f"{column} LIKE :{name} ESCAPE '\\'")
    return f"({' OR '.join(terms)})", params
