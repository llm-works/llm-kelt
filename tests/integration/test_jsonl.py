# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""JSONL reading and writing on files."""

from io import StringIO
from pathlib import Path

from llm_kelt.core.export.jsonl import (
    _open_output,
    load_jsonl,
)


class TestOpenOutput:
    """Tests for _open_output context manager."""

    def test_with_string_path(self, tmp_path):
        """Opens file from string path."""
        filepath = str(tmp_path / "test.jsonl")
        with _open_output(filepath) as f:
            f.write("test\n")

        assert Path(filepath).read_text() == "test\n"

    def test_with_path_object(self, tmp_path):
        """Opens file from Path object."""
        filepath = tmp_path / "test.jsonl"
        with _open_output(filepath) as f:
            f.write("test\n")

        assert filepath.read_text() == "test\n"

    def test_with_file_object(self):
        """Passes through file-like object unchanged."""
        sio = StringIO()
        with _open_output(sio) as f:
            f.write("test\n")

        assert sio.getvalue() == "test\n"

    def test_closes_file_on_exit(self, tmp_path):
        """File is closed after context exits."""
        filepath = tmp_path / "test.jsonl"
        with _open_output(filepath) as f:
            f.write("test\n")
            file_handle = f  # Capture handle to check closure

        # Verify file was closed and content was written
        assert file_handle.closed
        assert Path(filepath).read_text() == "test\n"


class TestLoadJsonl:
    """Tests for load_jsonl function."""

    def test_loads_empty_file(self, tmp_path):
        """Empty file returns empty list."""
        filepath = tmp_path / "empty.jsonl"
        filepath.write_text("")

        result = load_jsonl(filepath)
        assert result == []

    def test_loads_single_record(self, tmp_path):
        """Single line file returns one record."""
        filepath = tmp_path / "single.jsonl"
        filepath.write_text('{"key": "value"}\n')

        result = load_jsonl(filepath)
        assert result == [{"key": "value"}]

    def test_loads_multiple_records(self, tmp_path):
        """Multiple lines return multiple records."""
        filepath = tmp_path / "multi.jsonl"
        filepath.write_text('{"a": 1}\n{"b": 2}\n{"c": 3}\n')

        result = load_jsonl(filepath)
        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_skips_blank_lines(self, tmp_path):
        """Blank lines are skipped."""
        filepath = tmp_path / "blanks.jsonl"
        filepath.write_text('{"a": 1}\n\n{"b": 2}\n  \n{"c": 3}\n')

        result = load_jsonl(filepath)
        assert result == [{"a": 1}, {"b": 2}, {"c": 3}]

    def test_handles_unicode(self, tmp_path):
        """Unicode content is handled correctly."""
        filepath = tmp_path / "unicode.jsonl"
        filepath.write_text('{"text": "Hello"}\n', encoding="utf-8")

        result = load_jsonl(filepath)
        assert result == [{"text": "Hello"}]

    def test_accepts_string_path(self, tmp_path):
        """Accepts string path as well as Path."""
        filepath = tmp_path / "test.jsonl"
        filepath.write_text('{"key": "value"}\n')

        result = load_jsonl(str(filepath))
        assert result == [{"key": "value"}]
