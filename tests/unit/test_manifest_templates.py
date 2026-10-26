# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The llm-kelt Authors

"""Unit tests for shared prompt templates in SFT manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_kelt.training.manifest import Client, Manifest, template_id
from llm_kelt.training.manifest.loader import load_manifest, save_manifest, validate_manifest
from llm_kelt.training.manifest.schema import Data
from llm_kelt.training.manifest.templates import (
    render_record,
    render_records,
    slot_names,
    validate_templates,
)

TEMPLATE = "Classify the text below.\n\n{{text}}\n\nLabels: foo, bar.\nAnswer with one label."
TID = template_id(TEMPLATE)


def _expanded(text: str, output: str) -> dict:
    """The plain record a templated record must render to."""
    return {"instruction": TEMPLATE.replace("{{text}}", text), "output": output}


def _templated(text: str, output: str) -> dict:
    return {"template": TID, "slots": {"text": text}, "output": output}


def _sft_manifest(records: list[dict], templates: dict[str, str]) -> Manifest:
    return Manifest(
        adapter="test-sft",
        method="sft",
        data=Data(format="inline", records=records, templates=templates),
    )


class TestTemplateId:
    def test_is_sha256_hex_of_utf8_text(self):
        import hashlib

        assert TID == hashlib.sha256(TEMPLATE.encode("utf-8")).hexdigest()
        assert len(TID) == 64

    def test_changes_with_text(self):
        assert template_id(TEMPLATE + " ") != TID

    def test_short_id_is_128_bit_prefix(self):
        assert template_id(TEMPLATE, length=32) == TID[:32]

    @pytest.mark.parametrize("length", [0, 16, 40, 65])
    def test_other_lengths_rejected(self, length: int):
        with pytest.raises(ValueError, match="must be one of"):
            template_id(TEMPLATE, length=length)


class TestShortIds:
    SHORT = TID[:32]

    def test_valid(self):
        assert validate_templates({self.SHORT: TEMPLATE}) == []

    def test_mismatch_rejected(self):
        errors = validate_templates({"0" * 32: TEMPLATE})
        assert len(errors) == 1
        assert "does not match" in errors[0]

    @pytest.mark.parametrize("tid", [TID[:31], TID[:33], TID[:40], 12345])
    def test_other_lengths_and_non_strings_rejected(self, tid):
        errors = validate_templates({tid: TEMPLATE})
        assert len(errors) == 1
        assert "hex characters" in errors[0]

    def test_renders_like_full_id(self):
        """A set keyed by the 32-char id renders to the same records as the full-id set."""
        full = [_templated(t, "o") for t in ("foo", "bar")]
        short = [{**r, "template": self.SHORT} for r in full]
        assert render_records({self.SHORT: TEMPLATE}, short) == render_records(
            {TID: TEMPLATE}, full
        )

    def test_manifest_round_trip(self, tmp_path: Path):
        record = {"template": self.SHORT, "slots": {"text": "t"}, "output": "o"}
        path = tmp_path / "manifest.yaml"
        save_manifest(_sft_manifest([record], {self.SHORT: TEMPLATE}), path)
        assert validate_manifest(load_manifest(path)) == []


class TestSlotNames:
    def test_finds_named_slots(self):
        assert slot_names("{{a}} and {{b_2}} and {{a}}") == {"a", "b_2"}

    def test_ignores_single_braces_and_invalid_names(self):
        assert slot_names('{"key": 1} {{ a }} {{1x}} {{}}') == set()


class TestValidateTemplates:
    def test_valid(self):
        assert validate_templates({TID: TEMPLATE}) == []

    def test_id_mismatch(self):
        errors = validate_templates({"0" * 64: TEMPLATE})
        assert len(errors) == 1
        assert "does not match" in errors[0]

    def test_non_string_text(self):
        assert "must be a string" in validate_templates({TID: 42})[0]

    def test_not_a_mapping(self):
        assert "must be a mapping" in validate_templates([TEMPLATE])[0]


class TestRenderRecord:
    def test_renders_instruction_and_drops_template_fields(self):
        rendered = render_record({TID: TEMPLATE}, _templated("foo text", "foo"))
        assert rendered == _expanded("foo text", "foo")

    def test_keeps_other_fields(self):
        record = {**_templated("foo text", "foo"), "input": "extra", "weight": 0.5}
        rendered = render_record({TID: TEMPLATE}, record)
        assert rendered["input"] == "extra"
        assert rendered["weight"] == 0.5

    def test_repeated_slot_filled_everywhere(self):
        text = "{{x}} then {{x}}"
        rendered = render_record(
            {template_id(text): text}, {"template": template_id(text), "slots": {"x": "foo"}}
        )
        assert rendered["instruction"] == "foo then foo"

    def test_slot_values_are_not_expanded(self):
        rendered = render_record({TID: TEMPLATE}, _templated("{{text}} $1 \\1", "bar"))
        assert "{{text}} $1 \\1" in rendered["instruction"]

    def test_template_without_slots(self):
        text = "Fixed prompt."
        rendered = render_record(
            {template_id(text): text}, {"template": template_id(text), "output": "x"}
        )
        assert rendered == {"instruction": "Fixed prompt.", "output": "x"}

    @pytest.mark.parametrize(
        ("record", "message"),
        [
            ({"template": ["not", "a", "string"], "slots": {}}, "must be a string"),
            ({"template": "f" * 64, "slots": {"text": "t"}}, "unknown template"),
            ({"template": TID, "slots": {}}, "missing slots ['text']"),
            ({"template": TID, "slots": {"text": "t", "other": "o"}}, "unknown slots ['other']"),
            ({"template": TID, "slots": {"text": 3}}, "must be strings"),
            ({"template": TID, "slots": ["t"]}, "slots must be a mapping"),
            ({"template": TID, "slots": {"text": "t"}, "instruction": "i"}, "must not also set"),
        ],
    )
    def test_invalid_records(self, record: dict, message: str):
        with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
            render_record({TID: TEMPLATE}, record)


class TestRenderRecords:
    def test_compact_set_renders_to_expanded_set(self):
        """A compact set and its expanded form are the same records after rendering."""
        texts = ["foo text", "bar text", 'with "quotes" and\nnewlines']
        compact = [_templated(t, f"out-{i}") for i, t in enumerate(texts)]
        expanded = [_expanded(t, f"out-{i}") for i, t in enumerate(texts)]
        assert render_records({TID: TEMPLATE}, compact) == expanded

    def test_plain_records_pass_through(self):
        plain = {"instruction": "i", "output": "o"}
        assert render_records({TID: TEMPLATE}, [plain, _templated("t", "o")])[0] is plain

    def test_reports_failing_record_indices_with_cap(self):
        bad = {"template": TID, "slots": {}}
        with pytest.raises(ValueError) as exc:
            render_records({TID: TEMPLATE}, [_templated("t", "o")] + [bad] * 12)
        assert "record 1:" in str(exc.value)
        assert "record 11:" not in str(exc.value)
        assert "(and 2 more)" in str(exc.value)


class TestValidateManifestTemplates:
    def test_valid_templated_manifest(self):
        assert validate_manifest(_sft_manifest([_templated("t", "o")], {TID: TEMPLATE})) == []

    def test_id_mismatch(self):
        errors = validate_manifest(_sft_manifest([_templated("t", "o")], {"0" * 64: TEMPLATE}))
        assert any("does not match" in e for e in errors)

    def test_bad_inline_record(self):
        record = {"template": TID, "slots": {"wrong": "t"}, "output": "o"}
        errors = validate_manifest(_sft_manifest([record], {TID: TEMPLATE}))
        assert any("record 0" in e and "missing slots" in e for e in errors)

    def test_external_records_not_checked_at_validation(self):
        manifest = Manifest(
            adapter="test-sft",
            method="sft",
            data=Data(format="external", path="data.jsonl", templates={TID: TEMPLATE}),
        )
        assert validate_manifest(manifest) == []

    def test_rejected_for_non_sft_methods(self):
        manifest = Manifest(
            adapter="test-dpo",
            method="dpo",
            data=Data(
                format="inline",
                records=[{"prompt": "p", "chosen": "c", "rejected": "r"}],
                templates={TID: TEMPLATE},
            ),
        )
        assert any("only supported for method 'sft'" in e for e in validate_manifest(manifest))


class TestTemplateRoundTrip:
    @pytest.mark.parametrize(
        "text",
        [
            TEMPLATE,
            "  leading spaces\ntrailing spaces  \n\n\ttab {{text}}\n",
            "unicode: é – 漢字 {{text}} 'single' \"double\" # not a comment: yes",
        ],
    )
    def test_save_load_keeps_text_and_id(self, tmp_path: Path, text: str):
        tid = template_id(text)
        record = {"template": tid, "slots": {"text": "yes: 1.0 # no"}, "output": "o"}
        path = tmp_path / "manifest.yaml"
        save_manifest(_sft_manifest([record], {tid: text}), path)

        loaded = load_manifest(path)
        assert loaded.data.templates[tid] == text
        assert loaded.data.records[0]["slots"] == {"text": "yes: 1.0 # no"}
        assert validate_manifest(loaded) == []

    def test_manifest_without_templates_has_no_templates_key(self, tmp_path: Path):
        path = tmp_path / "manifest.yaml"
        save_manifest(_sft_manifest([{"instruction": "i", "output": "o"}], {}), path)
        assert "templates" not in path.read_text()


class TestClientCreate:
    @pytest.fixture
    def client(self, lg, tmp_path: Path) -> Client:
        from llm_kelt.training.storage import FileStorage

        return Client(lg=lg, storage=FileStorage(lg, tmp_path))

    def test_create_with_templates(self, client: Client):
        manifest = client.create(
            adapter="test-sft",
            method="sft",
            data=[_templated("t", "o")],
            templates={TID: TEMPLATE},
        )
        assert dict(manifest.data.templates) == {TID: TEMPLATE}
        assert validate_manifest(manifest) == []

    def test_create_without_templates(self, client: Client):
        manifest = client.create(
            adapter="test-sft", method="sft", data=[{"instruction": "i", "output": "o"}]
        )
        assert dict(manifest.data.templates) == {}


class TestSftClientRendering:
    @pytest.fixture
    def sft_client(self, lg, tmp_path: Path):
        from llm_kelt.training.sft import Client as SftClient
        from llm_kelt.training.storage import FileStorage

        return SftClient(lg, FileStorage(lg, tmp_path / "registry"))

    def _write_jsonl(self, path: Path, records: list[dict]) -> Path:
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        return path

    def test_renders_external_records(self, sft_client, tmp_path: Path):
        data_path = self._write_jsonl(tmp_path / "data.jsonl", [_templated("foo text", "foo")])
        manifest = Manifest(
            adapter="test-sft",
            method="sft",
            data=Data(format="external", path=str(data_path), templates={TID: TEMPLATE}),
        )
        work_dir, rendered_path = sft_client._prepare_training(manifest)

        assert rendered_path.parent == work_dir
        lines = rendered_path.read_text(encoding="utf-8").splitlines()
        assert [json.loads(line) for line in lines] == [_expanded("foo text", "foo")]

    def test_without_templates_keeps_data_path(self, sft_client, tmp_path: Path):
        data_path = self._write_jsonl(
            tmp_path / "data.jsonl", [{"instruction": "i", "output": "o"}]
        )
        manifest = Manifest(
            adapter="test-sft", method="sft", data=Data(format="external", path=str(data_path))
        )
        _, resolved = sft_client._prepare_training(manifest)
        assert resolved == data_path

    def test_invalid_templates_raise(self, sft_client, tmp_path: Path):
        data_path = self._write_jsonl(tmp_path / "data.jsonl", [_templated("t", "o")])
        manifest = Manifest(
            adapter="test-sft",
            method="sft",
            data=Data(format="external", path=str(data_path), templates={"0" * 64: TEMPLATE}),
        )
        with pytest.raises(ValueError, match="Invalid SFT templates"):
            sft_client._prepare_training(manifest)

    def test_inline_templated_records_pass_record_validation(self, sft_client):
        sft_client._validate_records(_sft_manifest([_templated("t", "o")], {TID: TEMPLATE}))
