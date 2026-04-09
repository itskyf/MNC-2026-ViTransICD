"""Unit tests for ON-2 ontology normalizer."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from mnc.datasets._io import write_jsonl
from mnc.datasets.ontology_normalizer import (
    SOURCE,
    _build_search_text,
    _deduplicate_aliases,
    _find_section_split,
    _parse_code_section,
    _validate_output,
    normalize_icd10_ontology,
)
from mnc.schemas.document import DocumentRecord
from mnc.schemas.ontology import OntologyCode

if TYPE_CHECKING:
    from pathlib import Path


def _ts() -> datetime:
    return datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _make_bronze_page(page_no: int, text: str) -> DocumentRecord:
    """Create a mock bronze page record."""
    return DocumentRecord(
        doc_id=f"{SOURCE}:page:{page_no}",
        source=SOURCE,
        language="vi",
        raw_text=text,
        source_record_id=str(page_no),
        split=None,
        payload={
            "page_no": page_no,
            "extractor": "pymupdf",
            "is_empty_page": text == "",
        },
        normalized_text="",
        retrieval_text="",
        sentences=[],
        created_at=_ts(),
    )


# -- Sample bilingual page mimicking real PDF structure.
SAMPLE_PAGE_TEXT = (
    "\u2013 1 \u2013\n"
    "\n"
    "Chapter I\n"
    "Certain infectious and\n"
    "parasitic diseases\n"
    "(A00-B99)\n"
    "\n"
    "A00\n"
    "Cholera\n"
    "A00.0 Cholera due to Vibrio cholerae 01\n"
    "Classical cholera\n"
    "A00.9 Cholera, unspecified\n"
    "A01\n"
    "Typhoid and paratyphoid fevers\n"
    "A01.0 Typhoid fever\n"
    "A01.9 Paratyphoid fever, unspecified\n"
    "A02\n"
    "Other salmonella infections\n"
    "\n"
    "Ch\u01b0\u01a1ng I\n"
    "B\u1ec7nh nhi\u1ec5m tr\u00f9ng v\u00e0 k\u00fd sinh\n"
    "tr\u00f9ng\n"
    "(A00-B99)\n"
    "\n"
    "A00\n"
    "B\u1ec7nh t\u1ea3\n"
    "A00.0 B\u1ec7nh t\u1ea3 do Vibrio cholerae 01\n"
    "B\u1ec7nh t\u1ea3 c\u1ed5 \u0111i\u1ec3n\n"
    "A00.9 B\u1ec7nh t\u1ea3, kh\u00f4ng x\u00e1c \u0111\u1ecbnh\n"
    "A01\n"
    "B\u1ec7nh th\u01b0\u01a1ng h\u00e0n v\u00e0 ph\u00f3 th\u01b0\u01a1ng h\u00e0n\n"
    "A01.0 Th\u01b0\u01a1ng h\u00e0n\n"
    "A01.9 B\u1ec7nh ph\u00f3 th\u01b0\u01a1ng h\u00e0n, "
    "kh\u00f4ng x\u00e1c \u0111\u1ecbnh\n"
    "A02\n"
    "Nhi\u1ec5m salmonella kh\u00e1c\n"
)


def _write_bronze_page(
    text: str,
    page_no: int,
    path: Path,
) -> None:
    """Write a single bronze page to JSONL."""
    rec = _make_bronze_page(page_no, text)
    write_jsonl([rec], path)


# -- Schema validation --


class TestSchemaValidation:
    def test_valid_ontology_record(self) -> None:
        rec = OntologyCode(
            code_3char="A00",
            chapter_id="Ch-1",
            title_vi="Bệnh tả",
            title_en="Cholera",
            aliases=["Classical cholera"],
            search_text=_build_search_text(
                "A00",
                "Bệnh tả",
                "Cholera",
                ["Classical cholera"],
            ),
            created_at=_ts(),
        )
        assert rec.code_3char == "A00"
        assert rec.chapter_id == "Ch-1"
        assert rec.title_vi == "Bệnh tả"
        assert rec.title_en == "Cholera"
        assert "Classical cholera" in rec.aliases
        assert "A00" in rec.search_text
        assert "Bệnh tả" in rec.search_text
        assert "Cholera" in rec.search_text


# -- 3-char code normalization --


class TestCodeNormalization:
    @pytest.mark.parametrize("code", ["A00", "B99", "Z99"])
    def test_valid_codes(self, code: str) -> None:
        rec = OntologyCode(
            code_3char=code,
            chapter_id=None,
            title_vi="Test",
            title_en=None,
            aliases=[],
            search_text=_build_search_text(code, "Test", None, []),
            created_at=_ts(),
        )
        assert rec.code_3char == code

    def test_invalid_code_rejected(self) -> None:
        rec = OntologyCode(
            code_3char="invalid",
            chapter_id=None,
            title_vi="Test",
            title_en=None,
            aliases=[],
            search_text=_build_search_text("invalid", "Test", None, []),
            created_at=_ts(),
        )
        with pytest.raises(ValueError, match="Invalid 3-char code"):
            _validate_output([rec])


# -- Chapter propagation --


class TestChapterPropagation:
    def test_chapter_assigned_to_codes(self) -> None:
        page_text = (
            "Chapter I\n"
            "Certain infectious diseases\n"
            "(A00-B99)\n"
            "A00\nCholera\n"
            "A01\nTyphoid\n"
            "Chương I\n"
            "Bệnh nhiễm trùng\n"
            "(A00-B99)\n"
            "A00\nBệnh tả\n"
            "A01\nBệnh thương hàn\n"
        )
        titles, _ = _parse_code_section(page_text.split("\n")[:9])
        assert "A00" in titles
        assert titles["A00"] == "Cholera"


# -- Bilingual title extraction --


class TestBilingualTitles:
    def test_en_and_vi_titles_extracted(self) -> None:
        lines = SAMPLE_PAGE_TEXT.split("\n")
        split = _find_section_split(lines)
        en_lines = lines[:split]
        vi_lines = lines[split:]

        en_titles, _ = _parse_code_section(en_lines)
        vi_titles, _ = _parse_code_section(vi_lines)

        assert en_titles.get("A00") == "Cholera"
        assert en_titles.get("A01") == "Typhoid and paratyphoid fevers"
        assert vi_titles.get("A00") == "Bệnh tả"
        assert vi_titles.get("A01") == "Bệnh thương hàn và phó thương hàn"


# -- Alias deduplication --


class TestAliasDedup:
    def test_removes_duplicates(self) -> None:
        aliases = ["Cholera", "cholera", "Classical cholera", ""]
        result = _deduplicate_aliases(aliases)
        assert len(result) == 2
        assert "" not in result

    def test_preserves_order(self) -> None:
        aliases = ["Alpha", "Beta", "alpha"]
        result = _deduplicate_aliases(aliases)
        assert result == ["Alpha", "Beta"]


# -- search_text composition --


class TestSearchText:
    def test_search_text_contains_all_fields(self) -> None:
        rec = OntologyCode(
            code_3char="A00",
            chapter_id="Ch-1",
            title_vi="Bệnh tả",
            title_en="Cholera",
            aliases=["Classical cholera", "Bệnh tả cổ điển"],
            search_text=_build_search_text(
                "A00",
                "Bệnh tả",
                "Cholera",
                ["Classical cholera", "Bệnh tả cổ điển"],
            ),
            created_at=_ts(),
        )
        assert "A00" in rec.search_text
        assert "Bệnh tả" in rec.search_text
        assert "Cholera" in rec.search_text
        assert "Classical cholera" in rec.search_text


# -- JSONL writer --


class TestJsonlWriter:
    def test_write_and_validate(self, tmp_path: Path) -> None:
        bronze_path = tmp_path / "bronze" / "document_records.jsonl"
        _write_bronze_page(SAMPLE_PAGE_TEXT, 20, bronze_path)

        output_dir = tmp_path / "silver"
        normalize_icd10_ontology(
            input_path=bronze_path,
            output_dir=output_dir,
        )

        jsonl_path = output_dir / "ontology_codes.jsonl"
        assert jsonl_path.exists()

        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 3

        for line in lines:
            data = json.loads(line)
            rec = OntologyCode.model_validate(data)
            assert rec.code_3char[0].isupper()
            assert len(rec.code_3char) == 3


# -- Parser smoke test --


class TestParserSmoke:
    def test_extracts_codes_from_mock_page(self, tmp_path: Path) -> None:
        bronze_path = tmp_path / "bronze" / "document_records.jsonl"
        _write_bronze_page(SAMPLE_PAGE_TEXT, 20, bronze_path)

        records = normalize_icd10_ontology(
            input_path=bronze_path,
            output_dir=tmp_path / "silver",
        )

        codes = {r.code_3char for r in records}
        assert "A00" in codes
        assert "A01" in codes
        assert "A02" in codes

        a00 = next(r for r in records if r.code_3char == "A00")
        assert a00.title_vi == "Bệnh tả"
        assert a00.title_en == "Cholera"

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            normalize_icd10_ontology(
                input_path=tmp_path / "nonexistent.jsonl",
            )


# -- Deterministic output --


class TestDeterministic:
    def test_same_output_across_runs(self, tmp_path: Path) -> None:
        bronze_path = tmp_path / "bronze" / "document_records.jsonl"
        _write_bronze_page(SAMPLE_PAGE_TEXT, 20, bronze_path)

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        r1 = normalize_icd10_ontology(
            input_path=bronze_path,
            output_dir=out1,
        )
        r2 = normalize_icd10_ontology(
            input_path=bronze_path,
            output_dir=out2,
        )

        codes1 = [r.code_3char for r in r1]
        codes2 = [r.code_3char for r in r2]
        assert codes1 == codes2
