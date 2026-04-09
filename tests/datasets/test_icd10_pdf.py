"""Unit tests for the ICD-10 PDF bronze parser."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import pymupdf
import pytest

from mnc.datasets.icd10_pdf import (
    SOURCE,
    _build_record,
    _clean_text,
    _extract_pages,
    _PageContext,
    _write_jsonl,
    parse_icd10_official_pdf,
)
from mnc.schemas.document import DocumentRecord

if TYPE_CHECKING:
    from pathlib import Path


def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Create an in-memory PDF with one string of text per page."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def _ts() -> datetime:
    return datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)


def _ctx(total_pages: int = 1) -> _PageContext:
    return _PageContext(
        pdf_url="",
        pdf_sha256="hash",
        total_pages=total_pages,
        created_at=_ts(),
    )


# -- Schema validation --


class TestSchemaValidation:
    def test_valid_record(self) -> None:
        rec = _build_record(
            page_no=1,
            text="Hello",
            ctx=_PageContext(
                pdf_url="file:///test.pdf",
                pdf_sha256="abc123",
                total_pages=1,
                created_at=_ts(),
            ),
        )
        assert rec.doc_id == f"{SOURCE}:page:1"
        assert rec.source == SOURCE
        assert rec.language == "vi"
        assert rec.raw_text == "Hello"
        assert rec.source_record_id == "1"
        assert rec.split is None
        assert rec.normalized_text == ""
        assert rec.retrieval_text == ""
        assert rec.sentences == []
        assert rec.payload is not None
        assert rec.payload["page_no"] == 1
        assert rec.payload["is_empty_page"] is False


# -- doc_id formatting --


class TestDocIdFormatting:
    def test_page_key_format(self) -> None:
        rec = _build_record(
            page_no=42,
            text="text",
            ctx=_PageContext(
                pdf_url="https://example.com/x.pdf",
                pdf_sha256="deadbeef",
                total_pages=100,
                created_at=_ts(),
            ),
        )
        assert rec.doc_id == f"{SOURCE}:page:42"

    def test_page_1(self) -> None:
        rec = _build_record(page_no=1, text="first", ctx=_ctx())
        assert rec.doc_id == f"{SOURCE}:page:1"


# -- Empty page handling --


class TestEmptyPage:
    def test_empty_page_flag(self) -> None:
        rec = _build_record(page_no=5, text="", ctx=_ctx(10))
        assert rec.raw_text == ""
        assert rec.payload is not None
        assert rec.payload["is_empty_page"] is True

    def test_nonempty_page_flag(self) -> None:
        rec = _build_record(page_no=1, text="some text", ctx=_ctx())
        assert rec.payload is not None
        assert rec.payload["is_empty_page"] is False


# -- Text cleanup --


class TestCleanText:
    def test_crlf_normalization(self) -> None:
        assert _clean_text("a\r\nb\r\nc") == "a\nb\nc"

    def test_strip_whitespace(self) -> None:
        assert _clean_text("  hello  \n") == "hello"

    def test_collapse_blank_lines(self) -> None:
        assert _clean_text("a\n\n\n\nb") == "a\n\nb"

    def test_preserve_vietnamese(self) -> None:
        text = "Bệnh viêm loét dạ dày"
        assert _clean_text(text) == text


# -- JSONL writer --


class TestJsonlWriter:
    def test_write_and_read(self, tmp_path: Path) -> None:
        records = [
            _build_record(page_no=i, text=f"page {i}", ctx=_ctx(2)) for i in range(1, 3)
        ]
        out = tmp_path / "out.jsonl"
        _write_jsonl(records, out)

        lines = out.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            DocumentRecord.model_validate(data)

    def test_file_created(self, tmp_path: Path) -> None:
        out = tmp_path / "records.jsonl"
        _write_jsonl([], out)
        assert out.exists()


# -- Page extraction (mocked) --


class TestExtractPages:
    def test_extracts_text_per_page(self) -> None:
        pdf_bytes = _make_pdf_bytes(["page one", "page two"])
        pages = _extract_pages(pdf_bytes)
        assert len(pages) == 2
        assert pages[0][0] == 1
        assert pages[1][0] == 2
        assert "page one" in pages[0][1]
        assert "page two" in pages[1][1]

    def test_zero_pages_raises(self) -> None:
        p = patch("mnc.datasets.icd10_pdf.pymupdf.open")
        mocked_open = p.start()
        mocked_open.return_value.__enter__.return_value.page_count = 0
        try:
            with pytest.raises(ValueError, match="zero pages"):
                _extract_pages(b"fake")
        finally:
            p.stop()


# -- Idempotency --


class TestIdempotency:
    def test_same_keys_across_runs(self) -> None:
        ctx = _ctx(10)
        r1 = _build_record(page_no=7, text="x", ctx=ctx)
        r2 = _build_record(page_no=7, text="x", ctx=ctx)
        assert r1.doc_id == r2.doc_id
        assert r1.source_record_id == r2.source_record_id


# -- Integration: parse_icd10_official_pdf with local PDF --


class TestParseLocalPdf:
    def test_local_pdf(self, tmp_path: Path) -> None:
        pdf_bytes = _make_pdf_bytes(["alpha", "beta", "gamma"])
        pdf_path = tmp_path / "sample.pdf"
        pdf_path.write_bytes(pdf_bytes)

        out_dir = tmp_path / "output"
        records = parse_icd10_official_pdf(
            pdf_source=str(pdf_path),
            output_dir=str(out_dir),
        )

        assert len(records) == 3
        assert records[0].raw_text != ""
        assert "alpha" in records[0].raw_text
        assert records[1].source_record_id == "2"

        jsonl = out_dir / "document_records.jsonl"
        assert jsonl.exists()
        lines = jsonl.read_text().strip().split("\n")
        assert len(lines) == 3

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            parse_icd10_official_pdf("/nonexistent/path.pdf")
