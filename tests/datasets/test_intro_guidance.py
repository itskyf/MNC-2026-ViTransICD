"""Unit tests for ON-1b intro guidance extractor."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from mnc.datasets._io import write_jsonl
from mnc.datasets.intro_guidance import (
    EXPECTED_TOPIC_COUNT,
    REQUIRED_TOPICS,
    SOURCE,
    _build_record,
    _find_topic_excerpt,
    _normalize_text,
    _split_sentences,
    extract_intro_guidance,
)
from mnc.schemas.document import DocumentRecord

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


def _make_mock_bronze_pages() -> dict[int, DocumentRecord]:
    """Create mock bronze pages with intro content."""
    return {
        13: _make_bronze_page(13, "GIỚI THIỆU\n\n1. Giới thiệu khái quát về ICD-10"),
        14: _make_bronze_page(
            14,
            "ii\n\n4. Bộ mã 4 kí tự\n\nVới sự phân chia như trên",
        ),
        15: _make_bronze_page(
            15,
            "iii\n\nTrước mắt vì một số lý do về phương diện thống kê, "
            "tạm thời sử dụng bộ mã 3 kí tự hay nói cách khác "
            "tạm thời thống kê và phân loại đến tên bệnh.\n\n"
            "5. Một số nguyên tắc để mã hóa\n\n"
            "a. Xác lập chẩn đoán\n\n"
            "Nguyên tắc chung:\n"
            "Để có chẩn đoán xác định cuối cùng cần phân biệt "
            "bệnh chính và bệnh phụ. "
            "Bệnh chính được định nghĩa là bệnh lí được "
            "chẩn đoán sau cùng.",
        ),
        16: _make_bronze_page(
            16,
            "iv\n\n"
            "b. Mã hóa bệnh theo chẩn đoán\n\n"
            "Các trường hợp đặc biệt khác\n\n"
            "1. Khi không thể xác định chẩn đoán cuối cùng: "
            "ghi nhận và lựa chọn "
            "dấu hiệu và tình trang khẩn cấp nhất cần xử lý\n\n"
            "c. Tử vong\n\n"
            "Xác định nguyên nhân tử vong\n\n"
            "Khi chỉ có một nguyên nhân tử vong thì lấy "
            "nguyên nhân này là nguyên nhân chính.",
        ),
        17: _make_bronze_page(
            17,
            "v\n\nNguyên tắc lựa chọn nguyên nhân tử vong chính",
        ),
        18: _make_bronze_page(
            18,
            "vi\n\nMột số lưu ý khi lựa chọn nguyên nhân tử vong",
        ),
    }


def _write_bronze_pages(pages: dict[int, DocumentRecord], path: Path) -> None:
    """Write mock bronze pages to JSONL."""
    records = [pages[pn] for pn in sorted(pages.keys())]
    write_jsonl(records, path)


# -- Schema validation --


class TestSchemaValidation:
    def test_valid_guidance_record(self) -> None:
        rec = _build_record(
            topic="principal_diagnosis",
            raw_text="Bệnh chính được định nghĩa là bệnh lí",
            page_start=15,
            page_end=15,
            created_at=_ts(),
        )
        assert rec.doc_id == f"{SOURCE}:intro_guidance:principal_diagnosis"
        assert rec.source == SOURCE
        assert rec.language == "vi"
        assert rec.raw_text != ""
        assert rec.source_record_id == "principal_diagnosis"
        assert rec.payload is not None
        assert rec.payload["topic"] == "principal_diagnosis"
        assert rec.payload["page_start"] == 15
        assert rec.payload["page_end"] == 15
        assert rec.payload["page_span"] == "15"


# -- doc_id formatting --


class TestDocIdFormatting:
    @pytest.mark.parametrize("topic", REQUIRED_TOPICS)
    def test_doc_id_format(self, topic: str) -> None:
        rec = _build_record(
            topic=topic,
            raw_text="sample",
            page_start=1,
            page_end=1,
            created_at=_ts(),
        )
        assert rec.doc_id == f"{SOURCE}:intro_guidance:{topic}"


# -- Missing topic failure --


class TestMissingTopic:
    def test_missing_topic_raises(self) -> None:
        pages = _make_mock_bronze_pages()
        # Remove page 15 which has 3-char policy.
        del pages[15]
        result = _find_topic_excerpt("official_3char_policy", pages)
        assert result is None


# -- Multi-page excerpt consolidation --


class TestMultiPageExcerpt:
    def test_multi_page_consolidation(self) -> None:
        pages = _make_mock_bronze_pages()
        result = _find_topic_excerpt("mortality_coding", pages)
        assert result is not None
        raw_text, page_start, page_end = result
        assert page_start == 16
        assert page_end == 18
        # Should contain text from both pages.
        assert "nguyên nhân tử vong" in raw_text.lower()
        assert "nguyên nhân tử vong chính" in raw_text.lower()


# -- Sentence splitting --


class TestSentenceSplit:
    def test_sentence_split_smoke(self) -> None:
        text = "Đây là câu đầu tiên. Đây là câu thứ hai. Đây là câu thứ ba."
        sentences = _split_sentences(text)
        assert len(sentences) >= 2

    def test_empty_text(self) -> None:
        sentences = _split_sentences("")
        assert sentences == []


# -- Normalization --


class TestNormalization:
    def test_collapse_spaces(self) -> None:
        assert _normalize_text("hello   world") == "hello world"

    def test_collapse_blank_lines(self) -> None:
        assert _normalize_text("a\n\n\n\nb") == "a\n\nb"

    def test_preserve_vietnamese(self) -> None:
        text = "Bệnh viêm loét dạ dày"
        assert _normalize_text(text) == text


# -- JSONL writer integration --


class TestJsonlWriter:
    def test_write_and_validate(self, tmp_path: Path) -> None:
        pages = _make_mock_bronze_pages()
        bronze_path = tmp_path / "bronze" / "document_records.jsonl"
        _write_bronze_pages(pages, bronze_path)

        output_dir = tmp_path / "silver"
        extract_intro_guidance(
            input_path=bronze_path,
            output_dir=output_dir,
        )

        jsonl_path = output_dir / "document_records.jsonl"
        assert jsonl_path.exists()

        lines = jsonl_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == EXPECTED_TOPIC_COUNT

        for line in lines:
            data = json.loads(line)
            rec = DocumentRecord.model_validate(data)
            assert rec.source == SOURCE
            assert rec.language == "vi"


# -- Extractor smoke test --


class TestExtractorSmoke:
    def test_extract_returns_four_records(self, tmp_path: Path) -> None:
        pages = _make_mock_bronze_pages()
        bronze_path = tmp_path / "bronze" / "document_records.jsonl"
        _write_bronze_pages(pages, bronze_path)

        records = extract_intro_guidance(
            input_path=bronze_path,
            output_dir=tmp_path / "silver",
        )

        assert len(records) == EXPECTED_TOPIC_COUNT
        topics = {r.payload["topic"] for r in records if r.payload}
        assert topics == set(REQUIRED_TOPICS)

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            extract_intro_guidance(
                input_path=tmp_path / "nonexistent.jsonl",
            )


# -- Deterministic output --


class TestDeterministic:
    def test_same_keys_across_runs(self, tmp_path: Path) -> None:
        pages = _make_mock_bronze_pages()
        bronze_path = tmp_path / "bronze" / "document_records.jsonl"
        _write_bronze_pages(pages, bronze_path)

        out1 = tmp_path / "run1"
        out2 = tmp_path / "run2"
        r1 = extract_intro_guidance(
            input_path=bronze_path,
            output_dir=out1,
        )
        r2 = extract_intro_guidance(
            input_path=bronze_path,
            output_dir=out2,
        )

        ids1 = [r.doc_id for r in r1]
        ids2 = [r.doc_id for r in r2]
        assert ids1 == ids2

        topics1 = [r.payload["topic"] for r in r1 if r.payload]
        topics2 = [r.payload["topic"] for r in r2 if r.payload]
        assert topics1 == topics2
