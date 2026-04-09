"""Unit tests for DC-1: normalize, segment, extract mentions."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from mnc.datasets._mentions import extract_mentions
from mnc.datasets._text import build_retrieval_text, normalize_document_text
from mnc.datasets.normalize import normalize_dataset, segment_sentences
from mnc.schemas.document import DocumentRecord
from mnc.schemas.manifest import BronzeManifest

if TYPE_CHECKING:
    from pathlib import Path


def _ts() -> datetime:
    return datetime.now(tz=UTC)


def _doc(raw_text: str, doc_id: str = "test:1") -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        source="test",
        language="vi",
        raw_text=raw_text,
        created_at=_ts(),
    )


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


class TestNormalizeDocumentText:
    def test_happy_path(self) -> None:
        text = "  Bệnh nhân  bị  viêm gan B.  \n\n  "
        result = normalize_document_text(text)
        assert "Bệnh nhân" in result
        assert "  " not in result
        assert result == result.strip()

    def test_preserves_vietnamese_diacritics(self) -> None:
        text = "Tiếng Việt có dấu"
        assert normalize_document_text(text) == "Tiếng Việt có dấu"

    def test_normalizes_unicode_nfkc(self) -> None:
        # Fullwidth Latin letters → ASCII
        text = "\uff28\uff25\uff2c\uff2c\uff2f"
        assert normalize_document_text(text) == "HELLO"

    def test_collapses_whitespace(self) -> None:
        text = "a\t\tb   c"
        result = normalize_document_text(text)
        assert "  " not in result
        assert "\t" not in result

    def test_empty_string(self) -> None:
        assert normalize_document_text("") == ""

    def test_removes_zero_width_chars(self) -> None:
        text = "a\u200bb\u200dc"
        assert normalize_document_text(text) == "abc"

    def test_normalizes_line_endings(self) -> None:
        text = "line1\r\nline2\rline3"
        result = normalize_document_text(text)
        assert "\r" not in result
        assert result == "line1\nline2\nline3"


class TestBuildRetrievalText:
    def test_happy_path(self) -> None:
        normalized = "Bệnh nhân bị viêm gan B."
        result = build_retrieval_text(normalized)
        assert "bệnh" in result
        assert "." not in result

    def test_preserves_vietnamese(self) -> None:
        result = build_retrieval_text("Bệnh viêm gan")
        assert "bệnh" in result
        assert "viêm" in result
        assert "gan" in result

    def test_collapses_whitespace(self) -> None:
        result = build_retrieval_text("a  b   c")
        assert "  " not in result

    def test_does_not_stem(self) -> None:
        result = build_retrieval_text("điều trị")
        assert "điều trị" in result

    def test_empty(self) -> None:
        assert build_retrieval_text("") == ""


# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------


class TestSegmentSentences:
    def test_returns_ordered_spans(self) -> None:
        text = "Bệnh nhân bị sốt. Họ đến khám."
        spans = segment_sentences(text, "doc:1", _ts())
        assert len(spans) >= 1
        for i in range(len(spans) - 1):
            assert spans[i].char_end <= spans[i + 1].char_start

    def test_offsets_valid(self) -> None:
        text = "Bệnh nhân bị sốt. Đi khám."
        spans = segment_sentences(text, "doc:1", _ts())
        for span in spans:
            assert span.text == text[span.char_start : span.char_end]

    def test_deterministic_sentence_id(self) -> None:
        text = "Câu một. Câu hai."
        spans = segment_sentences(text, "doc:1", _ts())
        for i, span in enumerate(spans):
            assert span.sentence_id == f"doc:1:s:{i}"

    def test_empty_text_returns_empty(self) -> None:
        spans = segment_sentences("", "doc:1", _ts())
        assert spans == []

    def test_no_overlapping_spans(self) -> None:
        text = "A. B. C."
        spans = segment_sentences(text, "doc:1", _ts())
        for i in range(len(spans) - 1):
            assert spans[i].char_end <= spans[i + 1].char_start


# ---------------------------------------------------------------------------
# Mention extraction
# ---------------------------------------------------------------------------


class TestExtractMentions:
    def test_disease_extraction(self) -> None:
        doc = _doc("Bệnh nhân bị viêm gan B và ung thư gan.")
        mentions = extract_mentions(doc, _ts())
        disease_texts = [m.text for m in mentions if m.mention_type == "disease"]
        assert any(
            "viêm gan" in t.lower() or "ung thư" in t.lower() for t in disease_texts
        )

    def test_symptom_extraction(self) -> None:
        doc = _doc("Bệnh nhân sốt cao và ho kéo dài.")
        mentions = extract_mentions(doc, _ts())
        symptom_texts = [m.text for m in mentions if m.mention_type == "symptom"]
        assert len(symptom_texts) >= 1

    def test_abbreviation_extraction(self) -> None:
        doc = _doc("Điều trị HIV và AIDS cho bệnh nhân.")
        mentions = extract_mentions(doc, _ts())
        abbr_texts = [m.text for m in mentions if m.mention_type == "abbreviation"]
        assert any("HIV" in t for t in abbr_texts)
        assert any("AIDS" in t for t in abbr_texts)

    def test_deterministic_mention_id(self) -> None:
        doc = _doc("Bệnh viêm gan B cần điều trị.")
        m1 = extract_mentions(doc, _ts())
        m2 = extract_mentions(doc, _ts())
        assert [m.mention_id for m in m1] == [m.mention_id for m in m2]

    def test_no_zero_length_spans(self) -> None:
        doc = _doc("Bệnh nhân bị viêm gan.")
        mentions = extract_mentions(doc, _ts())
        for m in mentions:
            assert m.char_start < m.char_end

    def test_offsets_valid_against_raw_text(self) -> None:
        doc = _doc("Bệnh nhân bị viêm gan B và ung thư phổi.")
        mentions = extract_mentions(doc, _ts())
        for m in mentions:
            assert m.text == doc.raw_text[m.char_start : m.char_end]

    def test_deduplication_overlapping(self) -> None:
        doc = _doc("Bệnh viêm gan B")
        mentions = extract_mentions(doc, _ts())
        starts = [m.char_start for m in mentions]
        assert len(starts) == len(set(starts))  # no duplicate starts


# ---------------------------------------------------------------------------
# DC-1 pipeline integration
# ---------------------------------------------------------------------------


class TestNormalizeDataset:
    def _write_bronze(
        self,
        tmp_path: Path,
        records: list[DocumentRecord],
        split: str = "train",
    ) -> Path:
        bronze_dir = tmp_path / "bronze" / "test-ds" / "documents"
        bronze_dir.mkdir(parents=True, exist_ok=True)
        with (bronze_dir / f"{split}.jsonl").open("w") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")
        return tmp_path / "bronze"

    def test_manifest_generation(self, tmp_path: Path) -> None:
        bronze_dir = self._write_bronze(tmp_path, [_doc("Bệnh viêm gan B.")])
        silver_dir = tmp_path / "silver"
        doc_m, _sent_m, _ment_m = normalize_dataset("test-ds", bronze_dir, silver_dir)
        assert isinstance(doc_m, BronzeManifest)
        assert "train" in doc_m.record_count_by_split

    def test_silver_documents_populated(self, tmp_path: Path) -> None:
        bronze_dir = self._write_bronze(
            tmp_path,
            [_doc("Bệnh viêm gan B cần điều trị.")],
        )
        silver_dir = tmp_path / "silver"
        normalize_dataset("test-ds", bronze_dir, silver_dir)

        doc_file = silver_dir / "test-ds" / "documents" / "train.jsonl"
        assert doc_file.exists()
        with doc_file.open() as f:
            rec = json.loads(f.readline())
            assert rec["normalized_text"]
            assert rec["retrieval_text"]
            assert rec["sentences"] is not None

    def test_invalid_input_skipped(self, tmp_path: Path) -> None:
        bronze_dir = tmp_path / "bronze" / "test-ds" / "documents"
        bronze_dir.mkdir(parents=True)
        with (bronze_dir / "train.jsonl").open("w") as f:
            f.write("not valid json\n")
        silver_dir = tmp_path / "silver"
        doc_m, _, _ = normalize_dataset("test-ds", bronze_dir.parent.parent, silver_dir)
        assert doc_m.failed_count_by_split.get("train", 0) == 1

    def test_missing_input_dir_fails(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            normalize_dataset("nonexistent", tmp_path, tmp_path / "silver")
