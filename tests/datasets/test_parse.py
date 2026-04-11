"""Tests for DE-3: Parse source formats to bronze documents."""

import json
from pathlib import Path

import pytest

from mnc.datasets._io import write_jsonl
from mnc.datasets.parse import (
    _vietmed_sum_doc,
    _vihealthqa_doc,
    parse_dataset,
)
from mnc.schemas.snapshot import SnapshotRecord


def _vietmed_snapshot(
    record_id: str = "0",
    split: str = "train",
    transcript: str = "Bệnh nhân sốt cao ba ngày.",
    summary: str = "Tóm tắt bệnh án.",
) -> SnapshotRecord:
    """Build a minimal VietMed-Sum SnapshotRecord."""
    return SnapshotRecord(
        dataset_name="vietmed-sum",
        source_split=split,
        source_record_id=record_id,
        payload={"transcript": transcript, "summary": summary},
        source_format="parquet",
        source_path="data/train.train_whole-00000-of-00001.parquet",
        ingest_version="1.0.0",
        language="vi",
    )


def _vihealthqa_snapshot(
    record_id: str = "1",
    split: str = "train",
    question: str = "Đau đầu kéo dài là bệnh gì?",
    answer: str = "Có thể là triệu chứng.",
    link: str = "https://example.com",
) -> SnapshotRecord:
    """Build a minimal ViHealthQA SnapshotRecord."""
    return SnapshotRecord(
        dataset_name="vihealthqa",
        source_split=split,
        source_record_id=record_id,
        payload={
            "question": question,
            "answer": answer,
            "link": link,
            "id": int(record_id),
        },
        source_format="csv",
        source_path="train.csv",
        ingest_version="1.0.0",
        language="vi",
    )


class TestVietMedSumDoc:
    """VietMed-Sum document construction."""

    def test_happy_path(self) -> None:
        snap = _vietmed_snapshot()
        doc = _vietmed_sum_doc(snap)
        assert doc.doc_id == "vietmed-sum:0"
        assert doc.source == "vietmed-sum"
        assert doc.language == "vi"
        assert doc.raw_text == "Bệnh nhân sốt cao ba ngày.\nTóm tắt bệnh án."
        assert doc.payload["transcript"] == "Bệnh nhân sốt cao ba ngày."
        assert doc.payload["summary"] == "Tóm tắt bệnh án."
        assert doc.split == "train"
        assert doc.source_record_id == "0"

    def test_empty_transcript_raises(self) -> None:
        snap = _vietmed_snapshot(transcript="")
        with pytest.raises(ValueError, match="transcript"):
            _vietmed_sum_doc(snap)

    def test_none_transcript_raises(self) -> None:
        snap = _vietmed_snapshot()
        snap.payload = {**snap.payload, "transcript": None}
        with pytest.raises(ValueError, match="transcript"):
            _vietmed_sum_doc(snap)

    def test_summary_content_included_in_raw_text(self) -> None:
        """Disease names in summary must be available in raw_text."""
        snap = _vietmed_snapshot(
            transcript="Bệnh nhân đến khám.",
            summary="Chẩn đoán bệnh Parkinson.",
        )
        doc = _vietmed_sum_doc(snap)
        assert "Parkinson" in doc.raw_text
        assert doc.raw_text == "Bệnh nhân đến khám.\nChẩn đoán bệnh Parkinson."


class TestViHealthQADoc:
    """ViHealthQA document construction."""

    def test_happy_path(self) -> None:
        snap = _vihealthqa_snapshot()
        doc = _vihealthqa_doc(snap)
        assert doc.doc_id == "vihealthqa:train:1"
        assert doc.source == "vihealthqa"
        assert doc.language == "vi"
        expected = "Đau đầu kéo dài là bệnh gì?\nCó thể là triệu chứng."
        assert doc.raw_text == expected
        assert doc.payload["question"] == "Đau đầu kéo dài là bệnh gì?"
        assert doc.payload["answer"] == "Có thể là triệu chứng."
        assert doc.payload["link"] == "https://example.com"
        assert doc.split == "train"

    def test_empty_question_raises(self) -> None:
        snap = _vihealthqa_snapshot(question="")
        with pytest.raises(ValueError, match="question"):
            _vihealthqa_doc(snap)

    def test_empty_answer_raises(self) -> None:
        snap = _vihealthqa_snapshot(answer="")
        with pytest.raises(ValueError, match="answer"):
            _vihealthqa_doc(snap)


class TestDeterministicDocId:
    """doc_id must be deterministic and stable."""

    def test_vietmed_sum_deterministic(self) -> None:
        snap = _vietmed_snapshot(record_id="42")
        doc1 = _vietmed_sum_doc(snap)
        doc2 = _vietmed_sum_doc(snap)
        assert doc1.doc_id == doc2.doc_id == "vietmed-sum:42"

    def test_vihealthqa_deterministic(self) -> None:
        snap = _vihealthqa_snapshot(record_id="99", split="test")
        doc1 = _vihealthqa_doc(snap)
        doc2 = _vihealthqa_doc(snap)
        assert doc1.doc_id == doc2.doc_id == "vihealthqa:test:99"


class TestInvalidRecordErrorLogging:
    """Invalid records are logged to errors.jsonl."""

    def test_errors_file_created(self, tmp_path: Path) -> None:
        bad_snap = _vietmed_snapshot(transcript="")
        snapshot_dir = tmp_path / "vietmed-sum" / "snapshots"
        snapshot_dir.mkdir(parents=True)
        write_jsonl([bad_snap], snapshot_dir / "train.jsonl")

        manifest = parse_dataset("vietmed-sum", tmp_path)

        assert manifest.failed_count_by_split["train"] == 1
        assert manifest.record_count_by_split["train"] == 0

        errors_path = tmp_path / "vietmed-sum" / "documents" / "errors.jsonl"
        assert errors_path.exists()
        lines = errors_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        error = json.loads(lines[0])
        assert "transcript" in error["error"]

    def test_no_errors_file_when_all_valid(self, tmp_path: Path) -> None:
        snap = _vietmed_snapshot()
        snapshot_dir = tmp_path / "vietmed-sum" / "snapshots"
        snapshot_dir.mkdir(parents=True)
        write_jsonl([snap], snapshot_dir / "train.jsonl")

        parse_dataset("vietmed-sum", tmp_path)

        errors_path = tmp_path / "vietmed-sum" / "documents" / "errors.jsonl"
        assert not errors_path.exists()


class TestManifestGeneration:
    """Manifest generation for document parsing."""

    def test_manifest_written(self, tmp_path: Path) -> None:
        snaps = [_vietmed_snapshot(record_id=str(i)) for i in range(3)]
        snapshot_dir = tmp_path / "vietmed-sum" / "snapshots"
        snapshot_dir.mkdir(parents=True)
        write_jsonl(snaps, snapshot_dir / "train.jsonl")

        manifest = parse_dataset("vietmed-sum", tmp_path)

        assert manifest.dataset == "vietmed-sum"
        assert manifest.input_splits == ["train"]
        assert manifest.output_splits == ["train"]
        assert manifest.record_count_by_split["train"] == 3
        assert manifest.failed_count_by_split["train"] == 0

        manifest_path = tmp_path / "vietmed-sum" / "documents" / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["record_count_by_split"]["train"] == 3

    def test_de2_artifacts_preserved(self, tmp_path: Path) -> None:
        """DE-2 snapshot files remain intact after DE-3."""
        snap = _vietmed_snapshot()
        snapshot_dir = tmp_path / "vietmed-sum" / "snapshots"
        snapshot_dir.mkdir(parents=True)
        write_jsonl([snap], snapshot_dir / "train.jsonl")

        parse_dataset("vietmed-sum", tmp_path)

        assert (snapshot_dir / "train.jsonl").exists()
        assert (tmp_path / "vietmed-sum" / "documents" / "train.jsonl").exists()


class TestSchemaValidation:
    """Schema validation catches bad snapshot data."""

    def test_invalid_snapshot_skipped(self, tmp_path: Path) -> None:
        """A line that fails SnapshotRecord validation is counted as failed."""
        snapshot_dir = tmp_path / "vietmed-sum" / "snapshots"
        snapshot_dir.mkdir(parents=True)
        valid = _vietmed_snapshot()
        bad_json = json.dumps({"no": "required_fields"})
        path = snapshot_dir / "train.jsonl"
        path.write_text(
            valid.model_dump_json() + "\n" + bad_json + "\n",
            encoding="utf-8",
        )

        manifest = parse_dataset("vietmed-sum", tmp_path)
        assert manifest.record_count_by_split["train"] == 1
        assert manifest.failed_count_by_split["train"] == 1
