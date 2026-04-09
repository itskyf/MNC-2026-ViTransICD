"""Tests for DE-2: Snapshot public datasets to bronze."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from pathlib import Path

from mnc.datasets._io import iter_jsonl, write_jsonl, write_manifest
from mnc.datasets.ingest import INGEST_VERSION, _SnapshotParams, snapshots_from_df
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.snapshot import SnapshotRecord


def _make_snapshot(
    dataset_name: str = "vietmed-sum",
    source_split: str = "train",
    source_record_id: str = "0",
    payload: dict | None = None,
) -> SnapshotRecord:
    """Build a minimal SnapshotRecord for testing."""
    if payload is None:
        payload = {"transcript": "Bệnh nhân sốt cao.", "summary": "Tóm tắt."}
    return SnapshotRecord(
        dataset_name=dataset_name,
        source_split=source_split,
        source_record_id=source_record_id,
        payload=payload,
        source_format="parquet",
        source_path="data/train.train_whole-00000-of-00001.parquet",
        ingest_version=INGEST_VERSION,
        source_url="https://huggingface.co/datasets/leduckhai/VietMed-Sum",
        language="vi",
    )


class TestSnapshotSchema:
    """Schema validation for SnapshotRecord."""

    def test_valid_snapshot(self) -> None:
        rec = _make_snapshot()
        assert rec.dataset_name == "vietmed-sum"
        assert rec.source_split == "train"
        assert rec.ingest_version == INGEST_VERSION

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SnapshotRecord.model_validate(
                {
                    "dataset_name": "test",
                    "source_split": "train",
                    "source_record_id": "0",
                    "payload": {},
                    "source_format": "parquet",
                    "source_path": "x.parquet",
                    "ingest_version": "1.0",
                    "unknown_field": True,
                },
            )

    def test_optional_fields_default_none(self) -> None:
        rec = SnapshotRecord(
            dataset_name="test",
            source_split="train",
            source_record_id="0",
            payload={},
            source_format="parquet",
            source_path="x.parquet",
            ingest_version="1.0",
        )
        assert rec.source_url is None
        assert rec.language is None
        assert rec.raw_checksum is None


class TestSnapshotsFromDf:
    """Test snapshots_from_df core logic."""

    def test_vietmed_sum_happy_path(self) -> None:
        df = pl.DataFrame(
            {
                "transcript": ["Bệnh nhân sốt.", "Ho kéo dài."],
                "summary": ["Tóm tắt 1.", "Tóm tắt 2."],
            },
        )
        records, failed = snapshots_from_df(
            df,
            _SnapshotParams(
                dataset_name="vietmed-sum",
                split="train",
                source_format="parquet",
                source_path="data/train.train_whole-00000-of-00001.parquet",
                source_url="https://huggingface.co/datasets/leduckhai/VietMed-Sum",
                language="vi",
            ),
        )
        assert failed == 0
        assert len(records) == 2
        assert records[0].source_record_id == "0"
        assert records[1].source_record_id == "1"
        assert records[0].payload["transcript"] == "Bệnh nhân sốt."

    def test_vihealthqa_happy_path(self) -> None:
        df = pl.DataFrame(
            {
                "id": [1, 2],
                "question": ["Sốt cao không?", "Đau đầu?"],
                "answer": ["Có thể là viêm.", "Nên nghỉ ngơi."],
                "link": ["https://a.com", "https://b.com"],
            },
        )
        records, failed = snapshots_from_df(
            df,
            _SnapshotParams(
                dataset_name="vihealthqa",
                split="train",
                source_format="csv",
                source_path="train.csv",
                source_url="https://huggingface.co/datasets/tarudesu/ViHealthQA",
                language="vi",
                id_column="id",
            ),
        )
        assert failed == 0
        assert len(records) == 2
        assert records[0].source_record_id == "1"
        assert records[1].source_record_id == "2"

    def test_empty_df(self) -> None:
        df = pl.DataFrame({"transcript": [], "summary": []})
        records, failed = snapshots_from_df(
            df,
            _SnapshotParams(
                dataset_name="vietmed-sum",
                split="train",
                source_format="parquet",
                source_path="x.parquet",
            ),
        )
        assert len(records) == 0
        assert failed == 0


class TestManifestGeneration:
    """Test manifest file generation."""

    def test_manifest_written(self, tmp_path: Path) -> None:
        manifest = BronzeManifest(
            dataset="vietmed-sum",
            input_splits=["train"],
            output_splits=["train"],
            record_count_by_split={"train": 100},
            failed_count_by_split={"train": 0},
            created_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        manifest_path = tmp_path / "manifest.json"
        write_manifest(manifest, manifest_path)

        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["dataset"] == "vietmed-sum"
        assert data["record_count_by_split"]["train"] == 100

    def test_manifest_roundtrip(self, tmp_path: Path) -> None:
        original = BronzeManifest(
            dataset="vihealthqa",
            input_splits=["train", "val"],
            output_splits=["train", "val"],
            record_count_by_split={"train": 7000, "val": 1000},
            failed_count_by_split={"train": 2, "val": 0},
            created_at=datetime(2026, 4, 9, tzinfo=UTC),
        )
        path = tmp_path / "manifest.json"
        write_manifest(original, path)
        loaded = BronzeManifest.model_validate_json(
            path.read_text(encoding="utf-8"),
        )
        assert loaded == original


class TestJsonlRoundtrip:
    """Test JSONL write/read round-trip."""

    def test_write_and_read(self, tmp_path: Path) -> None:
        records = [_make_snapshot(source_record_id=str(i)) for i in range(5)]
        path = tmp_path / "train.jsonl"
        count = write_jsonl(records, path)
        assert count == 5

        loaded = list(iter_jsonl(path, SnapshotRecord))
        assert len(loaded) == 5
        for line_num, rec in loaded:
            assert isinstance(rec, SnapshotRecord)
            assert line_num >= 1
