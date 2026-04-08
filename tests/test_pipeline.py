"""Tests for the core ingestion pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from mnc.datasets.pipeline import ingest
from mnc.datasets.vietmed_sum import VietMedSumAdapter
from mnc.datasets.vihealthqa import ViHealthQAAdapter
from tests.conftest import FIXTURE_TS

_TOTAL = 5
_OK = 4
_FAIL = 1


def test_vietmed_full_ingestion(vietmed_dir: Path, tmp_path: Path) -> None:
    """End-to-end VietMed-Sum ingestion produces expected files and counts."""
    manifest = ingest(
        VietMedSumAdapter(),
        vietmed_dir,
        tmp_path,
        created_at=FIXTURE_TS,
    )

    assert manifest.dataset == "vietmed-sum"
    assert manifest.total_rows == _TOTAL
    assert manifest.successful_rows == _OK
    assert manifest.failed_rows == _FAIL
    assert "train_whole" in manifest.splits
    assert "dev_whole" in manifest.splits
    assert "test_whole" in manifest.splits


def test_vihealthqa_full_ingestion(vihealthqa_dir: Path, tmp_path: Path) -> None:
    """End-to-end ViHealthQA ingestion produces expected files and counts."""
    manifest = ingest(
        ViHealthQAAdapter(),
        vihealthqa_dir,
        tmp_path,
        created_at=FIXTURE_TS,
    )

    assert manifest.dataset == "vihealthqa"
    assert manifest.total_rows == _TOTAL
    assert manifest.successful_rows == _OK
    assert manifest.failed_rows == _FAIL
    assert "validation" in manifest.splits


def test_per_split_jsonl_created(vietmed_dir: Path, tmp_path: Path) -> None:
    """Each split gets its own JSONL file."""
    ingest(VietMedSumAdapter(), vietmed_dir, tmp_path, created_at=FIXTURE_TS)

    bronze = tmp_path / "vietmed-sum"
    assert (bronze / "train_whole.jsonl").exists()
    assert (bronze / "dev_whole.jsonl").exists()
    assert (bronze / "test_whole.jsonl").exists()


def test_manifest_file_created(vietmed_dir: Path, tmp_path: Path) -> None:
    """manifest.json is written to the bronze directory."""
    ingest(VietMedSumAdapter(), vietmed_dir, tmp_path, created_at=FIXTURE_TS)

    manifest_path = tmp_path / "vietmed-sum" / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["dataset"] == "vietmed-sum"
    assert isinstance(data["total_rows"], int)


def test_errors_file_created(vietmed_dir: Path, tmp_path: Path) -> None:
    """errors.jsonl contains entries for failed rows."""
    ingest(VietMedSumAdapter(), vietmed_dir, tmp_path, created_at=FIXTURE_TS)

    errors_path = tmp_path / "vietmed-sum" / "errors.jsonl"
    assert errors_path.exists()
    lines = errors_path.read_text().strip().split("\n")
    assert len(lines) == _FAIL
    entry = json.loads(lines[0])
    assert "error" in entry
    assert entry["split"] == "dev_whole"


def test_records_are_valid_jsonl(vietmed_dir: Path, tmp_path: Path) -> None:
    """Each line in JSONL output is valid JSON with required fields."""
    ingest(VietMedSumAdapter(), vietmed_dir, tmp_path, created_at=FIXTURE_TS)

    train_file = tmp_path / "vietmed-sum" / "train_whole.jsonl"
    for line in train_file.read_text().strip().split("\n"):
        rec = json.loads(line)
        assert "doc_id" in rec
        assert "source" in rec
        assert "raw_text" in rec
        assert "language" in rec


def test_deterministic_output(vietmed_dir: Path, tmp_path: Path) -> None:
    """Repeated runs with the same timestamp produce identical records."""
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    m1 = ingest(VietMedSumAdapter(), vietmed_dir, dir_a, created_at=FIXTURE_TS)
    m2 = ingest(VietMedSumAdapter(), vietmed_dir, dir_b, created_at=FIXTURE_TS)

    file_a = dir_a / "vietmed-sum" / "train_whole.jsonl"
    file_b = dir_b / "vietmed-sum" / "train_whole.jsonl"
    assert file_a.read_text() == file_b.read_text()
    assert m1.total_rows == m2.total_rows
    assert m1.successful_rows == m2.successful_rows


def test_pipeline_continues_after_row_failure(
    vihealthqa_dir: Path,
    tmp_path: Path,
) -> None:
    """Pipeline does not crash on individual row failures."""
    manifest = ingest(
        ViHealthQAAdapter(),
        vihealthqa_dir,
        tmp_path,
        created_at=FIXTURE_TS,
    )
    assert manifest.successful_rows > 0
    assert manifest.failed_rows > 0


def test_manifest_counts_match_output(
    vihealthqa_dir: Path,
    tmp_path: Path,
) -> None:
    """Manifest row counts match actual record/error line counts."""
    manifest = ingest(
        ViHealthQAAdapter(),
        vihealthqa_dir,
        tmp_path,
        created_at=FIXTURE_TS,
    )

    total_records = 0
    for split_path_str in manifest.splits.values():
        content = Path(split_path_str).read_text().strip()
        if content:
            total_records += len(content.split("\n"))

    content = Path(manifest.errors_path).read_text().strip()
    error_count = len(content.split("\n")) if content else 0

    assert total_records == manifest.successful_rows
    assert error_count == manifest.failed_rows
