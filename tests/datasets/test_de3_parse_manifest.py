"""Tests for DE-3 manifest generation."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mnc.datasets.de3_parse import parse_dataset
from mnc.schemas.document import DocumentRecord
from tests.conftest import FIXTURE_TS

if TYPE_CHECKING:
    from pathlib import Path


def _write_de2_two_splits(bronze_dir: Path) -> Path:
    """Write DE-2 data with two splits. Returns input root."""
    ds_dir = bronze_dir / "vietmed-sum"
    ds_dir.mkdir(parents=True, exist_ok=True)

    train_rec = DocumentRecord(
        doc_id="vietmed-sum:t1",
        source="vietmed-sum",
        language="vi",
        raw_text="transcript A",
        source_record_id="t1",
        split="train_whole",
        payload={"transcript": "transcript A", "summary": "summary A"},
        created_at=FIXTURE_TS,
    )
    dev_rec = DocumentRecord(
        doc_id="vietmed-sum:d1",
        source="vietmed-sum",
        language="vi",
        raw_text="transcript B",
        source_record_id="d1",
        split="dev_whole",
        payload={"transcript": "transcript B", "summary": "summary B"},
        created_at=FIXTURE_TS,
    )

    with (ds_dir / "train_whole.jsonl").open("w", encoding="utf-8") as f:
        f.write(train_rec.model_dump_json() + "\n")

    with (ds_dir / "dev_whole.jsonl").open("w", encoding="utf-8") as f:
        f.write(dev_rec.model_dump_json() + "\n")

    return bronze_dir


def test_manifest_file_created(tmp_path: Path) -> None:
    """manifest.json is written to the bronze docs directory."""
    input_dir = _write_de2_two_splits(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vietmed-sum", input_dir, output_dir, created_at=FIXTURE_TS)

    manifest_path = output_dir / "vietmed-sum" / "manifest.json"
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text())
    assert data["dataset"] == "vietmed-sum"


def test_manifest_counts_match_output(tmp_path: Path) -> None:
    """Manifest row counts match actual record and error line counts."""
    input_dir = _write_de2_two_splits(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    manifest = parse_dataset(
        "vietmed-sum",
        input_dir,
        output_dir,
        created_at=FIXTURE_TS,
    )

    total_records = sum(
        len(
            [
                line
                for line in (output_dir / "vietmed-sum" / f"{s}.jsonl")
                .read_text()
                .strip()
                .split("\n")
                if line.strip()
            ],
        )
        for s in manifest.written_splits
    )

    errors_path = output_dir / "vietmed-sum" / "errors.jsonl"
    content = errors_path.read_text().strip()
    error_count = (
        len([line for line in content.split("\n") if line.strip()]) if content else 0
    )

    assert total_records == manifest.successful_rows
    assert error_count == manifest.failed_rows
    assert manifest.total_rows == manifest.successful_rows + manifest.failed_rows


def test_manifest_record_count_by_split(tmp_path: Path) -> None:
    """record_count_by_split has correct per-split counts."""
    input_dir = _write_de2_two_splits(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    manifest = parse_dataset(
        "vietmed-sum",
        input_dir,
        output_dir,
        created_at=FIXTURE_TS,
    )

    assert manifest.record_count_by_split["train_whole"] == 1
    assert manifest.record_count_by_split["dev_whole"] == 1
    assert manifest.record_count_by_split == manifest.failed_count_by_split | {
        "train_whole": 1,
        "dev_whole": 1,
    }


def test_manifest_required_fields(tmp_path: Path) -> None:
    """Manifest contains all required fields from the spec."""
    input_dir = _write_de2_two_splits(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vietmed-sum", input_dir, output_dir, created_at=FIXTURE_TS)

    manifest_path = output_dir / "vietmed-sum" / "manifest.json"
    data = json.loads(manifest_path.read_text())

    required_fields = [
        "dataset",
        "input_root",
        "output_root",
        "written_splits",
        "record_count_by_split",
        "failed_count_by_split",
        "total_rows",
        "successful_rows",
        "failed_rows",
        "created_at",
    ]
    for field in required_fields:
        assert field in data, f"missing field: {field}"


def test_manifest_errors_file_created(tmp_path: Path) -> None:
    """errors.jsonl is always written, even with no errors."""
    input_dir = _write_de2_two_splits(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vietmed-sum", input_dir, output_dir, created_at=FIXTURE_TS)

    errors_path = output_dir / "vietmed-sum" / "errors.jsonl"
    assert errors_path.exists()
