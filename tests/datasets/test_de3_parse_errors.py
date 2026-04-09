"""Tests for DE-3 error handling in bronze parsing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mnc.datasets.de3_parse import parse_dataset
from mnc.schemas.document import DocumentRecord
from tests.conftest import FIXTURE_TS

if TYPE_CHECKING:
    from pathlib import Path


def _write_de2_with_bad_record(bronze_dir: Path) -> Path:
    """Write DE-2 data with one invalid record. Returns input root."""
    ds_dir = bronze_dir / "vietmed-sum"
    ds_dir.mkdir(parents=True, exist_ok=True)

    good_rec = DocumentRecord(
        doc_id="vietmed-sum:good123",
        source="vietmed-sum",
        language="vi",
        raw_text="transcript nội dung",
        source_record_id="good123",
        split="train_whole",
        payload={"transcript": "transcript nội dung", "summary": "tóm tắt"},
        created_at=FIXTURE_TS,
    )
    # Bad record: empty transcript in payload
    bad_rec = DocumentRecord(
        doc_id="vietmed-sum:bad456",
        source="vietmed-sum",
        language="vi",
        raw_text="",
        source_record_id="bad456",
        split="train_whole",
        payload={"transcript": "", "summary": "tóm tắt"},
        created_at=FIXTURE_TS,
    )

    with (ds_dir / "train_whole.jsonl").open("w", encoding="utf-8") as f:
        f.write(good_rec.model_dump_json() + "\n")
        f.write(bad_rec.model_dump_json() + "\n")

    return bronze_dir


def test_invalid_record_excluded_from_output(tmp_path: Path) -> None:
    """Invalid records are excluded from output JSONL."""
    input_dir = _write_de2_with_bad_record(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    manifest = parse_dataset(
        "vietmed-sum",
        input_dir,
        output_dir,
        created_at=FIXTURE_TS,
    )

    assert manifest.successful_rows == 1
    assert manifest.failed_rows == 1

    train_file = output_dir / "vietmed-sum" / "train_whole.jsonl"
    lines = [
        line for line in train_file.read_text().strip().split("\n") if line.strip()
    ]
    assert len(lines) == 1


def test_invalid_record_written_to_errors(tmp_path: Path) -> None:
    """Invalid records appear in errors.jsonl with context."""
    input_dir = _write_de2_with_bad_record(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vietmed-sum", input_dir, output_dir, created_at=FIXTURE_TS)

    errors_file = output_dir / "vietmed-sum" / "errors.jsonl"
    assert errors_file.exists()
    lines = [
        line for line in errors_file.read_text().strip().split("\n") if line.strip()
    ]
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["dataset"] == "vietmed-sum"
    assert entry["split"] == "train_whole"
    assert entry["source_record_id"] == "bad456"
    assert "transcript" in entry["error"]


def test_malformed_json_produces_error(tmp_path: Path) -> None:
    """A non-JSON line in DE-2 output is logged as error."""
    ds_dir = tmp_path / "bronze" / "vietmed-sum"
    ds_dir.mkdir(parents=True, exist_ok=True)

    good_rec = DocumentRecord(
        doc_id="vietmed-sum:x1",
        source="vietmed-sum",
        language="vi",
        raw_text="nội dung",
        source_record_id="x1",
        split="train_whole",
        payload={"transcript": "nội dung", "summary": "tóm tắt"},
        created_at=FIXTURE_TS,
    )

    with (ds_dir / "train_whole.jsonl").open("w", encoding="utf-8") as f:
        f.write(good_rec.model_dump_json() + "\n")
        f.write("THIS IS NOT JSON\n")

    output_dir = tmp_path / "bronze_docs"
    manifest = parse_dataset(
        "vietmed-sum",
        tmp_path / "bronze",
        output_dir,
        created_at=FIXTURE_TS,
    )

    assert manifest.successful_rows == 1
    assert manifest.failed_rows == 1

    errors_file = output_dir / "vietmed-sum" / "errors.jsonl"
    entry = json.loads(errors_file.read_text().strip())
    assert "schema validation failed" in entry["error"]
