"""Tests for DE-3 VietMed-Sum bronze parsing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mnc.datasets.de3_parse import parse_dataset
from mnc.schemas.document import DocumentRecord
from tests.conftest import FIXTURE_TS

if TYPE_CHECKING:
    from pathlib import Path


def _write_de2_vietmed(
    bronze_dir: Path,
    *,
    include_invalid: bool = False,
) -> Path:
    """Write minimal DE-2 VietMed-Sum JSONL files. Returns input root."""
    ds_dir = bronze_dir / "vietmed-sum"
    ds_dir.mkdir(parents=True, exist_ok=True)

    train_rec = DocumentRecord(
        doc_id="vietmed-sum:abc12345",
        source="vietmed-sum",
        language="vi",
        raw_text="bệnh nhân sốt cao ba ngày",
        source_record_id="abc12345",
        split="train_whole",
        payload={"transcript": "bệnh nhân sốt cao ba ngày", "summary": "sốt cao"},
        created_at=FIXTURE_TS,
    )
    dev_rec = DocumentRecord(
        doc_id="vietmed-sum:def67890",
        source="vietmed-sum",
        language="vi",
        raw_text="ho khan kéo dài",
        source_record_id="def67890",
        split="dev_whole",
        payload={"transcript": "ho khan kéo dài", "summary": "ho kéo dài"},
        created_at=FIXTURE_TS,
    )

    with (ds_dir / "train_whole.jsonl").open("w", encoding="utf-8") as f:
        f.write(train_rec.model_dump_json() + "\n")

    with (ds_dir / "dev_whole.jsonl").open("w", encoding="utf-8") as f:
        f.write(dev_rec.model_dump_json() + "\n")

    if include_invalid:
        bad_rec = train_rec.model_copy(
            update={
                "payload": {"transcript": "", "summary": "x"},
                "raw_text": "",
            },
        )
        with (ds_dir / "test_whole.jsonl").open("w", encoding="utf-8") as f:
            f.write(bad_rec.model_dump_json() + "\n")

    return bronze_dir


def test_happy_path_parse(tmp_path: Path) -> None:
    """Valid VietMed-Sum DE-2 records produce bronze docs with correct files."""
    input_dir = _write_de2_vietmed(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    manifest = parse_dataset(
        "vietmed-sum",
        input_dir,
        output_dir,
        created_at=FIXTURE_TS,
    )

    assert manifest.dataset == "vietmed-sum"
    assert manifest.successful_rows == 2
    assert manifest.failed_rows == 0
    assert "train_whole" in manifest.written_splits
    assert "dev_whole" in manifest.written_splits


def test_raw_text_equals_transcript(tmp_path: Path) -> None:
    """Bronze raw_text must equal payload['transcript']."""
    input_dir = _write_de2_vietmed(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vietmed-sum", input_dir, output_dir, created_at=FIXTURE_TS)

    train_file = output_dir / "vietmed-sum" / "train_whole.jsonl"
    line = train_file.read_text().strip()
    rec = json.loads(line)
    assert rec["raw_text"] == "bệnh nhân sốt cao ba ngày"
    assert rec["payload"]["transcript"] == "bệnh nhân sốt cao ba ngày"


def test_payload_preserved(tmp_path: Path) -> None:
    """Both transcript and summary must be preserved in payload."""
    input_dir = _write_de2_vietmed(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vietmed-sum", input_dir, output_dir, created_at=FIXTURE_TS)

    dev_file = output_dir / "vietmed-sum" / "dev_whole.jsonl"
    rec = json.loads(dev_file.read_text().strip())
    assert rec["payload"]["transcript"] == "ho khan kéo dài"
    assert rec["payload"]["summary"] == "ho kéo dài"


def test_doc_id_deterministic(tmp_path: Path) -> None:
    """doc_id is deterministic: 'vietmed-sum:{source_record_id}'."""
    input_dir = _write_de2_vietmed(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vietmed-sum", input_dir, output_dir, created_at=FIXTURE_TS)

    train_file = output_dir / "vietmed-sum" / "train_whole.jsonl"
    rec = json.loads(train_file.read_text().strip())
    assert rec["doc_id"] == f"vietmed-sum:{rec['source_record_id']}"


def test_split_names_unchanged(tmp_path: Path) -> None:
    """Split names from DE-2 are carried through unchanged."""
    input_dir = _write_de2_vietmed(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    manifest = parse_dataset(
        "vietmed-sum",
        input_dir,
        output_dir,
        created_at=FIXTURE_TS,
    )

    assert "train_whole" in manifest.written_splits
    assert "dev_whole" in manifest.written_splits
