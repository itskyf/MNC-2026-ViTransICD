"""Tests for DE-3 ViHealthQA bronze parsing."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from mnc.datasets.de3_parse import parse_dataset
from mnc.schemas.document import DocumentRecord
from tests.conftest import FIXTURE_TS

if TYPE_CHECKING:
    from pathlib import Path


def _write_de2_vihealthqa(
    bronze_dir: Path,
    *,
    include_invalid: bool = False,
) -> Path:
    """Write minimal DE-2 ViHealthQA JSONL files. Returns input root."""
    ds_dir = bronze_dir / "vihealthqa"
    ds_dir.mkdir(parents=True, exist_ok=True)

    train_rec = DocumentRecord(
        doc_id="vihealthqa:train:1",
        source="vihealthqa",
        language="vi",
        raw_text="Tiêm vaccine có an toàn không?\nCó, vaccine đã qua kiểm định.",
        source_record_id="1",
        split="train",
        payload={
            "question": "Tiêm vaccine có an toàn không?",
            "answer": "Có, vaccine đã qua kiểm định.",
            "link": "https://example.com/1",
        },
        created_at=FIXTURE_TS,
    )
    val_rec = DocumentRecord(
        doc_id="vihealthqa:validation:10",
        source="vihealthqa",
        language="vi",
        raw_text="Uống thuốc khi đói được không?\nNên ăn no trước khi uống.",
        source_record_id="10",
        split="validation",
        payload={
            "question": "Uống thuốc khi đói được không?",
            "answer": "Nên ăn no trước khi uống.",
            "link": "https://example.com/3",
        },
        created_at=FIXTURE_TS,
    )

    with (ds_dir / "train.jsonl").open("w", encoding="utf-8") as f:
        f.write(train_rec.model_dump_json() + "\n")

    with (ds_dir / "validation.jsonl").open("w", encoding="utf-8") as f:
        f.write(val_rec.model_dump_json() + "\n")

    if include_invalid:
        bad_rec = train_rec.model_copy(
            update={
                "payload": {"question": "", "answer": "x", "link": ""},
                "raw_text": "\nx",
            },
        )
        with (ds_dir / "test.jsonl").open("w", encoding="utf-8") as f:
            f.write(bad_rec.model_dump_json() + "\n")

    return bronze_dir


def test_happy_path_parse(tmp_path: Path) -> None:
    """Valid ViHealthQA DE-2 records produce bronze docs with correct files."""
    input_dir = _write_de2_vihealthqa(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    manifest = parse_dataset(
        "vihealthqa",
        input_dir,
        output_dir,
        created_at=FIXTURE_TS,
    )

    assert manifest.dataset == "vihealthqa"
    assert manifest.successful_rows == 2
    assert manifest.failed_rows == 0
    assert "train" in manifest.written_splits
    assert "validation" in manifest.written_splits


def test_raw_text_is_question_newline_answer(tmp_path: Path) -> None:
    """raw_text must equal question + newline + answer."""
    input_dir = _write_de2_vihealthqa(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vihealthqa", input_dir, output_dir, created_at=FIXTURE_TS)

    train_file = output_dir / "vihealthqa" / "train.jsonl"
    rec = json.loads(train_file.read_text().strip())
    assert rec["raw_text"] == (
        "Tiêm vaccine có an toàn không?\nCó, vaccine đã qua kiểm định."
    )


def test_payload_preserved(tmp_path: Path) -> None:
    """question, answer, and link must be preserved in payload."""
    input_dir = _write_de2_vihealthqa(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vihealthqa", input_dir, output_dir, created_at=FIXTURE_TS)

    val_file = output_dir / "vihealthqa" / "validation.jsonl"
    rec = json.loads(val_file.read_text().strip())
    assert rec["payload"]["question"] == "Uống thuốc khi đói được không?"
    assert rec["payload"]["answer"] == "Nên ăn no trước khi uống."
    assert rec["payload"]["link"] == "https://example.com/3"


def test_doc_id_deterministic(tmp_path: Path) -> None:
    """doc_id is deterministic: 'vihealthqa:{split}:{source_record_id}'."""
    input_dir = _write_de2_vihealthqa(tmp_path / "bronze")
    output_dir = tmp_path / "bronze_docs"

    parse_dataset("vihealthqa", input_dir, output_dir, created_at=FIXTURE_TS)

    val_file = output_dir / "vihealthqa" / "validation.jsonl"
    rec = json.loads(val_file.read_text().strip())
    assert rec["doc_id"] == f"vihealthqa:validation:{rec['source_record_id']}"
