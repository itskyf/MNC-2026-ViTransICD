"""DE-3: Parse DE-2 snapshot JSONL into source-faithful bronze documents."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from mnc.schemas.document import DocumentRecord

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

log = logging.getLogger(__name__)

_EXCLUDED_JSONL = {"errors", "manifest"}

_DATASET_NAMES = {"vietmed-sum", "vihealthqa"}


class BronzeManifest(BaseModel):
    """Summary of a completed DE-3 bronze parse run."""

    dataset: str
    input_root: str
    output_root: str
    written_splits: list[str]
    record_count_by_split: dict[str, int]
    failed_count_by_split: dict[str, int]
    total_rows: int
    successful_rows: int
    failed_rows: int
    created_at: datetime


def _validate_vietmed_sum_payload(payload: dict[str, object]) -> str | None:
    """Return error message if payload is invalid for VietMed-Sum, else None."""
    transcript = payload.get("transcript")
    summary = payload.get("summary")

    if not isinstance(transcript, str) or not transcript.strip():
        return "missing or empty transcript"
    if not isinstance(summary, str):
        return "missing or invalid summary"
    return None


def _validate_vihealthqa_payload(payload: dict[str, object]) -> str | None:
    """Return error message if payload is invalid for ViHealthQA, else None."""
    question = payload.get("question")
    answer = payload.get("answer")

    if not isinstance(question, str) or not question.strip():
        return "missing or empty question"
    if not isinstance(answer, str) or not answer.strip():
        return "missing or empty answer"
    return None


_PAYLOAD_VALIDATORS: dict[str, Callable[[dict[str, object]], str | None]] = {
    "vietmed-sum": _validate_vietmed_sum_payload,
    "vihealthqa": _validate_vihealthqa_payload,
}


def _build_doc_id(dataset: str, record: DocumentRecord) -> str:
    """Build deterministic doc_id per dataset-specific rules."""
    source_id = record.source_record_id or ""
    if dataset == "vietmed-sum":
        return f"vietmed-sum:{source_id}"
    # vihealthqa
    split = record.split or ""
    return f"vihealthqa:{split}:{source_id}"


def _discover_input_splits(input_dir: Path, dataset: str) -> list[tuple[str, Path]]:
    """Return (split_name, path) for each DE-2 JSONL split file."""
    dataset_dir = input_dir / dataset
    if not dataset_dir.is_dir():
        msg = f"DE-2 output directory not found: {dataset_dir}"
        raise FileNotFoundError(msg)

    splits: list[tuple[str, Path]] = []
    for jsonl_file in sorted(dataset_dir.glob("*.jsonl")):
        if jsonl_file.stem in _EXCLUDED_JSONL:
            continue
        splits.append((jsonl_file.stem, jsonl_file))
    return splits


def _write_jsonl(path: Path, records: list[DocumentRecord]) -> None:
    """Write validated DocumentRecords as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def _write_errors(path: Path, entries: list[dict[str, str]]) -> None:
    """Write error entries as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def parse_dataset(
    dataset: str,
    input_dir: Path,
    output_dir: Path,
    *,
    created_at: datetime | None = None,
) -> BronzeManifest:
    """Parse DE-2 snapshot JSONL into bronze documents.

    Args:
        dataset: Canonical dataset name (e.g. "vietmed-sum" or "vihealthqa").
        input_dir: Root directory containing DE-2 output (e.g. data/bronze).
        output_dir: Root directory for bronze doc output (e.g. data/bronze_docs).
        created_at: Injected timestamp for deterministic testing.

    Returns:
        BronzeManifest summarizing the parse run.
    """
    if dataset not in _DATASET_NAMES:
        msg = f"unknown dataset: {dataset}"
        raise ValueError(msg)

    ts = created_at or datetime.now(UTC)
    validate_payload = _PAYLOAD_VALIDATORS[dataset]

    splits = _discover_input_splits(input_dir, dataset)
    dataset_out = output_dir / dataset
    dataset_out.mkdir(parents=True, exist_ok=True)

    all_errors: list[dict[str, str]] = []
    total = 0
    successful = 0
    written_splits: list[str] = []
    record_count_by_split: dict[str, int] = {}
    failed_count_by_split: dict[str, int] = {}

    for split_name, split_path in splits:
        records: list[DocumentRecord] = []
        split_errors = 0

        with split_path.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                total += 1

                # Parse DE-2 output as DocumentRecord
                try:
                    source_rec = DocumentRecord.model_validate_json(line)
                except ValidationError as exc:
                    all_errors.append(
                        {
                            "dataset": dataset,
                            "split": split_name,
                            "source_record_id": f"line_{line_no}",
                            "error": f"schema validation failed: {exc}",
                        },
                    )
                    split_errors += 1
                    continue

                # Validate dataset-specific payload fields
                payload = source_rec.payload or {}
                payload_error = validate_payload(payload)
                if payload_error is not None:
                    all_errors.append(
                        {
                            "dataset": dataset,
                            "split": split_name,
                            "source_record_id": source_rec.source_record_id
                            or f"line_{line_no}",
                            "error": payload_error,
                        },
                    )
                    split_errors += 1
                    continue

                # Build new bronze DocumentRecord
                doc_id = _build_doc_id(dataset, source_rec)
                bronze_rec = DocumentRecord(
                    doc_id=doc_id,
                    source=source_rec.source,
                    language=source_rec.language,
                    raw_text=source_rec.raw_text,
                    source_record_id=source_rec.source_record_id,
                    split=source_rec.split,
                    payload=source_rec.payload,
                    created_at=ts,
                )
                records.append(bronze_rec)
                successful += 1

        records_file = dataset_out / f"{split_name}.jsonl"
        _write_jsonl(records_file, records)
        written_splits.append(split_name)
        record_count_by_split[split_name] = len(records)
        failed_count_by_split[split_name] = split_errors
        log.info(
            "dataset=%s split=%s rows=%d ok=%d errors=%d",
            dataset,
            split_name,
            len(records) + split_errors,
            len(records),
            split_errors,
        )

    errors_file = dataset_out / "errors.jsonl"
    _write_errors(errors_file, all_errors)

    manifest = BronzeManifest(
        dataset=dataset,
        input_root=str(input_dir),
        output_root=str(output_dir),
        written_splits=written_splits,
        record_count_by_split=record_count_by_split,
        failed_count_by_split=failed_count_by_split,
        total_rows=total,
        successful_rows=successful,
        failed_rows=total - successful,
        created_at=ts,
    )

    manifest_file = dataset_out / "manifest.json"
    manifest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    log.info(
        "dataset=%s total=%d ok=%d failed=%d",
        dataset,
        total,
        successful,
        total - successful,
    )
    return manifest
