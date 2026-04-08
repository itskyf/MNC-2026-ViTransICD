"""Core ingestion pipeline: orchestrate adapters, write bronze outputs."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, ValidationError

from mnc.datasets.adapter import RawSample

if TYPE_CHECKING:
    from pathlib import Path

    from mnc.datasets.adapter import DatasetAdapter, RawSample
    from mnc.schemas.document import DocumentRecord

log = logging.getLogger(__name__)


class Manifest(BaseModel):
    """Summary of a completed ingestion run."""

    dataset: str
    total_rows: int
    successful_rows: int
    failed_rows: int
    splits: dict[str, str]
    errors_path: str
    created_at: datetime


def _error_entry(
    dataset: str,
    split: str,
    raw_sample: RawSample,
    message: str,
) -> dict[str, str]:
    """Build a structured error log entry."""
    source_id = raw_sample.get("id")
    return {
        "dataset": dataset,
        "split": split,
        "source_record_id": str(source_id) if source_id is not None else "",
        "error": message,
    }


def _write_jsonl(path: Path, records: list[DocumentRecord]) -> None:
    """Write validated DocumentRecords as JSONL."""
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def _write_jsonl_raw(path: Path, entries: list[dict[str, str]]) -> None:
    """Write raw dicts as JSONL (for error logs)."""
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def ingest(
    adapter: DatasetAdapter,
    input_path: Path,
    output_dir: Path,
    *,
    created_at: datetime | None = None,
) -> Manifest:
    """Run ingestion for a single dataset adapter.

    Args:
        adapter: Dataset-specific adapter implementing the ingestion protocol.
        input_path: Directory or file containing raw source data.
        output_dir: Root output directory (bronze layer parent).
        created_at: Injected timestamp for deterministic testing.

    Returns:
        Manifest summarizing the ingestion run.
    """
    ts = created_at or datetime.now(UTC)
    dataset = adapter.dataset_name()
    bronze_dir = output_dir / dataset
    bronze_dir.mkdir(parents=True, exist_ok=True)

    splits = adapter.discover_splits(input_path)
    all_errors: list[dict[str, str]] = []
    total = 0
    successful = 0
    split_paths: dict[str, str] = {}

    for split_name, split_path in splits:
        records_file = bronze_dir / f"{split_name}.jsonl"
        split_records: list[DocumentRecord] = []

        for raw_sample in adapter.iter_raw_samples(split_path):
            total += 1

            if not adapter.validate_raw_sample(raw_sample):
                all_errors.append(
                    _error_entry(
                        dataset,
                        split_name,
                        raw_sample,
                        "validation failed",
                    ),
                )
                continue

            try:
                record = adapter.to_de1_record(
                    raw_sample,
                    split=split_name,
                    created_at=ts,
                )
            except (ValueError, KeyError, TypeError, ValidationError) as exc:
                all_errors.append(
                    _error_entry(dataset, split_name, raw_sample, str(exc)),
                )
                continue

            split_records.append(record)
            successful += 1

        _write_jsonl(records_file, split_records)
        split_paths[split_name] = str(records_file)
        log.info(
            "split=%s  rows=%d  ok=%d",
            split_name,
            len(split_records) + sum(1 for e in all_errors if e["split"] == split_name),
            len(split_records),
        )

    errors_file = bronze_dir / "errors.jsonl"
    _write_jsonl_raw(errors_file, all_errors)

    manifest = Manifest(
        dataset=dataset,
        total_rows=total,
        successful_rows=successful,
        failed_rows=total - successful,
        splits=split_paths,
        errors_path=str(errors_file),
        created_at=ts,
    )

    manifest_file = bronze_dir / "manifest.json"
    manifest_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    log.info(
        "dataset=%s  total=%d  ok=%d  failed=%d",
        dataset,
        total,
        successful,
        total - successful,
    )
    return manifest
