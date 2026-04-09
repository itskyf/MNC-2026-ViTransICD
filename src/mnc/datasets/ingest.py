"""Snapshot public datasets into immutable, source-faithful bronze records.

Usage::

    uv run -m mnc.datasets.ingest --dataset vietmed-sum
    uv run -m mnc.datasets.ingest --dataset vihealthqa
    uv run -m mnc.datasets.ingest --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl
from huggingface_hub import hf_hub_download
from pydantic import ValidationError

from mnc.datasets._io import now_utc, write_jsonl, write_manifest
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.snapshot import SnapshotRecord

INGEST_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Dataset file maps: bronze split -> HF repo-relative path
# ---------------------------------------------------------------------------

_VIETMED_SUM_FILES: dict[str, str] = {
    "train": "data/train.train_whole-00000-of-00001.parquet",
    "dev": "data/dev.dev_whole-00000-of-00001.parquet",
    "test": "data/test.test_whole-00000-of-00001.parquet",
}
_VIETMED_SUM_HUB = "leduckhai/VietMed-Sum"
_VIETMED_SUM_URL = "https://huggingface.co/datasets/leduckhai/VietMed-Sum"

_VIHEALTHQA_FILES: dict[str, str] = {
    "train": "train.csv",
    "val": "val.csv",
    "test": "test.csv",
}
_VIHEALTHQA_HUB = "tarudesu/ViHealthQA"
_VIHEALTHQA_URL = "https://huggingface.co/datasets/tarudesu/ViHealthQA"


# ---------------------------------------------------------------------------
# Core snapshot logic (no network)
# ---------------------------------------------------------------------------


class IngestConfig:
    """Configuration for one dataset's ingest pipeline."""

    __slots__ = ("files", "hub_id", "id_column", "reader", "source_format", "url")

    def __init__(  # noqa: PLR0913
        self,
        hub_id: str,
        files: dict[str, str],
        url: str,
        source_format: str,
        reader: Any,  # noqa: ANN401
        id_column: str | None = None,
    ) -> None:
        """Initialize ingest configuration."""
        self.hub_id = hub_id
        self.files = files
        self.url = url
        self.source_format = source_format
        self.reader = reader
        self.id_column = id_column


def _ingest_config() -> dict[str, IngestConfig]:
    """Return dataset ingest configurations."""
    return {
        "vietmed-sum": IngestConfig(
            hub_id=_VIETMED_SUM_HUB,
            files=_VIETMED_SUM_FILES,
            url=_VIETMED_SUM_URL,
            source_format="parquet",
            reader=pl.read_parquet,
        ),
        "vihealthqa": IngestConfig(
            hub_id=_VIHEALTHQA_HUB,
            files=_VIHEALTHQA_FILES,
            url=_VIHEALTHQA_URL,
            source_format="csv",
            reader=pl.read_csv,
            id_column="id",
        ),
    }


def snapshots_from_df(  # noqa: PLR0913
    df: pl.DataFrame,
    *,
    dataset_name: str,
    split: str,
    source_format: str,
    source_path: str,
    source_url: str | None = None,
    language: str | None = None,
    id_column: str | None = None,
) -> tuple[list[SnapshotRecord], int]:
    """Create snapshot records from a polars DataFrame.

    Returns ``(valid_records, failed_count)``.
    """
    records: list[SnapshotRecord] = []
    failed = 0
    for idx in range(df.height):
        row = df.row(idx, named=True)
        payload = dict(row)
        source_record_id = str(row[id_column]) if id_column else str(idx)
        try:
            records.append(
                SnapshotRecord(
                    dataset_name=dataset_name,
                    source_split=split,
                    source_record_id=source_record_id,
                    payload=payload,  # type: ignore[arg-type]
                    source_format=source_format,
                    source_path=source_path,
                    ingest_version=INGEST_VERSION,
                    source_url=source_url,
                    language=language,
                ),
            )
        except ValidationError:
            failed += 1
    return records, failed


# ---------------------------------------------------------------------------
# Per-dataset ingesters (download + snapshot)
# ---------------------------------------------------------------------------


def ingest_dataset(dataset_name: str, bronze_dir: Path) -> BronzeManifest:
    """Download and snapshot a dataset identified by *dataset_name*."""
    configs = _ingest_config()
    if dataset_name not in configs:
        msg = f"Unknown dataset: {dataset_name}"
        raise ValueError(msg)

    cfg = configs[dataset_name]
    out = bronze_dir / dataset_name / "snapshots"
    out.mkdir(parents=True, exist_ok=True)

    record_counts: dict[str, int] = {}
    failed_counts: dict[str, int] = {}

    for split, hf_path in cfg.files.items():
        local = hf_hub_download(cfg.hub_id, hf_path, repo_type="dataset")
        df = cfg.reader(local)
        records, failed = snapshots_from_df(
            df,
            dataset_name=dataset_name,
            split=split,
            source_format=cfg.source_format,
            source_path=hf_path,
            source_url=cfg.url,
            language="vi",
            id_column=cfg.id_column,
        )
        write_jsonl(records, out / f"{split}.jsonl")
        record_counts[split] = len(records)
        failed_counts[split] = failed

    manifest = BronzeManifest(
        dataset=dataset_name,
        input_splits=list(cfg.files),
        output_splits=list(record_counts),
        record_count_by_split=record_counts,
        failed_count_by_split=failed_counts,
        created_at=now_utc(),
    )
    write_manifest(manifest, out / "manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DATASETS = ("vietmed-sum", "vihealthqa")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for dataset ingestion."""
    parser = argparse.ArgumentParser(
        description="Snapshot public datasets to bronze",
    )
    parser.add_argument(
        "--dataset",
        choices=_DATASETS,
        help="Dataset to ingest",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="ingest_all",
        help="Ingest all datasets",
    )
    parser.add_argument(
        "--output",
        default="data/bronze",
        type=Path,
        help="Bronze output root",
    )
    args = parser.parse_args(argv)

    if not args.dataset and not args.ingest_all:
        parser.error("Specify --dataset <name> or --all")

    names = list(_DATASETS) if args.ingest_all else [args.dataset]
    for name in names:
        print(f"Ingesting {name}...", file=sys.stderr)  # noqa: T201
        manifest = ingest_dataset(name, args.output)
        print(  # noqa: T201
            f"  {manifest.record_count_by_split}",
            file=sys.stderr,
        )
        total = sum(manifest.record_count_by_split.values())
        print(f"  Total: {total} records", file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    main()
