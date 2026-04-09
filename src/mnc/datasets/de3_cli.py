"""CLI entrypoint for DE-3 bronze document parsing."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mnc.datasets.de3_parse import _DATASET_NAMES, parse_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s  %(message)s")


def main(argv: list[str] | None = None) -> None:
    """Run DE-3 bronze parse for a dataset."""
    parser = argparse.ArgumentParser(
        description="Parse DE-2 snapshot JSONL into bronze documents.",
    )
    parser.add_argument(
        "dataset",
        choices=sorted(_DATASET_NAMES),
        help="Dataset identifier",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/bronze"),
        help="Root DE-2 output directory (default: data/bronze)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bronze_docs"),
        help="Root bronze docs output directory (default: data/bronze_docs)",
    )
    args = parser.parse_args(argv)

    manifest = parse_dataset(args.dataset, args.input_dir, args.output_dir)

    out = sys.stdout
    out.write(
        f"dataset={manifest.dataset}  "
        f"total={manifest.total_rows}  "
        f"ok={manifest.successful_rows}  "
        f"failed={manifest.failed_rows}\n",
    )
    for split_name in manifest.written_splits:
        count = manifest.record_count_by_split[split_name]
        out.write(f"  {split_name}: {count} records\n")


if __name__ == "__main__":
    main()
