"""CLI entrypoint for dataset ingestion."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from mnc.datasets import ADAPTERS, ingest

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s  %(message)s")


def main(argv: list[str] | None = None) -> None:
    """Run ingestion for a dataset."""
    parser = argparse.ArgumentParser(
        description="Ingest a public dataset into bronze-layer JSONL.",
    )
    parser.add_argument("dataset", choices=sorted(ADAPTERS), help="Dataset identifier")
    parser.add_argument(
        "input_path",
        type=Path,
        help="Directory with raw source files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/bronze"),
        help="Root output directory (default: data/bronze)",
    )
    args = parser.parse_args(argv)

    adapter_cls = ADAPTERS[args.dataset]
    adapter = adapter_cls()
    manifest = ingest(adapter, args.input_path, args.output_dir)

    out = sys.stdout
    out.write(
        f"dataset={manifest.dataset}  "
        f"total={manifest.total_rows}  "
        f"ok={manifest.successful_rows}  "
        f"failed={manifest.failed_rows}\n",
    )
    for split_name, path in manifest.splits.items():
        out.write(f"  {split_name}: {path}\n")


if __name__ == "__main__":
    main()
