"""Parse bronze snapshot records into document-level records.

Usage::

    uv run -m mnc.datasets.parse --dataset vietmed-sum
    uv run -m mnc.datasets.parse --dataset vihealthqa
    uv run -m mnc.datasets.parse --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from mnc.datasets._io import now_utc, write_jsonl, write_manifest
from mnc.schemas.document import DocumentRecord
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.snapshot import SnapshotRecord

# ---------------------------------------------------------------------------
# Per-dataset document constructors
# ---------------------------------------------------------------------------


def _vietmed_sum_doc(snapshot: SnapshotRecord) -> DocumentRecord:
    """Convert a VietMed-Sum snapshot to a DocumentRecord."""
    transcript = snapshot.payload.get("transcript")
    summary = snapshot.payload.get("summary")

    if not isinstance(transcript, str) or not transcript:
        msg = "transcript must be a non-empty string"
        raise ValueError(msg)
    if not isinstance(summary, str):
        msg = "summary must be a string"
        raise TypeError(msg)

    return DocumentRecord(
        doc_id=f"vietmed-sum:{snapshot.source_record_id}",
        source="vietmed-sum",
        language="vi",
        raw_text=transcript,
        source_record_id=snapshot.source_record_id,
        split=snapshot.source_split,
        payload={"transcript": transcript, "summary": summary},
        created_at=now_utc(),
    )


def _vihealthqa_doc(snapshot: SnapshotRecord) -> DocumentRecord:
    """Convert a ViHealthQA snapshot to a DocumentRecord."""
    question = snapshot.payload.get("question")
    answer = snapshot.payload.get("answer")
    link = snapshot.payload.get("link")

    if not isinstance(question, str) or not question:
        msg = "question must be a non-empty string"
        raise ValueError(msg)
    if not isinstance(answer, str) or not answer:
        msg = "answer must be a non-empty string"
        raise ValueError(msg)

    return DocumentRecord(
        doc_id=f"vihealthqa:{snapshot.source_split}:{snapshot.source_record_id}",
        source="vihealthqa",
        language="vi",
        raw_text=f"{question}\n{answer}",
        source_record_id=snapshot.source_record_id,
        split=snapshot.source_split,
        payload={"question": question, "answer": answer, "link": link},
        created_at=now_utc(),
    )


# ---------------------------------------------------------------------------
# Core parsing pipeline
# ---------------------------------------------------------------------------

_PARSERS: dict[str, object] = {
    "vietmed-sum": _vietmed_sum_doc,
    "vihealthqa": _vihealthqa_doc,
}


def parse_dataset(dataset_name: str, bronze_dir: Path) -> BronzeManifest:
    """Parse all snapshot splits for *dataset_name* into documents."""
    if dataset_name not in _PARSERS:
        msg = f"Unknown dataset: {dataset_name}"
        raise ValueError(msg)

    parser = _PARSERS[dataset_name]  # type: ignore[index]
    snapshot_dir = bronze_dir / dataset_name / "snapshots"
    document_dir = bronze_dir / dataset_name / "documents"
    document_dir.mkdir(parents=True, exist_ok=True)

    record_counts: dict[str, int] = {}
    failed_counts: dict[str, int] = {}
    input_splits: list[str] = []
    all_errors: list[dict] = []

    for snapshot_file in sorted(snapshot_dir.glob("*.jsonl")):
        split = snapshot_file.stem
        input_splits.append(split)

        docs: list[DocumentRecord] = []
        failed = 0

        with snapshot_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    snapshot = SnapshotRecord.model_validate_json(stripped)
                    doc = parser(snapshot)  # type: ignore[operator]
                    docs.append(doc)
                except (ValueError, TypeError, ValidationError) as exc:
                    failed += 1
                    all_errors.append(
                        {
                            "line": line_num,
                            "split": split,
                            "error": str(exc),
                        },
                    )

        write_jsonl(docs, document_dir / f"{split}.jsonl")
        record_counts[split] = len(docs)
        failed_counts[split] = failed

    # Write error log
    if all_errors:
        with (document_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
            for err in all_errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

    manifest = BronzeManifest(
        dataset=dataset_name,
        input_splits=input_splits,
        output_splits=list(record_counts),
        record_count_by_split=record_counts,
        failed_count_by_split=failed_counts,
        created_at=now_utc(),
    )
    write_manifest(manifest, document_dir / "manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DATASETS = ("vietmed-sum", "vihealthqa")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for document parsing."""
    parser = argparse.ArgumentParser(
        description="Parse bronze snapshots to documents",
    )
    parser.add_argument(
        "--dataset",
        choices=_DATASETS,
        help="Dataset to parse",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="parse_all",
        help="Parse all datasets",
    )
    parser.add_argument(
        "--bronze-dir",
        default="data/bronze",
        type=Path,
        help="Bronze root directory",
    )
    args = parser.parse_args(argv)

    if not args.dataset and not args.parse_all:
        parser.error("Specify --dataset <name> or --all")

    names = list(_DATASETS) if args.parse_all else [args.dataset]
    for name in names:
        print(f"Parsing {name}...", file=sys.stderr)  # noqa: T201
        manifest = parse_dataset(name, args.bronze_dir)
        print(  # noqa: T201
            f"  {manifest.record_count_by_split}",
            file=sys.stderr,
        )
        total = sum(manifest.record_count_by_split.values())
        print(f"  Total: {total} documents", file=sys.stderr)  # noqa: T201


if __name__ == "__main__":
    main()
