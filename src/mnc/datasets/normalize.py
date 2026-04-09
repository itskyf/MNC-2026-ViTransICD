"""DC-1: Normalize text, segment, and extract mentions.

Usage::

    uv run -m mnc.datasets.normalize --dataset vietmed-sum
    uv run -m mnc.datasets.normalize --dataset vihealthqa
    uv run -m mnc.datasets.normalize --all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import underthesea
from pydantic import ValidationError

from mnc.datasets._io import now_utc, write_jsonl, write_manifest
from mnc.datasets._mentions import extract_mentions
from mnc.datasets._text import build_retrieval_text, normalize_document_text
from mnc.schemas.document import DocumentRecord
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.sentence import SentenceSpanRecord

if TYPE_CHECKING:
    from datetime import datetime

    from mnc.schemas.mention import MentionRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence segmentation
# ---------------------------------------------------------------------------


def segment_sentences(
    normalized_text: str,
    doc_id: str,
    created_at: datetime,
) -> list[SentenceSpanRecord]:
    """Segment *normalized_text* into sentence spans with offsets."""
    if not normalized_text:
        return []

    try:
        sentence_texts = underthesea.sent_tokenize(normalized_text)
    except (RuntimeError, ValueError):
        sentence_texts = []

    if not sentence_texts:
        return _single_span(normalized_text, doc_id, created_at)

    spans: list[SentenceSpanRecord] = []
    search_pos = 0
    for sent_text in sentence_texts:
        if not sent_text.strip():
            continue
        char_start, char_end = _find_offsets(
            normalized_text,
            sent_text,
            search_pos,
        )
        if char_start is None:
            continue

        spans.append(
            SentenceSpanRecord(
                sentence_id=f"{doc_id}:s:{len(spans)}",
                doc_id=doc_id,
                sentence_index=len(spans),
                text=normalized_text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                created_at=created_at,
            ),
        )
        search_pos = char_end

    return spans or _single_span(normalized_text, doc_id, created_at)


def _find_offsets(
    text: str,
    sent: str,
    start_pos: int,
) -> tuple[int | None, int | None]:
    """Find *sent* in *text* starting from *start_pos*."""
    found = text.find(sent, start_pos)
    if found != -1:
        return found, found + len(sent)
    stripped = sent.strip()
    found = text.find(stripped, start_pos)
    if found != -1:
        return found, found + len(stripped)
    return None, None


def _single_span(
    text: str,
    doc_id: str,
    created_at: datetime,
) -> list[SentenceSpanRecord]:
    """Fallback: one span covering the whole text."""
    return [
        SentenceSpanRecord(
            sentence_id=f"{doc_id}:s:0",
            doc_id=doc_id,
            sentence_index=0,
            text=text,
            char_start=0,
            char_end=len(text),
            created_at=created_at,
        ),
    ]


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------


def _process_doc(
    doc: DocumentRecord,
    ts: datetime,
) -> tuple[DocumentRecord, list[SentenceSpanRecord], list[MentionRecord]]:
    """Normalize, segment, and extract mentions for one document."""
    normalized = normalize_document_text(doc.raw_text)
    retrieval = build_retrieval_text(normalized)
    sents = segment_sentences(normalized, doc.doc_id, ts)
    mentions = extract_mentions(doc, ts)

    enriched = DocumentRecord(
        doc_id=doc.doc_id,
        source=doc.source,
        language=doc.language,
        raw_text=doc.raw_text,
        source_record_id=doc.source_record_id,
        split=doc.split,
        payload=doc.payload,
        normalized_text=normalized,
        retrieval_text=retrieval,
        sentences=[s.text for s in sents],
        created_at=doc.created_at,
    )
    return enriched, sents, mentions


# ---------------------------------------------------------------------------
# DC-1 pipeline
# ---------------------------------------------------------------------------

_DATASETS = ("vietmed-sum", "vihealthqa")

_SKIP_FILES = {"errors.jsonl"}


def normalize_dataset(
    dataset_name: str,
    bronze_dir: Path,
    silver_dir: Path,
) -> tuple[BronzeManifest, BronzeManifest, BronzeManifest]:
    """Process one dataset through DC-1.

    Returns ``(docs_manifest, sentences_manifest, mentions_manifest)``.
    """
    input_dir = bronze_dir / dataset_name / "documents"
    if not input_dir.is_dir():
        msg = f"Input directory not found: {input_dir}"
        raise FileNotFoundError(msg)

    doc_out = silver_dir / dataset_name / "documents"
    sent_out = silver_dir / dataset_name / "sentence_spans"
    ment_out = silver_dir / dataset_name / "mentions"

    doc_counts: dict[str, int] = {}
    sent_counts: dict[str, int] = {}
    ment_counts: dict[str, int] = {}
    failed_counts: dict[str, int] = {}
    input_splits: list[str] = []
    all_errors: list[_ParseError] = []

    split_files = sorted(
        f for f in input_dir.glob("*.jsonl") if f.name not in _SKIP_FILES
    )

    for split_file in split_files:
        split = split_file.stem
        input_splits.append(split)
        docs, sents, ments, failed, errors = _process_split(split_file)
        all_errors.extend(errors)

        write_jsonl(docs, doc_out / f"{split}.jsonl")
        write_jsonl(sents, sent_out / f"{split}.jsonl")
        write_jsonl(ments, ment_out / f"{split}.jsonl")

        doc_counts[split] = len(docs)
        sent_counts[split] = len(sents)
        ment_counts[split] = len(ments)
        failed_counts[split] = failed

    if all_errors:
        ment_out.mkdir(parents=True, exist_ok=True)
        with (ment_out / "errors.jsonl").open("w", encoding="utf-8") as f:
            for err in all_errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

    ts = now_utc()
    doc_m = _make_manifest(dataset_name, doc_counts, failed_counts, input_splits, ts)
    sent_m = _make_manifest(dataset_name, sent_counts, failed_counts, input_splits, ts)
    ment_m = _make_manifest(dataset_name, ment_counts, failed_counts, input_splits, ts)

    write_manifest(doc_m, doc_out / "manifest.json")
    write_manifest(sent_m, sent_out / "manifest.json")
    write_manifest(ment_m, ment_out / "manifest.json")

    return doc_m, sent_m, ment_m


class _ParseError(TypedDict):
    line: int
    split: str
    error: str


def _process_split(
    split_file: Path,
) -> tuple[
    list[DocumentRecord],
    list[SentenceSpanRecord],
    list[MentionRecord],
    int,
    list[_ParseError],
]:
    """Process one split file, returning results and error info."""
    docs: list[DocumentRecord] = []
    all_sents: list[SentenceSpanRecord] = []
    all_ments: list[MentionRecord] = []
    failed = 0
    errors: list[_ParseError] = []

    with split_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                doc = DocumentRecord.model_validate_json(stripped)
            except (ValidationError, ValueError) as exc:
                failed += 1
                errors.append(
                    {"line": line_num, "split": split_file.stem, "error": str(exc)},
                )
                continue

            if not doc.raw_text:
                failed += 1
                errors.append(
                    {
                        "line": line_num,
                        "split": split_file.stem,
                        "error": "empty raw_text",
                    },
                )
                continue

            enriched, sents, ments = _process_doc(doc, now_utc())
            docs.append(enriched)
            all_sents.extend(sents)
            all_ments.extend(ments)

    return docs, all_sents, all_ments, failed, errors


def _make_manifest(
    dataset_name: str,
    counts: dict[str, int],
    failed: dict[str, int],
    splits: list[str],
    ts: datetime,
) -> BronzeManifest:
    """Create a manifest for one scope."""
    return BronzeManifest(
        dataset=dataset_name,
        input_splits=splits,
        output_splits=list(counts),
        record_count_by_split=counts,
        failed_count_by_split=failed,
        created_at=ts,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for DC-1 normalization."""
    parser = argparse.ArgumentParser(
        description="DC-1: normalize bronze documents",
    )
    parser.add_argument(
        "--dataset",
        choices=_DATASETS,
        help="Dataset to normalize",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Normalize all datasets",
    )
    parser.add_argument(
        "--bronze-dir",
        default="data/bronze",
        type=Path,
        help="Bronze root directory",
    )
    parser.add_argument(
        "--silver-dir",
        default="data/silver",
        type=Path,
        help="Silver output root directory",
    )
    args = parser.parse_args(argv)

    if not args.dataset and not args.run_all:
        parser.error("Specify --dataset <name> or --all")

    names = list(_DATASETS) if args.run_all else [args.dataset]
    for name in names:
        logger.info("Normalizing %s...", name)
        doc_m, _, _ = normalize_dataset(name, args.bronze_dir, args.silver_dir)
        total = sum(doc_m.record_count_by_split.values())
        failed = sum(doc_m.failed_count_by_split.values())
        logger.info(
            "  %s (%d docs, %d failed)",
            doc_m.record_count_by_split,
            total,
            failed,
        )


if __name__ == "__main__":
    main()
