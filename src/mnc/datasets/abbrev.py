"""DC-2: Abbreviation normalization for canonical mentions.

Usage::

    uv run -m mnc.datasets.abbrev --dataset vietmed-sum
    uv run -m mnc.datasets.abbrev --dataset vihealthqa
    uv run -m mnc.datasets.abbrev --all
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from pydantic import ValidationError

from mnc.datasets._io import now_utc, write_jsonl, write_manifest
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.mention import MentionRecord

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled patterns for definitional extraction
# ---------------------------------------------------------------------------

_RE_FULL_PAREN_ABBR = re.compile(r"([^()\n]{2,60}?)\s*\(([A-Z]{2,6})\)")
_RE_ABBR_PAREN_FULL = re.compile(r"\b([A-Z]{2,6})\s*\(([^)\n]{2,60}?)\)")

_MIN_FULL_FORM_LEN = 3

# ---------------------------------------------------------------------------
# Seed dictionary
# ---------------------------------------------------------------------------

_SEED_PATH = (
    Path(__file__).parent.parent
    / "resources"
    / "abbreviations"
    / "vi_medical_abbrev_seed.json"
)


def load_seed_dict(path: Path | None = None) -> dict[str, str]:
    """Load the seed abbreviation dictionary from JSON."""
    p = path or _SEED_PATH
    if not p.is_file():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {k.upper(): v for k, v in data.items()}


# ---------------------------------------------------------------------------
# Definitional pattern extraction
# ---------------------------------------------------------------------------


def find_definitions(raw_text: str) -> dict[str, str]:
    """Extract abbreviation definitions from document text.

    Supports patterns:
    - ``full form (ABBR)``
    - ``ABBR (full form)``

    Returns mapping of uppercase abbreviation -> full form.
    Ambiguous abbreviations (multiple different expansions) are excluded.
    """
    candidates: dict[str, list[str]] = {}

    for m in _RE_FULL_PAREN_ABBR.finditer(raw_text):
        full = m.group(1).strip()
        abbr = m.group(2).upper()
        if _is_valid_full_form(full):
            candidates.setdefault(abbr, []).append(full)

    for m in _RE_ABBR_PAREN_FULL.finditer(raw_text):
        abbr = m.group(1).upper()
        full = m.group(2).strip()
        if _is_valid_full_form(full):
            candidates.setdefault(abbr, []).append(full)

    result: dict[str, str] = {}
    for abbr, forms in candidates.items():
        unique = list(dict.fromkeys(forms))
        if len(unique) == 1:
            result[abbr] = unique[0]

    return result


def _is_valid_full_form(text: str) -> bool:
    """Check that *text* looks like a valid full-form expansion."""
    if len(text) < _MIN_FULL_FORM_LEN:
        return False
    return any(c.islower() for c in text)


# ---------------------------------------------------------------------------
# Abbreviation normalization
# ---------------------------------------------------------------------------


def normalize_abbreviations(
    mentions: list[MentionRecord],
    raw_text: str,
    seed_dict: dict[str, str],
) -> list[MentionRecord]:
    """Return canonical mentions with abbreviation expansion when resolved.

    Priority:
    1. In-document definitional pattern
    2. Seed dictionary
    3. Fallback to original DC-1 ``normalized_text``
    """
    definitions = find_definitions(raw_text)
    results: list[MentionRecord] = []

    for mention in mentions:
        if mention.mention_type != "abbreviation":
            results.append(mention)
            continue

        resolved = _resolve(mention.text.upper().strip(), definitions, seed_dict)
        if resolved is not None:
            results.append(_expand_mention(mention, resolved))
        else:
            results.append(mention)

    return results


def _resolve(
    abbr: str,
    definitions: dict[str, str],
    seed_dict: dict[str, str],
) -> str | None:
    """Resolve an abbreviation to its expansion, or ``None``."""
    if abbr in definitions:
        return definitions[abbr]
    if abbr in seed_dict:
        return seed_dict[abbr]
    return None


def _expand_mention(mention: MentionRecord, resolved: str) -> MentionRecord:
    """Return a copy of *mention* with ``normalized_text`` set to *resolved*."""
    return MentionRecord(
        mention_id=mention.mention_id,
        doc_id=mention.doc_id,
        text=mention.text,
        normalized_text=resolved,
        mention_type="abbreviation",
        char_start=mention.char_start,
        char_end=mention.char_end,
        confidence=mention.confidence,
        created_at=mention.created_at,
    )


# ---------------------------------------------------------------------------
# DC-2 pipeline
# ---------------------------------------------------------------------------

_DATASETS = ("vietmed-sum", "vihealthqa")


def _load_raw_text_map(doc_dir: Path) -> dict[str, str]:
    """Build doc_id -> raw_text lookup from silver documents."""
    raw_text_map: dict[str, str] = {}
    if not doc_dir.is_dir():
        return raw_text_map
    for doc_file in doc_dir.glob("*.jsonl"):
        if doc_file.name == "errors.jsonl":
            continue
        with doc_file.open("r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    rec = json.loads(stripped)
                    raw_text_map[rec["doc_id"]] = rec.get("raw_text", "")
                except (json.JSONDecodeError, KeyError):
                    continue
    return raw_text_map


def _read_mentions_by_doc(
    split_file: Path,
) -> tuple[dict[str, list[MentionRecord]], list[dict[str, object]], int]:
    """Read mentions grouped by doc_id, collecting errors."""
    mentions_by_doc: dict[str, list[MentionRecord]] = {}
    errors: list[dict[str, object]] = []
    failed = 0

    with split_file.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = MentionRecord.model_validate_json(stripped)
            except (ValidationError, ValueError) as exc:
                failed += 1
                errors.append(
                    {"line": line_num, "split": split_file.stem, "error": str(exc)},
                )
                continue
            mentions_by_doc.setdefault(rec.doc_id, []).append(rec)

    return mentions_by_doc, errors, failed


def abbrev_dataset(
    dataset_name: str,
    silver_dir: Path,
    seed_path: Path | None = None,
) -> BronzeManifest:
    """Process one dataset through DC-2 abbreviation normalization."""
    mention_dir = silver_dir / dataset_name / "mentions"
    doc_dir = silver_dir / dataset_name / "documents"
    out_dir = silver_dir / dataset_name / "canonical_mentions"

    if not mention_dir.is_dir():
        msg = f"Mentions directory not found: {mention_dir}"
        raise FileNotFoundError(msg)

    seed_dict = load_seed_dict(seed_path)
    raw_text_map = _load_raw_text_map(doc_dir)

    record_counts: dict[str, int] = {}
    failed_counts: dict[str, int] = {}
    input_splits: list[str] = []
    all_errors: list[dict[str, object]] = []

    split_files = sorted(
        f for f in mention_dir.glob("*.jsonl") if f.name != "errors.jsonl"
    )

    for split_file in split_files:
        split = split_file.stem
        input_splits.append(split)

        mentions_by_doc, errors, failed = _read_mentions_by_doc(split_file)
        all_errors.extend(errors)

        canonical: list[MentionRecord] = []
        for doc_id, mentions in mentions_by_doc.items():
            raw = raw_text_map.get(doc_id, "")
            canonical.extend(normalize_abbreviations(mentions, raw, seed_dict))

        write_jsonl(canonical, out_dir / f"{split}.jsonl")
        record_counts[split] = len(canonical)
        failed_counts[split] = failed

    if all_errors:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
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
    write_manifest(manifest, out_dir / "manifest.json")
    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for DC-2 abbreviation normalization."""
    parser = argparse.ArgumentParser(
        description="DC-2: normalize abbreviations in mentions",
    )
    parser.add_argument(
        "--dataset",
        choices=_DATASETS,
        help="Dataset to process",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Process all datasets",
    )
    parser.add_argument(
        "--silver-dir",
        default="data/silver",
        type=Path,
        help="Silver root directory",
    )
    args = parser.parse_args(argv)

    if not args.dataset and not args.run_all:
        parser.error("Specify --dataset <name> or --all")

    names = list(_DATASETS) if args.run_all else [args.dataset]
    for name in names:
        logger.info("Normalizing abbreviations for %s...", name)
        manifest = abbrev_dataset(name, args.silver_dir)
        total = sum(manifest.record_count_by_split.values())
        failed = sum(manifest.failed_count_by_split.values())
        logger.info(
            "  %s (%d mentions, %d failed)",
            manifest.record_count_by_split,
            total,
            failed,
        )


if __name__ == "__main__":
    main()
