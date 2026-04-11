"""DC-4: Weak supervision aggregation.

Aggregate DC-3 candidate links into positive 3-character ICD weak labels
with traceable evidence spans, using ON-3 rules for deterministic score
adjustments.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from mnc.datasets._io import iter_jsonl, now_utc, write_jsonl, write_manifest
from mnc.schemas.candidate import CandidateLink
from mnc.schemas.document import DocumentRecord
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.mention import MentionRecord
from mnc.schemas.ontology import OntologyCode
from mnc.schemas.rule import RuleRecord
from mnc.schemas.weak_label import WeakEvidenceSpan, WeakLabelRecord

logger = logging.getLogger(__name__)

_DATASETS = ("vietmed-sum", "vihealthqa")
_SPLITS = ("train", "dev", "val", "test")
_RULES_PATH = Path(
    "data/silver/icd10_official_pdf/coding_rules/rule_records.jsonl",
)
_ONTOLOGY_PATH = Path(
    "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
)

_REQUIRED_GLOBAL_TOPICS: frozenset[str] = frozenset(
    {
        "principal_diagnosis",
        "symptom_fallback",
        "mortality_coding",
        "official_3char_policy",
    },
)

_RULE_TOPICS: frozenset[str] = frozenset(
    {
        "include_note",
        "exclude_note",
        "use_additional_code",
        "code_first",
    },
)

_MENTION_METHODS: frozenset[str] = frozenset({"exact", "normalized", "fuzzy"})
_DOC_METHODS: frozenset[str] = frozenset({"tfidf", "bm25"})

_METHOD_WEIGHTS: dict[str, float] = {
    "exact": 1.00,
    "normalized": 0.90,
    "fuzzy": 0.80,
    "tfidf": 0.45,
    "bm25": 0.50,
}

_3CHAR_RE = re.compile(r"^[A-Z][0-9]{2}$")
_MAX_LABELS_PER_DOC = 5
_MAX_EVIDENCE_PER_LABEL = 3
_CONFIDENCE_DECIMALS = 6
_MENTION_SCORE_THRESHOLD = 0.40
_CONFIDENCE_THRESHOLD = 0.50
_EXCLUDE_MENTION_THRESHOLD = 0.90
_RULE_BONUS_INCLUDE = 0.10
_RULE_PENALTY_EXCLUDE = 0.25
_RULE_PENALTY_CODE_FIRST = 0.10
_DOC_SCORE_WEIGHT = 0.35


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_documents(
    silver_dir: Path,
    dataset_name: str,
    split: str,
) -> list[DocumentRecord]:
    """Read DocumentRecord list from silver documents JSONL."""
    path = silver_dir / dataset_name / "documents" / f"{split}.jsonl"
    if not path.exists():
        msg = f"Missing documents: {path}"
        raise FileNotFoundError(msg)
    records = [rec for _, rec in iter_jsonl(path, DocumentRecord)]
    if not records:
        msg = f"Empty documents file: {path}"
        raise ValueError(msg)
    logger.info("Read %d documents from %s", len(records), path)
    return records


def read_mentions(
    silver_dir: Path,
    dataset_name: str,
    split: str,
) -> list[MentionRecord]:
    """Read MentionRecord list from silver canonical_mentions JSONL."""
    path = silver_dir / dataset_name / "canonical_mentions" / f"{split}.jsonl"
    if not path.exists():
        msg = f"Missing canonical mentions: {path}"
        raise FileNotFoundError(msg)
    records = [rec for _, rec in iter_jsonl(path, MentionRecord)]
    if not records:
        msg = f"Empty canonical mentions file: {path}"
        raise ValueError(msg)
    logger.info("Read %d mentions from %s", len(records), path)
    return records


def read_candidates(
    silver_dir: Path,
    dataset_name: str,
    split: str,
) -> list[CandidateLink]:
    """Read CandidateLink list from silver candidate_links JSONL."""
    path = silver_dir / dataset_name / "candidate_links" / f"{split}.jsonl"
    if not path.exists():
        msg = f"Missing candidate links: {path}"
        raise FileNotFoundError(msg)
    records = [rec for _, rec in iter_jsonl(path, CandidateLink)]
    if not records:
        msg = f"Empty candidate links file: {path}"
        raise ValueError(msg)
    logger.info("Read %d candidate links from %s", len(records), path)
    return records


def read_rules(rules_path: Path) -> list[RuleRecord]:
    """Read RuleRecord list from coding rules JSONL."""
    if not rules_path.exists():
        msg = f"Missing coding rules: {rules_path}"
        raise FileNotFoundError(msg)
    records = [rec for _, rec in iter_jsonl(rules_path, RuleRecord)]
    if not records:
        msg = f"Empty coding rules file: {rules_path}"
        raise ValueError(msg)
    logger.info("Read %d rules from %s", len(records), rules_path)
    return records


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_global_rules(rules: list[RuleRecord]) -> None:
    """Fail if required global ON-3 topics are missing."""
    global_topics: set[str] = set()
    for rule in rules:
        if rule.scope == "global":
            global_topics.add(rule.topic)
    missing = _REQUIRED_GLOBAL_TOPICS - global_topics
    if missing:
        msg = f"Missing required global rule topics: {sorted(missing)}"
        raise ValueError(msg)


def _load_ontology_index(ontology_path: Path) -> dict[str, str]:
    """Load ontology codes into a ``{code_3char: chapter_id}`` index."""
    if not ontology_path.exists():
        logger.warning(
            "Ontology file not found, skipping validation: %s",
            ontology_path,
        )
        return {}
    index: dict[str, str] = {}
    for _, rec in iter_jsonl(ontology_path, OntologyCode):
        index[rec.code_3char] = rec.chapter_id or "unknown"
    logger.info("Loaded %d ontology codes from %s", len(index), ontology_path)
    return index


def _validate_against_ontology(
    labels: list[WeakLabelRecord],
    ontology_index: dict[str, str],
) -> None:
    """Log semantic validation stats for emitted labels against ontology."""
    if not ontology_index:
        return

    unknown_codes: set[str] = set()
    chapter_counts: dict[str, int] = defaultdict(int)
    for label in labels:
        chapter = ontology_index.get(label.code_3char)
        if chapter is None:
            unknown_codes.add(label.code_3char)
        else:
            chapter_counts[chapter] += 1

    if unknown_codes:
        logger.warning(
            "Unknown codes not in ontology (%d): %s",
            len(unknown_codes),
            sorted(unknown_codes),
        )

    for chapter in sorted(chapter_counts):
        logger.info("  Chapter %s: %d labels", chapter, chapter_counts[chapter])


# ---------------------------------------------------------------------------
# Grouper
# ---------------------------------------------------------------------------


def group_candidates(
    candidates: list[CandidateLink],
) -> dict[tuple[str, str], list[CandidateLink]]:
    """Group candidates by (doc_id, code_3char)."""
    groups: dict[tuple[str, str], list[CandidateLink]] = defaultdict(list)
    for c in candidates:
        groups[(c.doc_id, c.code_3char)].append(c)
    return dict(groups)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def compute_method_score(
    candidates: list[CandidateLink],
    methods: frozenset[str],
) -> float:
    """Compute max weighted score over candidates matching given methods."""
    best = 0.0
    for c in candidates:
        if c.method in methods:
            weighted = _METHOD_WEIGHTS[c.method] * c.score
            best = max(best, weighted)
    return best


def compute_rule_adjustment(
    code_3char: str,
    rules_by_code: dict[str, list[RuleRecord]],
    mention_score: float,
) -> tuple[float, list[str]]:
    """Compute rule-based bonus/penalty and return (adjustment, rule_ids).

    Adjustment is added to confidence (bonus positive, penalty negative).
    """
    adjustment = 0.0
    applied_rule_ids: list[str] = []

    code_rules = rules_by_code.get(code_3char, [])
    for rule in code_rules:
        if rule.topic not in _RULE_TOPICS:
            continue
        if rule.topic == "include_note":
            adjustment += _RULE_BONUS_INCLUDE
            applied_rule_ids.append(rule.rule_id)
        elif (
            rule.topic == "exclude_note" and mention_score < _EXCLUDE_MENTION_THRESHOLD
        ):
            adjustment -= _RULE_PENALTY_EXCLUDE
            applied_rule_ids.append(rule.rule_id)
        elif (
            rule.topic in ("code_first", "use_additional_code") and mention_score == 0.0
        ):
            adjustment -= _RULE_PENALTY_CODE_FIRST
            applied_rule_ids.append(rule.rule_id)

    return adjustment, sorted(set(applied_rule_ids))


# ---------------------------------------------------------------------------
# Evidence span builder
# ---------------------------------------------------------------------------


def build_evidence_spans(
    mention_candidates: list[CandidateLink],
    mentions_by_id: dict[str, MentionRecord],
) -> list[WeakEvidenceSpan]:
    """Build deduplicated evidence spans from mention-backed candidates."""
    span_data: dict[tuple[str, int, int], tuple[str, set[str], float]] = {}

    for c in mention_candidates:
        if c.mention_id is None or c.char_start is None or c.char_end is None:
            continue
        if c.method not in _MENTION_METHODS:
            continue

        key = (c.mention_id, c.char_start, c.char_end)
        mention = mentions_by_id.get(c.mention_id)
        if mention is None:
            continue

        text = mention.normalized_text or mention.text
        weighted_score = _METHOD_WEIGHTS[c.method] * c.score

        existing = span_data.get(key)
        if existing is None:
            span_data[key] = (text, {c.method}, weighted_score)
        else:
            old_text, old_methods, old_score = existing
            old_methods.add(c.method)
            new_score = max(old_score, weighted_score)
            span_data[key] = (old_text, old_methods, new_score)

    spans: list[WeakEvidenceSpan] = []
    for (mid, cstart, cend), (text, methods, score) in span_data.items():
        spans.append(
            WeakEvidenceSpan(
                mention_id=mid,
                char_start=cstart,
                char_end=cend,
                text=text,
                methods=sorted(methods),
                score=round(score, _CONFIDENCE_DECIMALS),
            ),
        )

    spans.sort(key=lambda s: (-s.score, s.char_start, s.mention_id))
    return spans[:_MAX_EVIDENCE_PER_LABEL]


# ---------------------------------------------------------------------------
# Weak label builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _WriteContext:
    """Bundle for output write parameters."""

    out_dir: Path
    dataset_name: str
    split: str
    ts: object


@dataclass(frozen=True)
class _GroupScores:
    """Bundle of scores and metadata for a single (doc, code) group."""

    doc_id: str
    code_3char: str
    mention_score: float
    doc_score: float
    rule_adjustment: float
    rule_ids: list[str]
    support_methods: list[str]
    evidence_spans: list[WeakEvidenceSpan]


def _try_build_label(
    scores: _GroupScores,
    ts: object,
) -> WeakLabelRecord | None:
    """Build a single WeakLabelRecord if emission criteria are met.

    Returns None if the label does not qualify.
    """
    if scores.mention_score < _MENTION_SCORE_THRESHOLD:
        return None
    if not scores.evidence_spans:
        return None
    if not _3CHAR_RE.match(scores.code_3char):
        return None

    confidence = min(
        1.0,
        scores.mention_score
        + _DOC_SCORE_WEIGHT * scores.doc_score
        + scores.rule_adjustment,
    )
    confidence = round(confidence, _CONFIDENCE_DECIMALS)

    if confidence < _CONFIDENCE_THRESHOLD:
        return None

    return WeakLabelRecord(
        doc_id=scores.doc_id,
        code_3char=scores.code_3char,
        label_type="positive",
        confidence=confidence,
        rank=0,
        support_methods=sorted(set(scores.support_methods)),
        support_rule_ids=scores.rule_ids,
        evidence_spans=scores.evidence_spans,
        created_at=ts,
    )


def _label_mention_score(label: WeakLabelRecord) -> float:
    """Extract the best evidence span score as proxy for mention_score."""
    if label.evidence_spans:
        return max(s.score for s in label.evidence_spans)
    return 0.0


# ---------------------------------------------------------------------------
# Per-group processing
# ---------------------------------------------------------------------------


def _process_group(
    group: list[CandidateLink],
    mentions_by_id: dict[str, MentionRecord],
    code_rules: dict[str, list[RuleRecord]],
    ts: object,
) -> tuple[WeakLabelRecord | None, dict[str, str] | None]:
    """Process one (doc_id, code_3char) group.

    Returns (label_or_none, error_or_none).
    """
    doc_id = group[0].doc_id
    code_3char = group[0].code_3char

    mention_candidates = [c for c in group if c.mention_id is not None]
    doc_candidates = [c for c in group if c.mention_id is None]

    mention_score = compute_method_score(mention_candidates, _MENTION_METHODS)
    doc_score = compute_method_score(doc_candidates, _DOC_METHODS)

    rule_adj, rule_ids = compute_rule_adjustment(
        code_3char,
        code_rules,
        mention_score,
    )

    evidence_spans = build_evidence_spans(mention_candidates, mentions_by_id)
    all_methods = [c.method for c in group]

    scores = _GroupScores(
        doc_id=doc_id,
        code_3char=code_3char,
        mention_score=mention_score,
        doc_score=doc_score,
        rule_adjustment=rule_adj,
        rule_ids=rule_ids,
        support_methods=all_methods,
        evidence_spans=evidence_spans,
    )

    try:
        label = _try_build_label(scores, ts)
    except (ValueError, TypeError) as exc:
        return None, {
            "doc_id": doc_id,
            "code_3char": code_3char,
            "error": str(exc),
        }

    return label, None


def _rank_and_assign(
    labels_by_doc: dict[str, list[WeakLabelRecord]],
) -> list[WeakLabelRecord]:
    """Apply top-K per document and assign contiguous ranks."""
    final: list[WeakLabelRecord] = []
    for doc_id in sorted(labels_by_doc):
        doc_labels = labels_by_doc[doc_id]
        doc_labels.sort(
            key=lambda lbl: (
                -lbl.confidence,
                -_label_mention_score(lbl),
                lbl.code_3char,
            ),
        )
        doc_labels = doc_labels[:_MAX_LABELS_PER_DOC]
        for rank, label in enumerate(doc_labels, start=1):
            final.append(
                WeakLabelRecord(
                    doc_id=label.doc_id,
                    code_3char=label.code_3char,
                    label_type="positive",
                    confidence=label.confidence,
                    rank=rank,
                    support_methods=label.support_methods,
                    support_rule_ids=label.support_rule_ids,
                    evidence_spans=label.evidence_spans,
                    created_at=label.created_at,
                ),
            )
    return final


def _write_outputs(
    labels: list[WeakLabelRecord],
    errors: list[dict[str, str]],
    ctx: _WriteContext,
) -> None:
    """Write JSONL output, manifest, and optional errors file."""
    write_jsonl(labels, ctx.out_dir / f"{ctx.split}.jsonl")

    counts = {ctx.split: len(labels)}
    failed_counts = {ctx.split: len(errors)}
    manifest = BronzeManifest(
        dataset=ctx.dataset_name,
        input_splits=[ctx.split],
        output_splits=list(counts),
        record_count_by_split=counts,
        failed_count_by_split=failed_counts,
        created_at=ctx.ts,
    )
    write_manifest(manifest, ctx.out_dir / "manifest.json")

    if errors:
        ctx.out_dir.mkdir(parents=True, exist_ok=True)
        with (ctx.out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Aggregation core
# ---------------------------------------------------------------------------


def aggregate_weak_labels(
    dataset_name: str,
    split: str,
    silver_dir: str = "data/silver",
    rules_path: str = (
        "data/silver/icd10_official_pdf/coding_rules/rule_records.jsonl"
    ),
) -> list[WeakLabelRecord]:
    """Aggregate candidate links into positive weak labels for one split.

    Args:
        dataset_name: Dataset identifier (e.g. ``"vietmed-sum"``).
        split: Data split name (e.g. ``"train"``).
        silver_dir: Root directory for silver data.
        rules_path: Path to ON-3 coding rules JSONL.

    Returns:
        List of :class:`WeakLabelRecord` instances.
    """
    silver = Path(silver_dir)
    rpath = Path(rules_path)

    logger.info("Aggregating weak labels: %s / %s", dataset_name, split)
    logger.info("  silver_dir: %s", silver)
    logger.info("  rules_path: %s", rpath)

    docs = read_documents(silver, dataset_name, split)
    mentions = read_mentions(silver, dataset_name, split)
    candidates = read_candidates(silver, dataset_name, split)
    rules = read_rules(rpath)

    validate_global_rules(rules)

    mentions_by_id: dict[str, MentionRecord] = {m.mention_id: m for m in mentions}

    code_rules: dict[str, list[RuleRecord]] = defaultdict(list)
    for rule in rules:
        if rule.scope == "code" and rule.code_3char:
            code_rules[rule.code_3char].append(rule)
    logger.info("Loaded %d code-scoped rules", sum(len(v) for v in code_rules.values()))

    groups = group_candidates(candidates)
    ts = now_utc()

    labels_by_doc: dict[str, list[WeakLabelRecord]] = defaultdict(list)
    errors: list[dict[str, str]] = []

    for group in groups.values():
        label, error = _process_group(group, mentions_by_id, code_rules, ts)
        if error is not None:
            errors.append(error)
        if label is not None:
            labels_by_doc[label.doc_id].append(label)

    final_labels = _rank_and_assign(labels_by_doc)
    final_labels.sort(key=lambda lbl: (lbl.doc_id, lbl.rank, lbl.code_3char))

    if not final_labels and docs:
        msg = f"No weak labels produced for non-empty split {dataset_name}/{split}"
        raise ValueError(msg)

    ontology_index = _load_ontology_index(
        silver / "icd10_official_pdf" / "normalized_ontology" / "ontology_codes.jsonl",
    )
    _validate_against_ontology(final_labels, ontology_index)

    _write_outputs(
        final_labels,
        errors,
        _WriteContext(
            out_dir=silver / dataset_name / "weak_labels",
            dataset_name=dataset_name,
            split=split,
            ts=ts,
        ),
    )

    logger.info(
        "Wrote %d weak labels to %s",
        len(final_labels),
        silver / dataset_name / "weak_labels" / f"{split}.jsonl",
    )
    return final_labels


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for DC-4 weak supervision aggregation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="DC-4: Aggregate weak labels from candidate links",
    )
    parser.add_argument("--dataset", choices=_DATASETS, help="Dataset to process")
    parser.add_argument(
        "--all",
        action="store_true",
        dest="run_all",
        help="Process all datasets",
    )
    parser.add_argument("--split", choices=_SPLITS, help="Split to process")
    parser.add_argument(
        "--silver-dir",
        default=Path("data/silver"),
        type=Path,
        help="Silver root directory",
    )
    parser.add_argument(
        "--rules-path",
        default=_RULES_PATH,
        type=Path,
        help="Path to coding rules JSONL",
    )
    args = parser.parse_args(argv)

    if not args.dataset and not args.run_all:
        parser.error("Specify --dataset <name> or --all")

    names = list(_DATASETS) if args.run_all else [args.dataset]
    splits = list(_SPLITS) if not args.split else [args.split]

    for name in names:
        for split in splits:
            doc_file = Path(args.silver_dir) / name / "documents" / f"{split}.jsonl"
            if not doc_file.exists():
                logger.info("Skipping %s/%s (no documents)", name, split)
                continue
            logger.info("Processing %s/%s...", name, split)
            result = aggregate_weak_labels(
                dataset_name=name,
                split=split,
                silver_dir=str(args.silver_dir),
                rules_path=str(args.rules_path),
            )
            logger.info("  %s/%s: %d weak labels", name, split, len(result))


if __name__ == "__main__":
    main()
