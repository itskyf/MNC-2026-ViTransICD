"""BM-1: Rule-based baseline using official PDF notes.

Deterministic ICD-10 3-character code prediction from document text and
extracted mentions using official PDF-derived ontology evidence.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch

from mnc.baselines._rule_scoring import (
    WEIGHT_EXACT_ALIAS,
    WEIGHT_EXACT_TITLE,
    WEIGHT_NORMALIZED,
    aggregate_scores,
    prune_by_rules,
    rank_predictions,
)
from mnc.datasets._io import iter_jsonl, now_utc, write_jsonl, write_manifest
from mnc.datasets._lexical_index import build_alias_index, build_title_index
from mnc.eval.metrics import EvalMetricConfig, MultilabelEvaluator
from mnc.schemas.alias import AliasRecord
from mnc.schemas.document import DocumentRecord
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.mention import MentionRecord
from mnc.schemas.ontology import OntologyCode
from mnc.schemas.prediction import PredictionRecord
from mnc.schemas.rule import RuleRecord
from mnc.schemas.silver import SilverRecord

logger = logging.getLogger(__name__)

_DATASETS = ("vietmed-sum", "vihealthqa")

_ONTO_PATH = Path(
    "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
)
_ALIAS_PATH = Path(
    "data/silver/icd10_official_pdf/alias_dictionary/alias_records.jsonl",
)
_RULES_PATH = Path(
    "data/silver/icd10_official_pdf/coding_rules/rule_records.jsonl",
)


class _ErrorRecord(TypedDict):
    line: int
    split: str
    error: str


@dataclass(frozen=True)
class _PipelineContext:
    """Shared context for the BM-1 pipeline run."""

    title_index: dict[str, str]
    alias_index: dict[str, str]
    rule_records: list[RuleRecord]
    top_k: int


@dataclass(frozen=True)
class _OutputSpec:
    """Parameters for writing BM-1 outputs."""

    dataset_name: str
    split: str
    gold_dir: Path
    ts: object
    errors: list[_ErrorRecord]
    failed: int
    targets_path: Path | None = None


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for BM-1 pipeline run."""

    silver_dir: Path = Path("data/silver")
    gold_dir: Path = Path("data/gold")
    ontology_path: Path = _ONTO_PATH
    alias_path: Path | None = _ALIAS_PATH
    rules_path: Path | None = _RULES_PATH
    targets_path: Path | None = None
    top_k: int = 5


# ---------------------------------------------------------------------------
# Mention matching helpers
# ---------------------------------------------------------------------------


def _match_mention(
    mention: MentionRecord,
    ctx: _PipelineContext,
) -> list[tuple[str, str, float]]:
    """Return (code_3char, source_type, weight) for a mention's matches."""
    matches: list[tuple[str, str, float]] = []

    key = mention.text.lower().strip()
    if key in ctx.title_index:
        matches.append((ctx.title_index[key], "exact_title", WEIGHT_EXACT_TITLE))

    norm_key = mention.normalized_text.lower().strip()
    if norm_key in ctx.alias_index:
        matches.append((ctx.alias_index[norm_key], "exact_alias", WEIGHT_EXACT_ALIAS))

    if norm_key != key and norm_key in ctx.title_index:
        code = ctx.title_index[norm_key]
        if not any(m[0] == code and m[1] == "exact_title" for m in matches):
            matches.append((code, "normalized", WEIGHT_NORMALIZED))

    return matches


def _doc_lexical_matches(
    doc: DocumentRecord,
    title_index: dict[str, str],
) -> list[tuple[str, float]]:
    """Return document-level lexical support scores.

    Scans retrieval text for ontology title matches and returns raw scores.
    """
    text = (doc.retrieval_text or doc.normalized_text).lower()
    matches: list[tuple[str, float]] = []
    seen: set[str] = set()

    for title, code in title_index.items():
        if code in seen:
            continue
        if title in text:
            seen.add(code)
            raw = min(len(title) / max(len(text), 1) * 10, 1.0)
            matches.append((code, max(raw, 0.3)))

    return matches


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


_MIN_LABELS_FOR_EVAL = 2


def _evaluate_predictions(
    predictions: list[PredictionRecord],
    targets: list[SilverRecord],
) -> dict[str, float]:
    """Compute evaluation metrics using TE-2 MultilabelEvaluator."""
    all_codes: set[str] = set()
    for pred in predictions:
        all_codes.update(pred.scores)
    for target in targets:
        all_codes.update(target.silver_labels)

    code_list = sorted(all_codes)
    code_to_idx = {code: i for i, code in enumerate(code_list)}
    num_labels = len(code_list)

    if num_labels < _MIN_LABELS_FOR_EVAL:
        return {}

    config = EvalMetricConfig(
        label_granularity="code_3char",
        num_labels=num_labels,
        threshold=0.5,
    )
    evaluator = MultilabelEvaluator(config)
    targets_by_doc: dict[str, SilverRecord] = {t.doc_id: t for t in targets}

    for pred in predictions:
        scores = torch.zeros(1, num_labels)
        target = torch.zeros(1, num_labels, dtype=torch.int)

        for code, score in pred.scores.items():
            if code in code_to_idx:
                scores[0, code_to_idx[code]] = score

        t = targets_by_doc.get(pred.doc_id)
        if t:
            for label in t.silver_labels:
                if label in code_to_idx:
                    target[0, code_to_idx[label]] = 1

        evaluator.update(scores, target)

    return evaluator.compute()


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _load_optional_aliases(alias_path: Path | None) -> list[AliasRecord]:
    """Load alias records from an optional path."""
    if not alias_path or not alias_path.exists():
        return []
    return [rec for _, rec in iter_jsonl(alias_path, AliasRecord)]


def _load_optional_rules(rules_path: Path | None) -> list[RuleRecord]:
    """Load rule records from an optional path."""
    if not rules_path or not rules_path.exists():
        return []
    return [rec for _, rec in iter_jsonl(rules_path, RuleRecord)]


def _load_ontology(ontology_path: Path) -> list[OntologyCode]:
    """Load ontology codes, raising FileNotFoundError if missing."""
    if not ontology_path.exists():
        msg = f"Missing ontology: {ontology_path}"
        raise FileNotFoundError(msg)
    return [rec for _, rec in iter_jsonl(ontology_path, OntologyCode)]


def _write_outputs(
    predictions: list[PredictionRecord],
    spec: _OutputSpec,
) -> None:
    """Write predictions, optional metrics, manifest, and errors."""
    out_dir = spec.gold_dir / spec.dataset_name / "bm_1_rule_based"
    write_jsonl(predictions, out_dir / f"{spec.split}.predictions.jsonl")

    if spec.targets_path and spec.targets_path.exists():
        targets = [rec for _, rec in iter_jsonl(spec.targets_path, SilverRecord)]
        metrics = _evaluate_predictions(predictions, targets)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{spec.split}.metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )

    counts = {spec.split: len(predictions)}
    failed_counts = {spec.split: spec.failed}
    manifest = BronzeManifest(
        dataset=spec.dataset_name,
        input_splits=[spec.split],
        output_splits=list(counts),
        record_count_by_split=counts,
        failed_count_by_split=failed_counts,
        created_at=spec.ts,
    )
    write_manifest(manifest, out_dir / "manifest.json")

    if spec.errors:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
            for err in spec.errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _process_documents(
    docs: list[DocumentRecord],
    mentions_by_doc: dict[str, list[MentionRecord]],
    ctx: _PipelineContext,
    split: str,
    ts: object,
) -> tuple[list[PredictionRecord], list[_ErrorRecord], int]:
    """Process documents and produce predictions."""
    predictions: list[PredictionRecord] = []
    errors: list[_ErrorRecord] = []
    failed = 0

    for doc in docs:
        try:
            mention_matches: list[tuple[str, str, float]] = []
            for mention in mentions_by_doc.get(doc.doc_id, []):
                mention_matches.extend(_match_mention(mention, ctx))

            doc_matches = _doc_lexical_matches(doc, ctx.title_index)
            code_scores = aggregate_scores(mention_matches, doc_matches)

            if ctx.rule_records:
                code_scores = prune_by_rules(code_scores, ctx.rule_records)

            if not code_scores:
                continue

            predicted_codes, scores = rank_predictions(code_scores, ctx.top_k)
            predictions.append(
                PredictionRecord(
                    doc_id=doc.doc_id,
                    model_name="bm_1_rule_based",
                    label_granularity="code_3char",
                    predicted_codes=predicted_codes,
                    scores=scores,
                    created_at=ts,
                ),
            )
        except (ValueError, TypeError) as exc:
            failed += 1
            errors.append({"line": 0, "split": split, "error": str(exc)})

    return predictions, errors, failed


def run_rule_based_baseline(
    dataset_name: str,
    split: str,
    config: _RunConfig | None = None,
) -> list[PredictionRecord]:
    """Run the BM-1 rule-based ICD baseline for one dataset split.

    Args:
        dataset_name: Dataset identifier.
        split: Data split name.
        config: Optional pipeline configuration; uses defaults if omitted.

    Returns:
        List of :class:`PredictionRecord` records.
    """
    cfg = config or _RunConfig()
    doc_path = cfg.silver_dir / dataset_name / "documents" / f"{split}.jsonl"
    mention_path = (
        cfg.silver_dir / dataset_name / "canonical_mentions" / f"{split}.jsonl"
    )

    if not doc_path.exists():
        msg = f"Missing documents: {doc_path}"
        raise FileNotFoundError(msg)
    if not mention_path.exists():
        msg = f"Missing mentions: {mention_path}"
        raise FileNotFoundError(msg)

    docs = [rec for _, rec in iter_jsonl(doc_path, DocumentRecord)]
    mentions = [rec for _, rec in iter_jsonl(mention_path, MentionRecord)]
    ontology_codes = _load_ontology(cfg.ontology_path)
    alias_records = _load_optional_aliases(cfg.alias_path)
    rule_records = _load_optional_rules(cfg.rules_path)

    ctx = _PipelineContext(
        title_index=build_title_index(ontology_codes),
        alias_index=build_alias_index(alias_records),
        rule_records=rule_records,
        top_k=cfg.top_k,
    )

    mentions_by_doc: dict[str, list[MentionRecord]] = defaultdict(list)
    for m in mentions:
        mentions_by_doc[m.doc_id].append(m)

    ts = now_utc()
    predictions, errors, failed = _process_documents(
        docs,
        mentions_by_doc,
        ctx,
        split,
        ts,
    )
    _write_outputs(
        predictions,
        _OutputSpec(
            dataset_name=dataset_name,
            split=split,
            gold_dir=cfg.gold_dir,
            ts=ts,
            errors=errors,
            failed=failed,
            targets_path=cfg.targets_path,
        ),
    )
    return predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for BM-1 rule-based baseline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="BM-1: Rule-based ICD baseline",
    )
    parser.add_argument("--dataset", choices=_DATASETS, help="Dataset to process")
    parser.add_argument("--split", required=True, help="Split to process")
    parser.add_argument(
        "--silver-dir",
        default=Path("data/silver"),
        type=Path,
        help="Silver root directory",
    )
    parser.add_argument(
        "--gold-dir",
        default=Path("data/gold"),
        type=Path,
        help="Gold root directory",
    )
    parser.add_argument(
        "--ontology-path",
        default=_ONTO_PATH,
        type=Path,
        help="Path to ontology codes JSONL",
    )
    parser.add_argument(
        "--alias-path",
        default=_ALIAS_PATH,
        type=Path,
        help="Path to alias records JSONL",
    )
    parser.add_argument(
        "--rules-path",
        default=_RULES_PATH,
        type=Path,
        help="Path to rule records JSONL",
    )
    parser.add_argument(
        "--targets-path",
        type=Path,
        help="Path to SilverRecord targets for evaluation",
    )
    parser.add_argument(
        "--top-k",
        default=5,
        type=int,
        help="Number of top predictions",
    )
    args = parser.parse_args(argv)

    if not args.dataset:
        parser.error("Specify --dataset <name>")

    logger.info("Processing %s/%s...", args.dataset, args.split)
    result = run_rule_based_baseline(
        dataset_name=args.dataset,
        split=args.split,
        config=_RunConfig(
            silver_dir=args.silver_dir,
            gold_dir=args.gold_dir,
            ontology_path=args.ontology_path,
            alias_path=args.alias_path,
            rules_path=args.rules_path,
            targets_path=args.targets_path,
            top_k=args.top_k,
        ),
    )
    logger.info("  %s/%s: %d predictions", args.dataset, args.split, len(result))


if __name__ == "__main__":
    main()
