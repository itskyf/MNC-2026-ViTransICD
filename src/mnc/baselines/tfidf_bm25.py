"""BM-2: TF-IDF/BM25 baseline over bilingual PDF ontology.

Produces two baseline variants:
- BM2-TFIDF: TF-IDF cosine similarity retrieval
- BM2-BM25: BM25 probabilistic retrieval

Both operate at document level over the bilingual ontology search text.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import torch

from mnc.baselines._bm25 import build_bm25_index
from mnc.baselines._vectorizer import build_tfidf_index, query_tfidf
from mnc.datasets._io import iter_jsonl, now_utc, write_jsonl, write_manifest
from mnc.eval.metrics import EvalMetricConfig, MultilabelEvaluator
from mnc.schemas.document import DocumentRecord
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.ontology import OntologyCode
from mnc.schemas.prediction import PredictionRecord
from mnc.schemas.silver import SilverRecord

logger = logging.getLogger(__name__)

_DATASETS = ("vietmed-sum", "vihealthqa")

_ONTO_PATH = Path(
    "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
)


class _ErrorRecord(TypedDict):
    line: int
    split: str
    error: str


@dataclass(frozen=True)
class _RetrievalArtefacts:
    """Bundle of TF-IDF and BM25 retrieval indexes."""

    tfidf_vectorizer: object
    tfidf_matrix: object
    tfidf_codes: list[str]
    bm25_index: object


@dataclass(frozen=True)
class _OutputSpec:
    """Parameters for writing BM-2 outputs."""

    dataset_name: str
    split: str
    gold_dir: Path
    ts: object
    errors: list[_ErrorRecord]
    failed: int
    targets_path: Path | None = None


@dataclass(frozen=True)
class _RunConfig:
    """Configuration for BM-2 pipeline run."""

    silver_dir: Path = Path("data/silver")
    gold_dir: Path = Path("data/gold")
    ontology_path: Path = _ONTO_PATH
    targets_path: Path | None = None
    top_k: int = 5


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
# Core pipeline
# ---------------------------------------------------------------------------


def _query_document(
    doc: DocumentRecord,
    artefacts: _RetrievalArtefacts,
    top_k: int,
    top_n_scores: int,
    ts: object,
) -> tuple[list[PredictionRecord], list[PredictionRecord]]:
    """Query a single document and return TF-IDF and BM25 predictions."""
    query = doc.retrieval_text or doc.normalized_text
    tfidf_preds: list[PredictionRecord] = []
    bm25_preds: list[PredictionRecord] = []

    tfidf_results = query_tfidf(
        artefacts.tfidf_vectorizer,
        artefacts.tfidf_matrix,
        artefacts.tfidf_codes,
        query,
        top_k=top_n_scores,
    )
    if tfidf_results:
        tfidf_preds.append(
            PredictionRecord(
                doc_id=doc.doc_id,
                model_name="bm2_tfidf",
                label_granularity="code_3char",
                predicted_codes=[code for code, _ in tfidf_results[:top_k]],
                scores=dict(tfidf_results),
                created_at=ts,
            ),
        )

    bm25_results = artefacts.bm25_index.query(query, top_k=top_n_scores)
    if bm25_results:
        bm25_preds.append(
            PredictionRecord(
                doc_id=doc.doc_id,
                model_name="bm2_bm25",
                label_granularity="code_3char",
                predicted_codes=[code for code, _ in bm25_results[:top_k]],
                scores=dict(bm25_results),
                created_at=ts,
            ),
        )

    return tfidf_preds, bm25_preds


def _write_outputs(
    tfidf_preds: list[PredictionRecord],
    bm25_preds: list[PredictionRecord],
    spec: _OutputSpec,
) -> None:
    """Write predictions, metrics, manifest, and errors."""
    out_dir = spec.gold_dir / spec.dataset_name / "bm_2_tfidf_bm25"
    write_jsonl(tfidf_preds, out_dir / f"{spec.split}.tfidf.predictions.jsonl")
    write_jsonl(bm25_preds, out_dir / f"{spec.split}.bm25.predictions.jsonl")

    if spec.targets_path and spec.targets_path.exists():
        targets = [rec for _, rec in iter_jsonl(spec.targets_path, SilverRecord)]
        if tfidf_preds:
            metrics = _evaluate_predictions(tfidf_preds, targets)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{spec.split}.tfidf.metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        if bm25_preds:
            metrics = _evaluate_predictions(bm25_preds, targets)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"{spec.split}.bm25.metrics.json").write_text(
                json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

    counts = {
        f"{spec.split}.tfidf": len(tfidf_preds),
        f"{spec.split}.bm25": len(bm25_preds),
    }
    manifest = BronzeManifest(
        dataset=spec.dataset_name,
        input_splits=[spec.split],
        output_splits=list(counts),
        record_count_by_split=counts,
        failed_count_by_split={spec.split: spec.failed},
        created_at=spec.ts,
    )
    write_manifest(manifest, out_dir / "manifest.json")

    if spec.errors:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
            for err in spec.errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")


def run_tfidf_bm25_baselines(
    dataset_name: str,
    split: str,
    config: _RunConfig | None = None,
) -> dict[str, list[PredictionRecord]]:
    """Run TF-IDF and BM25 ICD baselines for one dataset split.

    Args:
        dataset_name: Dataset identifier.
        split: Data split name.
        config: Optional pipeline configuration; uses defaults if omitted.

    Returns:
        Dict with keys ``"tfidf"`` and ``"bm25"``, each containing a list
        of :class:`PredictionRecord`.
    """
    cfg = config or _RunConfig()
    doc_path = cfg.silver_dir / dataset_name / "documents" / f"{split}.jsonl"

    if not doc_path.exists():
        msg = f"Missing documents: {doc_path}"
        raise FileNotFoundError(msg)

    docs = [rec for _, rec in iter_jsonl(doc_path, DocumentRecord)]

    if not cfg.ontology_path.exists():
        msg = f"Missing ontology: {cfg.ontology_path}"
        raise FileNotFoundError(msg)
    ontology_codes = [rec for _, rec in iter_jsonl(cfg.ontology_path, OntologyCode)]

    tfidf_vectorizer, tfidf_matrix, tfidf_codes = build_tfidf_index(ontology_codes)
    bm25_index = build_bm25_index(ontology_codes)
    artefacts = _RetrievalArtefacts(
        tfidf_vectorizer=tfidf_vectorizer,
        tfidf_matrix=tfidf_matrix,
        tfidf_codes=tfidf_codes,
        bm25_index=bm25_index,
    )

    ts = now_utc()
    tfidf_preds: list[PredictionRecord] = []
    bm25_preds: list[PredictionRecord] = []
    errors: list[_ErrorRecord] = []
    failed = 0
    top_n_scores = 100

    for doc in docs:
        try:
            doc_tfidf, doc_bm25 = _query_document(
                doc,
                artefacts,
                cfg.top_k,
                top_n_scores,
                ts,
            )
            tfidf_preds.extend(doc_tfidf)
            bm25_preds.extend(doc_bm25)
        except (ValueError, TypeError) as exc:
            failed += 1
            errors.append({"line": 0, "split": split, "error": str(exc)})

    _write_outputs(
        tfidf_preds,
        bm25_preds,
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
    return {"tfidf": tfidf_preds, "bm25": bm25_preds}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for BM-2 TF-IDF/BM25 baselines."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="BM-2: TF-IDF/BM25 ICD baselines",
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
    result = run_tfidf_bm25_baselines(
        dataset_name=args.dataset,
        split=args.split,
        config=_RunConfig(
            silver_dir=args.silver_dir,
            gold_dir=args.gold_dir,
            ontology_path=args.ontology_path,
            targets_path=args.targets_path,
            top_k=args.top_k,
        ),
    )
    logger.info(
        "  %s/%s: %d TF-IDF, %d BM25 predictions",
        args.dataset,
        args.split,
        len(result["tfidf"]),
        len(result["bm25"]),
    )


if __name__ == "__main__":
    main()
