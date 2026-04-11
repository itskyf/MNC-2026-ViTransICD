"""DC-3: Candidate ICD generation (PDF-first ontology).

Generate ICD-10 3-character candidate links for each document using the
official PDF-derived ontology.  Supports exact, normalized, fuzzy, TF-IDF,
and BM25 match channels.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from rapidfuzz import fuzz as rfuzz
from rapidfuzz import process as rprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mnc.datasets._candidate_rank import merge_candidates, rank_and_cut
from mnc.datasets._io import iter_jsonl, now_utc, write_jsonl, write_manifest
from mnc.datasets._lexical_index import (
    build_alias_index,
    build_fuzzy_index,
    build_search_corpus,
    build_title_index,
)
from mnc.schemas.alias import AliasRecord
from mnc.schemas.candidate import CandidateLink
from mnc.schemas.document import DocumentRecord
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.mention import MentionRecord
from mnc.schemas.ontology import OntologyCode

logger = logging.getLogger(__name__)

_DATASETS = ("vietmed-sum", "vihealthqa")
_SPLITS = ("train", "dev", "val", "test")

_ONTO_PATH = Path(
    "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
)
_ALIAS_PATH = Path(
    "data/silver/icd10_official_pdf/alias_dictionary/alias_records.jsonl",
)


class _ErrorRecord(TypedDict):
    line: int
    split: str
    error: str


@dataclass(frozen=True)
class _RetrievalIndex:
    """Bundle of TF-IDF and BM25 retrieval artefacts."""

    vectorizer: TfidfVectorizer
    tfidf_matrix: object
    tfidf_codes: list[str]
    corpus: list[dict[str, str]]
    corpus_tokens: list[list[str]]
    avgdl: float
    df: dict[str, int]
    k1: float = 1.5
    b: float = 0.75


@dataclass(frozen=True)
class _MentionIndex:
    """Bundle of mention-level lookup indexes."""

    title: dict[str, str]
    alias: dict[str, str]
    fuzzy: list[tuple[str, str]]


@dataclass(frozen=True)
class _PipelineInput:
    """Bundle of documents and their grouped mentions."""

    docs: list[DocumentRecord]
    mentions_by_doc: dict[str, list[MentionRecord]]


@dataclass(frozen=True)
class _PathConfig:
    """Path configuration for candidate generation."""

    silver_dir: Path = Path("data/silver")
    ontology_path: Path = _ONTO_PATH
    alias_path: Path | None = _ALIAS_PATH


# ---------------------------------------------------------------------------
# Scoring constants
# ---------------------------------------------------------------------------

_EXACT_SCORE = 1.0
_NORMALIZED_SCORE = 0.85
_FUZZY_MIN_SCORE = 0.75
_MENTION_TOP_K = 10
_DOC_TOP_K = 20
_FINAL_TOP_K = 50


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------


def _exact_title_matches(
    mention_text: str,
    title_index: dict[str, str],
) -> list[tuple[str, float]]:
    """Return (code_3char, score) for exact title matches."""
    key = mention_text.lower().strip()
    code = title_index.get(key)
    if code:
        return [(code, _EXACT_SCORE)]
    return []


def _normalized_alias_matches(
    mention_text: str,
    alias_index: dict[str, str],
) -> list[tuple[str, float]]:
    """Return (code_3char, score) for normalized alias matches."""
    key = mention_text.lower().strip()
    code = alias_index.get(key)
    if code:
        return [(code, _NORMALIZED_SCORE)]
    return []


def _fuzzy_matches(
    mention_text: str,
    fuzzy_index: list[tuple[str, str]],
    top_k: int = _MENTION_TOP_K,
) -> list[tuple[str, float]]:
    """Return (code_3char, score) for fuzzy lexical matches."""
    if not mention_text.strip() or not fuzzy_index:
        return []

    choices = {i: text for i, (text, _) in enumerate(fuzzy_index)}
    results = rprocess.extract(
        mention_text,
        choices,
        scorer=rfuzz.token_sort_ratio,
        limit=top_k,
        score_cutoff=int(_FUZZY_MIN_SCORE * 100),
    )
    matches: list[tuple[str, float]] = []
    seen_codes: set[str] = set()
    for _match, score_int, idx in results:
        code = fuzzy_index[idx][1]
        if code not in seen_codes:
            seen_codes.add(code)
            score = min(score_int / 100.0, 1.0)
            matches.append((code, score))
    return matches


# ---------------------------------------------------------------------------
# TF-IDF / BM25 document-level retrieval
# ---------------------------------------------------------------------------


def _build_retrieval_index(
    corpus: list[dict[str, str]],
) -> _RetrievalIndex:
    """Build TF-IDF and BM25 retrieval artefacts from the search corpus."""
    codes = [doc["code_3char"] for doc in corpus]
    texts = [doc["text"] for doc in corpus]
    vectorizer = TfidfVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform(texts)

    corpus_tokens = [doc["text"].lower().split() for doc in corpus]
    total_tokens = sum(len(t) for t in corpus_tokens)
    avgdl = total_tokens / len(corpus_tokens) if corpus_tokens else 1.0
    df: dict[str, int] = defaultdict(int)
    for tokens in corpus_tokens:
        for t in set(tokens):
            df[t] += 1

    return _RetrievalIndex(
        vectorizer=vectorizer,
        tfidf_matrix=matrix,
        tfidf_codes=codes,
        corpus=corpus,
        corpus_tokens=corpus_tokens,
        avgdl=avgdl,
        df=dict(df),
    )


def _query_tfidf(
    retrieval: _RetrievalIndex,
    query: str,
    top_k: int = _DOC_TOP_K,
) -> list[tuple[str, float]]:
    """Return (code_3char, score) pairs from TF-IDF retrieval."""
    if not query.strip():
        return []
    query_vec = retrieval.vectorizer.transform([query])
    sims = cosine_similarity(query_vec, retrieval.tfidf_matrix).flatten()
    max_sim = float(sims.max())
    if max_sim <= 0:
        return []
    sims_norm = sims / max_sim

    indexed = [
        (float(sims_norm[i]), retrieval.tfidf_codes[i])
        for i in range(len(retrieval.tfidf_codes))
    ]
    indexed.sort(key=lambda x: (-x[0], x[1]))
    return [(code, score) for score, code in indexed[:top_k] if score > 0]


def _bm25_score(
    retrieval: _RetrievalIndex,
    query_tokens: list[str],
    doc_tokens: list[str],
    doc_count: int,
) -> float:
    """Compute BM25 score for a single query-document pair."""
    score = 0.0
    dl = len(doc_tokens)
    for qt in query_tokens:
        n = retrieval.df.get(qt, 0)
        idf = math.log(
            (doc_count - n + 0.5) / (n + 0.5) + 1.0,
        )
        tf = doc_tokens.count(qt)
        denom = tf * (retrieval.k1 + 1)
        numer = tf * (
            retrieval.k1 * (1 - retrieval.b + retrieval.b * dl / retrieval.avgdl)
        ) + retrieval.k1 * (1 - retrieval.b + retrieval.b * dl / retrieval.avgdl)
        if numer > 0:
            score += idf * denom / numer
    return score


def _query_bm25(
    retrieval: _RetrievalIndex,
    query: str,
    top_k: int = _DOC_TOP_K,
) -> list[tuple[str, float]]:
    """Return (code_3char, score) pairs from BM25 retrieval."""
    if not query.strip():
        return []
    query_tokens = query.lower().split()
    doc_count = len(retrieval.corpus)
    scored: list[tuple[float, str]] = []
    for i, doc_tokens in enumerate(retrieval.corpus_tokens):
        s = _bm25_score(retrieval, query_tokens, doc_tokens, doc_count)
        scored.append((s, retrieval.corpus[i]["code_3char"]))

    max_score = max((s for s, _ in scored), default=0.0)
    if max_score <= 0:
        return []

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [(code, s / max_score) for s, code in scored[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _load_optional_aliases(alias_path: Path | None) -> list[AliasRecord]:
    """Load alias records from an optional path."""
    if not alias_path or not alias_path.exists():
        return []
    return [rec for _, rec in iter_jsonl(alias_path, AliasRecord)]


def _mention_candidates(
    mention: MentionRecord,
    mention_idx: _MentionIndex,
    ts: object,
) -> list[CandidateLink]:
    """Generate mention-level candidates for a single mention."""
    links: list[CandidateLink] = []
    seen: dict[tuple[str, str], float] = {}

    def _add(code: str, method: str, score: float) -> None:
        key = (code, method)
        if key not in seen or score > seen[key]:
            seen[key] = score
            links.append(
                CandidateLink(
                    doc_id=mention.doc_id,
                    mention_id=mention.mention_id,
                    code_3char=code,
                    method=method,
                    score=score,
                    char_start=mention.char_start,
                    char_end=mention.char_end,
                    created_at=ts,
                ),
            )

    for code, score in _exact_title_matches(mention.text, mention_idx.title):
        _add(code, "exact", score)

    for code, score in _normalized_alias_matches(
        mention.normalized_text,
        mention_idx.alias,
    ):
        _add(code, "normalized", score)

    for code, score in _fuzzy_matches(mention.text, mention_idx.fuzzy, _MENTION_TOP_K):
        _add(code, "fuzzy", score)

    return links


def _doc_candidates(
    doc: DocumentRecord,
    retrieval: _RetrievalIndex,
    ts: object,
) -> list[CandidateLink]:
    """Generate document-level retrieval candidates."""
    query = doc.retrieval_text or doc.normalized_text
    links: list[CandidateLink] = []

    for code, score in _query_tfidf(retrieval, query, _DOC_TOP_K):
        links.append(
            CandidateLink(
                doc_id=doc.doc_id,
                mention_id=None,
                code_3char=code,
                method="tfidf",
                score=score,
                char_start=None,
                char_end=None,
                created_at=ts,
            ),
        )

    for code, score in _query_bm25(retrieval, query, _DOC_TOP_K):
        links.append(
            CandidateLink(
                doc_id=doc.doc_id,
                mention_id=None,
                code_3char=code,
                method="bm25",
                score=score,
                char_start=None,
                char_end=None,
                created_at=ts,
            ),
        )

    return links


def _process_documents(
    pipeline_input: _PipelineInput,
    mention_idx: _MentionIndex,
    retrieval: _RetrievalIndex,
    split: str,
    ts: object,
) -> tuple[list[CandidateLink], list[CandidateLink], list[_ErrorRecord], int]:
    """Process documents and return mention/doc links, errors, and fail count."""
    all_mention_links: list[CandidateLink] = []
    all_doc_links: list[CandidateLink] = []
    errors: list[_ErrorRecord] = []
    failed = 0

    for doc in pipeline_input.docs:
        try:
            for mention in pipeline_input.mentions_by_doc.get(doc.doc_id, []):
                all_mention_links.extend(
                    _mention_candidates(mention, mention_idx, ts),
                )
            all_doc_links.extend(_doc_candidates(doc, retrieval, ts))
        except (ValueError, TypeError) as exc:
            failed += 1
            errors.append({"line": 0, "split": split, "error": str(exc)})

    return all_mention_links, all_doc_links, errors, failed


def generate_icd_candidates(
    dataset_name: str,
    split: str,
    paths: _PathConfig | None = None,
) -> list[CandidateLink]:
    """Generate PDF-first ICD candidate links for one dataset split.

    Args:
        dataset_name: Dataset identifier (e.g. ``"vietmed-sum"``).
        split: Data split name (e.g. ``"train"``).
        paths: Optional path configuration; uses defaults if omitted.

    Returns:
        List of :class:`CandidateLink` records.
    """
    cfg = paths or _PathConfig()
    doc_path = cfg.silver_dir / dataset_name / "documents" / f"{split}.jsonl"
    mention_path = (
        cfg.silver_dir / dataset_name / "canonical_mentions" / f"{split}.jsonl"
    )

    if not doc_path.exists():
        msg = f"Missing documents: {doc_path}"
        raise FileNotFoundError(msg)
    if not mention_path.exists():
        msg = f"Missing canonical mentions: {mention_path}"
        raise FileNotFoundError(msg)

    docs = [rec for _, rec in iter_jsonl(doc_path, DocumentRecord)]
    mentions = [rec for _, rec in iter_jsonl(mention_path, MentionRecord)]

    if not cfg.ontology_path.exists():
        msg = f"Missing ontology: {cfg.ontology_path}"
        raise FileNotFoundError(msg)
    ontology_codes = [rec for _, rec in iter_jsonl(cfg.ontology_path, OntologyCode)]

    alias_records = _load_optional_aliases(cfg.alias_path)

    mention_idx = _MentionIndex(
        title=build_title_index(ontology_codes),
        alias=build_alias_index(alias_records),
        fuzzy=build_fuzzy_index(ontology_codes),
    )
    corpus = build_search_corpus(ontology_codes)
    retrieval = _build_retrieval_index(corpus)

    mentions_by_doc: dict[str, list[MentionRecord]] = defaultdict(list)
    for m in mentions:
        mentions_by_doc[m.doc_id].append(m)

    pipeline_input = _PipelineInput(docs=docs, mentions_by_doc=mentions_by_doc)

    ts = now_utc()
    mention_links, doc_links, errors, failed = _process_documents(
        pipeline_input,
        mention_idx,
        retrieval,
        split,
        ts,
    )

    merged = merge_candidates(mention_links, doc_links)
    ranked = rank_and_cut(merged, _FINAL_TOP_K)

    out_dir = cfg.silver_dir / dataset_name / "candidate_links"
    write_jsonl(ranked, out_dir / f"{split}.jsonl")

    counts = {split: len(ranked)}
    failed_counts = {split: failed}
    manifest = BronzeManifest(
        dataset=dataset_name,
        input_splits=[split],
        output_splits=list(counts),
        record_count_by_split=counts,
        failed_count_by_split=failed_counts,
        created_at=ts,
    )
    write_manifest(manifest, out_dir / "manifest.json")

    if errors:
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "errors.jsonl").open("w", encoding="utf-8") as f:
            for err in errors:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

    return ranked


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for DC-3 candidate generation."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="DC-3: Generate ICD candidate links",
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
            result = generate_icd_candidates(
                dataset_name=name,
                split=split,
                paths=_PathConfig(
                    silver_dir=args.silver_dir,
                    ontology_path=args.ontology_path,
                    alias_path=args.alias_path,
                ),
            )
            logger.info("  %s/%s: %d candidates", name, split, len(result))


if __name__ == "__main__":
    main()
