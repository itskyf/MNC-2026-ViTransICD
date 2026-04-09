"""Lightweight BM25 implementation for BM-2 baseline.

Implements Okapi BM25 scoring with deterministic tokenization.
No external BM25 library required.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnc.schemas.ontology import OntologyCode


class BM25Index:
    """Deterministic BM25 index over ontology documents.

    Each ontology code is treated as a document.  Tokenization is
    whitespace-based and lowercased.
    """

    def __init__(
        self,
        code_list: list[str],
        doc_tokens: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """Initialize the BM25 index.

        Args:
            code_list: code_3char for each document.
            doc_tokens: Pre-tokenized documents.
            k1: BM25 term frequency saturation parameter.
            b: BM25 length normalization parameter.
        """
        self._codes = code_list
        self._doc_tokens = doc_tokens
        self._k1 = k1
        self._b = b
        self._doc_count = len(doc_tokens)
        self._avgdl = (
            sum(len(t) for t in doc_tokens) / len(doc_tokens) if doc_tokens else 1.0
        )

        # Document frequency: token -> number of docs containing it
        self._df: dict[str, int] = defaultdict(int)
        for tokens in doc_tokens:
            for t in set(tokens):
                self._df[t] += 1

    def query(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Score all documents against the query and return top-k.

        Args:
            query_text: Query string.
            top_k: Number of results to return.

        Returns:
            List of (code_3char, normalized_score) sorted by score descending.
            Scores normalized to [0.0, 1.0].
        """
        if not query_text.strip():
            return []

        query_tokens = query_text.lower().split()
        scored: list[tuple[float, str]] = []

        for i, doc_tokens in enumerate(self._doc_tokens):
            s = self._bm25_score(query_tokens, doc_tokens)
            scored.append((s, self._codes[i]))

        max_score = max((s for s, _ in scored), default=0.0)
        if max_score <= 0:
            return []

        scored.sort(key=lambda x: (-x[0], x[1]))
        return [(code, s / max_score) for s, code in scored[:top_k] if s > 0]

    def _bm25_score(
        self,
        query_tokens: list[str],
        doc_tokens: list[str],
    ) -> float:
        """Compute BM25 score for a query-document pair."""
        score = 0.0
        dl = len(doc_tokens)
        for qt in query_tokens:
            n = self._df.get(qt, 0)
            idf = math.log((self._doc_count - n + 0.5) / (n + 0.5) + 1.0)
            tf = doc_tokens.count(qt)
            if tf == 0:
                continue
            denom = tf * (self._k1 + 1)
            numer = tf * (
                self._k1 * (1 - self._b + self._b * dl / self._avgdl)
            ) + self._k1 * (1 - self._b + self._b * dl / self._avgdl)
            if numer > 0:
                score += idf * denom / numer
        return score


def build_bm25_index(
    ontology_codes: list[OntologyCode],
    k1: float = 1.5,
    b: float = 0.75,
) -> BM25Index:
    """Build a BM25 index over ontology search text.

    Args:
        ontology_codes: List of ontology code records.
        k1: BM25 term frequency saturation parameter.
        b: BM25 length normalization parameter.

    Returns:
        Initialized :class:`BM25Index`.
    """
    codes = [c.code_3char for c in ontology_codes]
    doc_tokens = [c.search_text.lower().split() for c in ontology_codes]
    return BM25Index(codes, doc_tokens, k1=k1, b=b)
