"""TF-IDF vectorizer wrapper for BM-2 baseline.

Builds a deterministic TF-IDF index over ontology search text and supports
cosine-similarity retrieval queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix

    from mnc.schemas.ontology import OntologyCode


def build_tfidf_index(
    ontology_codes: list[OntologyCode],
) -> tuple[TfidfVectorizer, csr_matrix, list[str]]:
    """Build a TF-IDF index over ontology search text.

    Args:
        ontology_codes: List of ontology code records.

    Returns:
        Tuple of (vectorizer, tfidf_matrix, code_list).
        ``code_list[i]`` maps matrix row ``i`` to ``code_3char``.
    """
    codes = [c.code_3char for c in ontology_codes]
    texts = [c.search_text for c in ontology_codes]
    vectorizer = TfidfVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix, codes


def query_tfidf(
    vectorizer: TfidfVectorizer,
    matrix: csr_matrix,
    codes: list[str],
    query: str,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Query the TF-IDF index and return ranked results.

    Args:
        vectorizer: Fitted TF-IDF vectorizer.
        matrix: TF-IDF document matrix.
        codes: code_3char list aligned with matrix rows.
        query: Query text.
        top_k: Number of results to return.

    Returns:
        List of (code_3char, normalized_score) sorted by score descending.
        Scores normalized to [0.0, 1.0].
    """
    if not query.strip():
        return []

    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, matrix).flatten()

    max_sim = float(sims.max())
    if max_sim <= 0:
        return []

    sims_norm = sims / max_sim
    indexed = [(float(sims_norm[i]), codes[i]) for i in range(len(codes))]
    indexed.sort(key=lambda x: (-x[0], x[1]))
    return [(code, score) for score, code in indexed[:top_k] if score > 0]
