"""Candidate merge and ranking utilities for DC-3.

All functions are deterministic. Tie-breaking uses code_3char ascending.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnc.schemas.candidate import CandidateLink


def merge_candidates(
    mention_links: list[CandidateLink],
    doc_links: list[CandidateLink],
) -> list[CandidateLink]:
    """Merge mention-level and document-level candidate links.

    Deduplicates by ``(doc_id, mention_id, code_3char, method)``, keeping the
    row with the highest score.  Returns a flat list sorted by
    ``(doc_id, code_3char, method)`` for deterministic output.
    """
    seen: dict[tuple[str, str | None, str, str], CandidateLink] = {}
    for link in mention_links + doc_links:
        key = (link.doc_id, link.mention_id, link.code_3char, link.method)
        existing = seen.get(key)
        if existing is None or link.score > existing.score:
            seen[key] = link

    merged = list(seen.values())
    merged.sort(key=lambda c: (c.doc_id, c.code_3char, c.method))
    return merged


def rank_and_cut(
    candidates: list[CandidateLink],
    top_k_per_doc: int = 50,
) -> list[CandidateLink]:
    """Keep the top-k highest-scoring unique codes per document.

    Tie-breaking: higher score first, then code_3char ascending.
    Returns candidates sorted by ``(doc_id, -score, code_3char)``.
    """
    best_by_doc_code: dict[tuple[str, str], CandidateLink] = {}
    for c in candidates:
        key = (c.doc_id, c.code_3char)
        existing = best_by_doc_code.get(key)
        if (
            existing is None
            or c.score > existing.score
            or (c.score == existing.score and c.code_3char < existing.code_3char)
        ):
            best_by_doc_code[key] = c

    by_doc: dict[str, list[CandidateLink]] = {}
    for c in best_by_doc_code.values():
        by_doc.setdefault(c.doc_id, []).append(c)

    result: list[CandidateLink] = []
    for doc_id in sorted(by_doc):
        group = by_doc[doc_id]
        group.sort(key=lambda c: (-c.score, c.code_3char))
        result.extend(group[:top_k_per_doc])

    return result
