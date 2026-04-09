"""Deterministic rule scoring engine for BM-1 baseline.

Evidence sources in descending priority:
1. exact mention match to official title
2. exact mention match to official alias
3. normalized mention match
4. document-level lexical support
5. optional inclusion-style support from ON-2b
6. optional exclusion-style pruning from ON-3
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnc.schemas.rule import RuleRecord


# ---------------------------------------------------------------------------
# Rule weight constants
# ---------------------------------------------------------------------------

WEIGHT_EXACT_TITLE = 1.00
WEIGHT_EXACT_ALIAS = 0.95
WEIGHT_NORMALIZED = 0.85
WEIGHT_DOC_LEXICAL = 0.60
WEIGHT_INCLUSION = 0.75
BONUS_REPEATED_SOURCE = 0.05
BONUS_CAP = 0.20


# ---------------------------------------------------------------------------
# Scoring types
# ---------------------------------------------------------------------------


class _Evidence:
    """Accumulated evidence for a single code_3char."""

    __slots__ = ("code", "score", "sources")

    def __init__(self, code: str) -> None:
        self.code = code
        self.sources: set[str] = set()
        self.score = 0.0

    def add(self, source: str, weight: float) -> None:
        """Add evidence from a source with a given weight."""
        if source in self.sources:
            return
        self.sources.add(source)
        bonus = (
            min(len(self.sources) - 1, int(BONUS_CAP / BONUS_REPEATED_SOURCE))
            * BONUS_REPEATED_SOURCE
        )
        self.score = max(self.score, weight + bonus)


# ---------------------------------------------------------------------------
# Core scoring
# ---------------------------------------------------------------------------


def aggregate_scores(
    mention_matches: list[tuple[str, str, float]],
    doc_lexical_matches: list[tuple[str, float]],
    inclusion_matches: list[tuple[str, float]] | None = None,
) -> dict[str, float]:
    """Aggregate evidence per code and compute final scores.

    Args:
        mention_matches: List of (code_3char, source_type, weight) tuples.
        doc_lexical_matches: List of (code_3char, score) tuples.
        inclusion_matches: Optional list of (code_3char, score) from ON-2b.

    Returns:
        Mapping from code_3char to aggregated score in [0, 1].
    """
    evidence_by_code: dict[str, _Evidence] = {}

    for code, source_type, weight in mention_matches:
        ev = evidence_by_code.setdefault(code, _Evidence(code))
        ev.add(f"mention:{source_type}", weight)

    for code, score in doc_lexical_matches:
        ev = evidence_by_code.setdefault(code, _Evidence(code))
        ev.add("doc_lexical", min(score * WEIGHT_DOC_LEXICAL, WEIGHT_DOC_LEXICAL))

    if inclusion_matches:
        for code, score in inclusion_matches:
            ev = evidence_by_code.setdefault(code, _Evidence(code))
            ev.add("inclusion", min(score * WEIGHT_INCLUSION, WEIGHT_INCLUSION))

    # Normalize scores to [0, 1]
    raw_scores = {code: ev.score for code, ev in evidence_by_code.items()}
    max_score = max(raw_scores.values(), default=0.0)
    if max_score <= 0:
        return {}

    return {
        code: min(score / max_score, 1.0)
        for code, score in raw_scores.items()
        if score > 0
    }


def prune_by_rules(
    code_scores: dict[str, float],
    rules: list[RuleRecord],
) -> dict[str, float]:
    """Remove codes with conflicting exclude rules.

    Args:
        code_scores: Mapping from code_3char to score.
        rules: Rule records from ON-3.

    Returns:
        Pruned mapping with conflicting codes removed.
    """
    exclude_codes: set[str] = set()
    for rule in rules:
        if rule.topic == "exclude_note" and rule.code_3char:
            exclude_codes.add(rule.code_3char)

    if not exclude_codes:
        return code_scores

    return {
        code: score for code, score in code_scores.items() if code not in exclude_codes
    }


def rank_predictions(
    code_scores: dict[str, float],
    top_k: int = 5,
) -> tuple[list[str], dict[str, float]]:
    """Rank codes by score and return top-k with full score map.

    Args:
        code_scores: Mapping from code_3char to score.
        top_k: Number of top predictions to return.

    Returns:
        Tuple of (predicted_codes, scores_dict).
        Tie-breaking: higher score first, then code_3char ascending.
    """
    ranked = sorted(code_scores.items(), key=lambda x: (-x[1], x[0]))
    predicted = [code for code, _ in ranked[:top_k]]
    return predicted, dict(ranked)
