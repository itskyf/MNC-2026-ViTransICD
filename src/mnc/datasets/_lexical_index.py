"""Lexical index builders for DC-3 candidate generation.

All functions are deterministic and side-effect-free.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mnc.schemas.alias import AliasRecord
    from mnc.schemas.ontology import OntologyCode


def build_title_index(
    ontology_codes: list[OntologyCode],
) -> dict[str, str]:
    """Build a lowercase title lookup mapping to code_3char.

    Returns:
        Mapping from lowercase title text to code_3char.
        Titles from ``title_vi`` and ``title_en`` are included.
    """
    index: dict[str, str] = {}
    for code in ontology_codes:
        vi_key = code.title_vi.lower().strip()
        if vi_key:
            index[vi_key] = code.code_3char
        if code.title_en:
            en_key = code.title_en.lower().strip()
            if en_key:
                index[en_key] = code.code_3char
    return index


def build_alias_index(
    alias_records: list[AliasRecord],
) -> dict[str, str]:
    """Build a lowercase alias lookup mapping to code_3char.

    Returns:
        Mapping from lowercase ``alias_norm`` to code_3char.
    """
    index: dict[str, str] = {}
    for alias in alias_records:
        key = alias.alias_norm.lower().strip()
        if key:
            index[key] = alias.code_3char
    return index


def build_search_corpus(
    ontology_codes: list[OntologyCode],
) -> list[dict[str, str]]:
    """Build a search corpus for TF-IDF and BM25 retrieval.

    Returns:
        List of dicts with ``code_3char`` and ``text`` keys, one per code.
        ``text`` is the code's ``search_text`` field.
    """
    return [
        {"code_3char": code.code_3char, "text": code.search_text}
        for code in ontology_codes
    ]


def build_fuzzy_index(
    ontology_codes: list[OntologyCode],
) -> list[tuple[str, str]]:
    """Build a fuzzy matching index from ontology titles and aliases.

    Returns:
        List of (text, code_3char) pairs covering titles and aliases.
    """
    pairs: list[tuple[str, str]] = []
    for code in ontology_codes:
        if code.title_vi:
            pairs.append((code.title_vi, code.code_3char))
        if code.title_en:
            pairs.append((code.title_en, code.code_3char))
        pairs.extend((alias, code.code_3char) for alias in code.aliases if alias)
    return pairs
