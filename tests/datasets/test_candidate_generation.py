"""Tests for DC-3 candidate generation pipeline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

import pytest

from mnc.datasets._candidate_rank import merge_candidates, rank_and_cut
from mnc.datasets._lexical_index import (
    build_alias_index,
    build_fuzzy_index,
    build_search_corpus,
    build_title_index,
)
from mnc.datasets.candidate_generation import (
    _exact_title_matches,
    _fuzzy_matches,
    _normalized_alias_matches,
    _PathConfig,
    generate_icd_candidates,
)
from mnc.schemas.alias import AliasRecord
from mnc.schemas.candidate import CandidateLink
from mnc.schemas.document import DocumentRecord
from mnc.schemas.mention import MentionRecord
from mnc.schemas.ontology import OntologyCode


def _ts() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(tz=UTC)


def _onto(
    code: str,
    title_vi: str,
    title_en: str = "",
    aliases: list[str] | None = None,
) -> OntologyCode:
    """Create test OntologyCode."""
    all_aliases = aliases or []
    search = " | ".join([code, title_vi, title_en, *all_aliases])
    return OntologyCode(
        code_3char=code,
        chapter_id="Ch-1",
        title_vi=title_vi,
        title_en=title_en or None,
        aliases=all_aliases,
        search_text=search,
        created_at=_ts(),
    )


def _alias(code: str, alias: str, alias_type: str = "title_vi") -> AliasRecord:
    """Create test AliasRecord."""
    return AliasRecord(
        alias_id=f"{code}:{alias}",
        code_3char=code,
        alias=alias,
        alias_norm=alias.lower(),
        alias_type=alias_type,
        language="vi",
        match_level="exact",
        created_at=_ts(),
    )


def _doc(doc_id: str, text: str, retrieval_text: str = "") -> DocumentRecord:
    """Create test DocumentRecord."""
    return DocumentRecord(
        doc_id=doc_id,
        source="test",
        language="vi",
        raw_text=text,
        normalized_text=text,
        retrieval_text=retrieval_text or text.lower(),
        created_at=_ts(),
    )


def _mention(
    mention_id: str,
    doc_id: str,
    text: str,
    start: int = 0,
    end: int = 0,
) -> MentionRecord:
    """Create test MentionRecord."""
    return MentionRecord(
        mention_id=mention_id,
        doc_id=doc_id,
        text=text,
        normalized_text=text.lower(),
        mention_type="disease",
        char_start=start,
        char_end=end or len(text),
        created_at=_ts(),
    )


def _write_jsonl(path: Path, records: list[BaseModel]) -> None:
    """Write records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# Lexical index tests
# ---------------------------------------------------------------------------


class TestBuildTitleIndex:
    def test_maps_vi_and_en_titles(self) -> None:
        codes = [_onto("A00", "Bệnh tả", "Cholera")]
        idx = build_title_index(codes)
        assert idx["bệnh tả"] == "A00"
        assert idx["cholera"] == "A00"

    def test_skips_empty_titles(self) -> None:
        codes = [_onto("A00", "Bệnh tả", "")]
        idx = build_title_index(codes)
        assert "bệnh tả" in idx


class TestBuildAliasIndex:
    def test_maps_normalized_aliases(self) -> None:
        aliases = [_alias("A00", "bệnh tả")]
        idx = build_alias_index(aliases)
        assert idx["bệnh tả"] == "A00"


class TestBuildSearchCorpus:
    def test_creates_corpus_entries(self) -> None:
        codes = [_onto("A00", "Bệnh tả")]
        corpus = build_search_corpus(codes)
        assert len(corpus) == 1
        assert corpus[0]["code_3char"] == "A00"
        assert "Bệnh tả" in corpus[0]["text"]


class TestBuildFuzzyIndex:
    def test_includes_titles_and_aliases(self) -> None:
        codes = [_onto("A00", "Bệnh tả", "Cholera", ["tả"])]
        idx = build_fuzzy_index(codes)
        texts = [t for t, _ in idx]
        assert "Bệnh tả" in texts
        assert "Cholera" in texts
        assert "tả" in texts


# ---------------------------------------------------------------------------
# Matching tests
# ---------------------------------------------------------------------------


class TestExactTitleMatches:
    def test_exact_match(self) -> None:
        title_index = {"bệnh tả": "A00"}
        result = _exact_title_matches("Bệnh tả", title_index)
        assert result == [("A00", 1.0)]

    def test_no_match(self) -> None:
        title_index = {"bệnh tả": "A00"}
        result = _exact_title_matches("sốt xuất huyết", title_index)
        assert result == []


class TestNormalizedAliasMatches:
    def test_normalized_match(self) -> None:
        alias_index = {"bệnh tả": "A00"}
        result = _normalized_alias_matches("Bệnh Tả", alias_index)
        assert result == [("A00", 0.85)]


class TestFuzzyMatches:
    def test_fuzzy_match_similar_text(self) -> None:
        fuzzy_index = [("Bệnh tả", "A00"), ("Sốt rét", "B50")]
        result = _fuzzy_matches("bệnh tả", fuzzy_index)
        assert len(result) > 0
        assert result[0][0] == "A00"
        assert result[0][1] > 0.5

    def test_empty_query(self) -> None:
        result = _fuzzy_matches("", [("Bệnh tả", "A00")])
        assert result == []


# ---------------------------------------------------------------------------
# Candidate rank tests
# ---------------------------------------------------------------------------


class TestMergeCandidates:
    def test_deduplicates_by_key(self) -> None:
        ts = _ts()
        links = [
            CandidateLink(
                doc_id="d1",
                mention_id="m1",
                code_3char="A00",
                method="exact",
                score=1.0,
                created_at=ts,
            ),
            CandidateLink(
                doc_id="d1",
                mention_id="m1",
                code_3char="A00",
                method="exact",
                score=0.9,
                created_at=ts,
            ),
        ]
        merged = merge_candidates(links, [])
        assert len(merged) == 1
        assert merged[0].score == 1.0

    def test_keeps_different_methods(self) -> None:
        ts = _ts()
        links = [
            CandidateLink(
                doc_id="d1",
                mention_id="m1",
                code_3char="A00",
                method="exact",
                score=1.0,
                created_at=ts,
            ),
            CandidateLink(
                doc_id="d1",
                mention_id="m1",
                code_3char="A00",
                method="fuzzy",
                score=0.8,
                created_at=ts,
            ),
        ]
        merged = merge_candidates(links, [])
        assert len(merged) == 2


class TestRankAndCut:
    def test_top_k_per_doc(self) -> None:
        ts = _ts()
        links = [
            CandidateLink(
                doc_id="d1",
                code_3char=c,
                method="exact",
                score=float(i) / 10,
                created_at=ts,
            )
            for i, c in enumerate(["A00", "B50", "C00"])
        ]
        result = rank_and_cut(links, top_k_per_doc=2)
        assert len(result) == 2

    def test_tie_breaking_by_code(self) -> None:
        ts = _ts()
        links = [
            CandidateLink(
                doc_id="d1",
                code_3char="B50",
                method="exact",
                score=0.8,
                created_at=ts,
            ),
            CandidateLink(
                doc_id="d1",
                code_3char="A00",
                method="exact",
                score=0.8,
                created_at=ts,
            ),
        ]
        result = rank_and_cut(links, top_k_per_doc=10)
        assert result[0].code_3char == "A00"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestGenerateICDCandidates:
    def _setup_silver(self, tmp_path: Path) -> _PathConfig:
        """Create minimal silver directory with test data."""
        silver = tmp_path / "silver"
        dataset = silver / "test-ds"

        onto_dir = silver / "icd10_official_pdf" / "normalized_ontology"
        onto_dir.mkdir(parents=True)
        codes = [
            _onto("A00", "Bệnh tả", "Cholera"),
            _onto("B50", "Sốt rét", "Malaria"),
            _onto("J00", "Viêm họng", "Pharyngitis"),
            _onto("K35", "Viêm ruột thừa", "Appendicitis"),
            _onto("E11", "Tiểu đường", "Diabetes"),
        ]
        onto_path = onto_dir / "ontology_codes.jsonl"
        _write_jsonl(onto_path, codes)

        doc_dir = dataset / "documents"
        docs = [
            _doc(
                "d1",
                "Bệnh tả là một bệnh truyền nhiễm",
                "bệnh tả là một bệnh truyền nhiễm",
            ),
        ]
        _write_jsonl(doc_dir / "train.jsonl", docs)

        men_dir = dataset / "canonical_mentions"
        mentions = [_mention("d1:m1", "d1", "Bệnh tả", 0, 8)]
        _write_jsonl(men_dir / "train.jsonl", mentions)

        return _PathConfig(silver_dir=silver, ontology_path=onto_path)

    def test_exact_title_match_produces_candidate(self, tmp_path: Path) -> None:
        cfg = self._setup_silver(tmp_path)
        result = generate_icd_candidates(
            dataset_name="test-ds",
            split="train",
            paths=cfg,
        )
        codes = {c.code_3char for c in result}
        assert "A00" in codes

    def test_doc_level_retrieval_produces_candidates(self, tmp_path: Path) -> None:
        cfg = self._setup_silver(tmp_path)
        result = generate_icd_candidates(
            dataset_name="test-ds",
            split="train",
            paths=cfg,
        )
        doc_level = [c for c in result if c.method in ("tfidf", "bm25")]
        assert len(doc_level) > 0
        assert all(c.mention_id is None for c in doc_level)

    def test_bm25_produces_doc_level_candidates(self, tmp_path: Path) -> None:
        cfg = self._setup_silver(tmp_path)
        result = generate_icd_candidates(
            dataset_name="test-ds",
            split="train",
            paths=cfg,
        )
        bm25_links = [c for c in result if c.method == "bm25"]
        assert len(bm25_links) > 0
        assert all(c.mention_id is None for c in bm25_links)

    def test_offsets_preserved_for_mention_matches(self, tmp_path: Path) -> None:
        cfg = self._setup_silver(tmp_path)
        result = generate_icd_candidates(
            dataset_name="test-ds",
            split="train",
            paths=cfg,
        )
        mention_links = [c for c in result if c.mention_id is not None]
        for link in mention_links:
            assert link.char_start is not None
            assert link.char_end is not None

    def test_missing_canonical_mentions_fails(self, tmp_path: Path) -> None:
        silver = tmp_path / "silver"
        dataset = silver / "test-ds"
        doc_dir = dataset / "documents"
        doc_dir.mkdir(parents=True)
        _write_jsonl(doc_dir / "train.jsonl", [_doc("d1", "test")])

        with pytest.raises(FileNotFoundError, match="Missing canonical mentions"):
            generate_icd_candidates(
                dataset_name="test-ds",
                split="train",
                paths=_PathConfig(silver_dir=silver),
            )

    def test_manifest_generation(self, tmp_path: Path) -> None:
        cfg = self._setup_silver(tmp_path)
        generate_icd_candidates(
            dataset_name="test-ds",
            split="train",
            paths=cfg,
        )
        manifest_path = cfg.silver_dir / "test-ds" / "candidate_links" / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "train" in data["record_count_by_split"]

    def test_output_validates_against_candidate_link(self, tmp_path: Path) -> None:
        cfg = self._setup_silver(tmp_path)
        result = generate_icd_candidates(
            dataset_name="test-ds",
            split="train",
            paths=cfg,
        )
        for link in result:
            assert isinstance(link, CandidateLink)
            assert 0.0 <= link.score <= 1.0
