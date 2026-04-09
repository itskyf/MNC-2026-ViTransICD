"""Tests for BM-2 TF-IDF/BM25 baselines."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

from mnc.baselines._bm25 import build_bm25_index
from mnc.baselines._vectorizer import build_tfidf_index, query_tfidf
from mnc.baselines.tfidf_bm25 import _RunConfig, run_tfidf_bm25_baselines
from mnc.schemas.document import DocumentRecord
from mnc.schemas.ontology import OntologyCode
from mnc.schemas.prediction import PredictionRecord
from mnc.schemas.silver import SilverRecord


def _ts() -> datetime:
    return datetime.now(tz=UTC)


def _onto(
    code: str,
    title_vi: str,
    title_en: str = "",
    search_text: str = "",
) -> OntologyCode:
    search = search_text or f"{code} | {title_vi} | {title_en}"
    return OntologyCode(
        code_3char=code,
        chapter_id="Ch-1",
        title_vi=title_vi,
        title_en=title_en or None,
        aliases=[],
        search_text=search,
        created_at=_ts(),
    )


def _doc(doc_id: str, text: str, retrieval_text: str = "") -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        source="test",
        language="vi",
        raw_text=text,
        normalized_text=text,
        retrieval_text=retrieval_text or text.lower(),
        created_at=_ts(),
    )


def _silver(doc_id: str, labels: list[str]) -> SilverRecord:
    return SilverRecord(
        doc_id=doc_id,
        label_granularity="code_3char",
        silver_labels=labels,
        candidate_codes=labels,
        split="train",
        created_at=_ts(),
    )


def _write_jsonl(path: Path, records: list[BaseModel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


# ---------------------------------------------------------------------------
# TF-IDF unit tests
# ---------------------------------------------------------------------------


class TestTfidfVectorizer:
    def test_builds_index(self) -> None:
        codes = [
            _onto("A00", "Bệnh tả", "Cholera"),
            _onto("B50", "Sốt rét", "Malaria"),
        ]
        _vectorizer, matrix, code_list = build_tfidf_index(codes)
        assert code_list == ["A00", "B50"]
        assert matrix.shape[0] == 2

    def test_query_returns_ranked_results(self) -> None:
        codes = [
            _onto("A00", "Bệnh tả", "Cholera"),
            _onto("B50", "Sốt rét", "Malaria"),
        ]
        vectorizer, matrix, code_list = build_tfidf_index(codes)
        results = query_tfidf(vectorizer, matrix, code_list, "cholera bệnh tả", top_k=5)
        assert len(results) > 0
        assert results[0][0] == "A00"

    def test_empty_query(self) -> None:
        codes = [_onto("A00", "Bệnh tả")]
        vectorizer, matrix, code_list = build_tfidf_index(codes)
        results = query_tfidf(vectorizer, matrix, code_list, "", top_k=5)
        assert results == []


# ---------------------------------------------------------------------------
# BM25 unit tests
# ---------------------------------------------------------------------------


class TestBM25Index:
    def test_query_returns_results(self) -> None:
        codes = [
            _onto("A00", "Bệnh tả", "Cholera"),
            _onto("B50", "Sốt rét", "Malaria"),
        ]
        index = build_bm25_index(codes)
        results = index.query("cholera", top_k=5)
        assert len(results) > 0

    def test_empty_query(self) -> None:
        codes = [_onto("A00", "Bệnh tả")]
        index = build_bm25_index(codes)
        results = index.query("", top_k=5)
        assert results == []

    def test_scores_normalized(self) -> None:
        codes = [_onto("A00", "Bệnh tả"), _onto("B50", "Sốt rét")]
        index = build_bm25_index(codes)
        results = index.query("bệnh tả", top_k=5)
        for _, score in results:
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRunTfidfBm25Baselines:
    def _setup(self, tmp_path: Path) -> _RunConfig:
        """Create minimal silver and gold directories."""
        silver = tmp_path / "silver"
        gold = tmp_path / "gold"

        onto_dir = silver / "icd10_official_pdf" / "normalized_ontology"
        onto_dir.mkdir(parents=True)
        codes = [
            _onto("A00", "Bệnh tả", "Cholera"),
            _onto("B50", "Sốt rét", "Malaria"),
        ]
        onto_path = onto_dir / "ontology_codes.jsonl"
        _write_jsonl(onto_path, codes)

        doc_dir = silver / "test-ds" / "documents"
        docs = [
            _doc("d1", "Bệnh tả là bệnh truyền nhiễm", "bệnh tả là bệnh truyền nhiễm"),
        ]
        _write_jsonl(doc_dir / "train.jsonl", docs)

        return _RunConfig(
            silver_dir=silver,
            gold_dir=gold,
            ontology_path=onto_path,
        )

    def test_tfidf_predicts_correct_code(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        result = run_tfidf_bm25_baselines(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        assert len(result["tfidf"]) > 0
        assert result["tfidf"][0].predicted_codes[0] == "A00"

    def test_bm25_predicts_correct_code(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        result = run_tfidf_bm25_baselines(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        assert len(result["bm25"]) > 0

    def test_retrieval_text_fallback(self, tmp_path: Path) -> None:
        """When retrieval_text is empty, normalized_text is used."""
        base_cfg = self._setup(tmp_path)
        doc_dir = base_cfg.silver_dir / "test-ds" / "documents"
        docs = [_doc("d2", "Bệnh tả", retrieval_text="")]
        _write_jsonl(doc_dir / "train.jsonl", docs)

        result = run_tfidf_bm25_baselines(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        assert len(result["tfidf"]) > 0

    def test_output_validates_against_prediction_record(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        result = run_tfidf_bm25_baselines(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        for variant in ("tfidf", "bm25"):
            for pred in result[variant]:
                assert isinstance(pred, PredictionRecord)

    def test_predictions_written_without_targets(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        run_tfidf_bm25_baselines(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        out = base_cfg.gold_dir / "test-ds" / "bm_2_tfidf_bm25"
        assert (out / "train.tfidf.predictions.jsonl").exists()
        assert (out / "train.bm25.predictions.jsonl").exists()

    def test_metrics_written_with_targets(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        targets_path = base_cfg.silver_dir / "test-ds" / "targets.jsonl"
        _write_jsonl(targets_path, [_silver("d1", ["A00"])])

        run_tfidf_bm25_baselines(
            dataset_name="test-ds",
            split="train",
            config=_RunConfig(
                silver_dir=base_cfg.silver_dir,
                gold_dir=base_cfg.gold_dir,
                ontology_path=base_cfg.ontology_path,
                targets_path=targets_path,
            ),
        )
        out = base_cfg.gold_dir / "test-ds" / "bm_2_tfidf_bm25"
        assert (out / "train.tfidf.metrics.json").exists()
        assert (out / "train.bm25.metrics.json").exists()

    def test_deterministic_ordering(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        result1 = run_tfidf_bm25_baselines(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        result2 = run_tfidf_bm25_baselines(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        codes1 = [p.predicted_codes for p in result1["tfidf"]]
        codes2 = [p.predicted_codes for p in result2["tfidf"]]
        assert codes1 == codes2

    def test_manifest_generation(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        run_tfidf_bm25_baselines(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        out = base_cfg.gold_dir / "test-ds" / "bm_2_tfidf_bm25"
        manifest_path = out / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "train.tfidf" in data["record_count_by_split"]
