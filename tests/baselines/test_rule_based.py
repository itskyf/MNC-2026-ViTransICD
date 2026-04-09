"""Tests for BM-1 rule-based baseline."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

from mnc.baselines._rule_scoring import (
    aggregate_scores,
    prune_by_rules,
    rank_predictions,
)
from mnc.baselines.rule_based import _RunConfig, run_rule_based_baseline
from mnc.schemas.document import DocumentRecord
from mnc.schemas.mention import MentionRecord
from mnc.schemas.ontology import OntologyCode
from mnc.schemas.prediction import PredictionRecord
from mnc.schemas.rule import RuleRecord
from mnc.schemas.silver import SilverRecord


def _ts() -> datetime:
    return datetime.now(tz=UTC)


def _onto(code: str, title_vi: str, title_en: str = "") -> OntologyCode:
    search = f"{code} | {title_vi} | {title_en}"
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


def _mention(mention_id: str, doc_id: str, text: str) -> MentionRecord:
    return MentionRecord(
        mention_id=mention_id,
        doc_id=doc_id,
        text=text,
        normalized_text=text.lower(),
        mention_type="disease",
        char_start=0,
        char_end=len(text),
        created_at=_ts(),
    )


def _rule(code: str, topic: str = "exclude_note") -> RuleRecord:
    return RuleRecord(
        rule_id=f"rule:{code}",
        scope="code",
        code_3char=code,
        topic=topic,
        action="restrict",
        priority=1,
        evidence_text=f"Exclude {code}",
        normalized_text=f"exclude {code}",
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
# Rule scoring tests
# ---------------------------------------------------------------------------


class TestAggregateScores:
    def test_exact_title_match_highest(self) -> None:
        matches = [("A00", "exact_title", 1.0)]
        doc_matches = [("A00", 0.5)]
        scores = aggregate_scores(matches, doc_matches)
        assert "A00" in scores
        assert scores["A00"] == 1.0

    def test_repeated_evidence_increases_score(self) -> None:
        single = aggregate_scores([("A00", "exact_title", 1.0)], [])
        both = aggregate_scores(
            [("A00", "exact_title", 1.0)],
            [("A00", 0.8)],
        )
        assert both["A00"] >= single["A00"]

    def test_empty_inputs(self) -> None:
        scores = aggregate_scores([], [])
        assert scores == {}


class TestPruneByRules:
    def test_removes_excluded_code(self) -> None:
        scores = {"A00": 0.9, "B50": 0.8}
        rules = [_rule("A00")]
        pruned = prune_by_rules(scores, rules)
        assert "A00" not in pruned
        assert "B50" in pruned

    def test_no_rules_returns_unchanged(self) -> None:
        scores = {"A00": 0.9}
        pruned = prune_by_rules(scores, [])
        assert pruned == scores


class TestRankPredictions:
    def test_top_k(self) -> None:
        scores = {"A00": 0.9, "B50": 0.8, "C00": 0.7}
        predicted, _ = rank_predictions(scores, top_k=2)
        assert len(predicted) == 2
        assert predicted[0] == "A00"

    def test_tie_breaking_by_code(self) -> None:
        scores = {"B50": 0.8, "A00": 0.8}
        predicted, _ = rank_predictions(scores, top_k=10)
        assert predicted[0] == "A00"


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestRunRuleBasedBaseline:
    def _setup(self, tmp_path: Path) -> _RunConfig:
        """Create minimal silver and gold directories."""
        silver = tmp_path / "silver"
        gold = tmp_path / "gold"

        onto_dir = silver / "icd10_official_pdf" / "normalized_ontology"
        onto_dir.mkdir(parents=True)
        codes = [_onto("A00", "Bệnh tả", "Cholera"), _onto("B50", "Sốt rét", "Malaria")]
        onto_path = onto_dir / "ontology_codes.jsonl"
        _write_jsonl(onto_path, codes)

        doc_dir = silver / "test-ds" / "documents"
        docs = [_doc("d1", "Bệnh tả là bệnh truyền nhiễm")]
        _write_jsonl(doc_dir / "train.jsonl", docs)

        men_dir = silver / "test-ds" / "canonical_mentions"
        mentions = [_mention("d1:m1", "d1", "Bệnh tả")]
        _write_jsonl(men_dir / "train.jsonl", mentions)

        return _RunConfig(
            silver_dir=silver,
            gold_dir=gold,
            ontology_path=onto_path,
        )

    def test_exact_mention_ranks_correct_code(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        result = run_rule_based_baseline(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        assert len(result) > 0
        assert result[0].predicted_codes[0] == "A00"

    def test_output_validates_against_prediction_record(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        result = run_rule_based_baseline(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        for pred in result:
            assert isinstance(pred, PredictionRecord)
            assert pred.model_name == "bm_1_rule_based"

    def test_predictions_written_without_targets(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        run_rule_based_baseline(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        out = base_cfg.gold_dir / "test-ds" / "bm_1_rule_based"
        assert (out / "train.predictions.jsonl").exists()
        assert not (out / "train.metrics.json").exists()

    def test_metrics_written_with_targets(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        targets_path = base_cfg.silver_dir / "test-ds" / "targets.jsonl"
        _write_jsonl(targets_path, [_silver("d1", ["A00"])])

        run_rule_based_baseline(
            dataset_name="test-ds",
            split="train",
            config=_RunConfig(
                silver_dir=base_cfg.silver_dir,
                gold_dir=base_cfg.gold_dir,
                ontology_path=base_cfg.ontology_path,
                targets_path=targets_path,
            ),
        )
        out = base_cfg.gold_dir / "test-ds" / "bm_1_rule_based"
        assert (out / "train.metrics.json").exists()

    def test_manifest_generation(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        run_rule_based_baseline(
            dataset_name="test-ds",
            split="train",
            config=base_cfg,
        )
        out = base_cfg.gold_dir / "test-ds" / "bm_1_rule_based"
        manifest_path = out / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "train" in data["record_count_by_split"]

    def test_exclusion_pruning_removes_code(self, tmp_path: Path) -> None:
        base_cfg = self._setup(tmp_path)
        rules_dir = base_cfg.silver_dir / "icd10_official_pdf" / "coding_rules"
        rules_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(rules_dir / "rule_records.jsonl", [_rule("A00")])

        result = run_rule_based_baseline(
            dataset_name="test-ds",
            split="train",
            config=_RunConfig(
                silver_dir=base_cfg.silver_dir,
                gold_dir=base_cfg.gold_dir,
                ontology_path=base_cfg.ontology_path,
                rules_path=rules_dir / "rule_records.jsonl",
            ),
        )
        for pred in result:
            assert "A00" not in pred.predicted_codes
