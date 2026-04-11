"""Tests for DC-4 weak supervision aggregation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from pydantic import BaseModel

import pytest
from pydantic import ValidationError

from mnc.datasets.weak_supervision import (
    _MENTION_METHODS,
    aggregate_weak_labels,
    build_evidence_spans,
    compute_method_score,
    compute_rule_adjustment,
    group_candidates,
    validate_global_rules,
)
from mnc.schemas.candidate import CandidateLink
from mnc.schemas.document import DocumentRecord
from mnc.schemas.mention import MentionRecord
from mnc.schemas.rule import RuleRecord
from mnc.schemas.weak_label import WeakEvidenceSpan, WeakLabelRecord

_RULES_DIR = "icd10_official_pdf/coding_rules/rule_records.jsonl"


def _ts() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(tz=UTC)


def _doc(doc_id: str, text: str) -> DocumentRecord:
    """Create test DocumentRecord."""
    return DocumentRecord(
        doc_id=doc_id,
        source="test",
        language="vi",
        raw_text=text,
        normalized_text=text.lower(),
        retrieval_text=text.lower(),
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


def _link(**kwargs: object) -> CandidateLink:
    """Create test CandidateLink from keyword arguments."""
    defaults: dict[str, object] = {
        "mention_id": None,
        "char_start": None,
        "char_end": None,
        "created_at": _ts(),
    }
    defaults.update(kwargs)
    return CandidateLink(**defaults)  # type: ignore[arg-type]


def _rule(**kwargs: object) -> RuleRecord:
    """Create test RuleRecord from keyword arguments."""
    defaults: dict[str, object] = {
        "action": "note",
        "priority": 1,
        "source_page_start": None,
        "source_page_end": None,
        "created_at": _ts(),
    }
    # Set evidence_text/normalized_text from topic if not provided
    topic = kwargs.get("topic", "")
    defaults.setdefault("evidence_text", f"Test {topic}")
    defaults.setdefault("normalized_text", f"test {topic}")
    defaults.update(kwargs)
    return RuleRecord(**defaults)  # type: ignore[arg-type]


def _write_jsonl(path: Path, records: list[BaseModel]) -> None:
    """Write records as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")


def _rules_path(silver: Path) -> str:
    """Return rules path string under a temp silver dir."""
    return str(silver / _RULES_DIR)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_REQUIRED_GLOBAL_RULES = [
    _rule(rule_id="g1", scope="global", topic="principal_diagnosis", action="prefer"),
    _rule(rule_id="g2", scope="global", topic="symptom_fallback", action="fallback"),
    _rule(rule_id="g3", scope="global", topic="mortality_coding", action="restrict"),
    _rule(rule_id="g4", scope="global", topic="official_3char_policy", action="allow"),
]


def _setup_silver(
    tmp_path: Path,
    *,
    extra_rules: list[RuleRecord] | None = None,
) -> Path:
    """Create minimal silver directory with test data."""
    silver = tmp_path / "silver"
    dataset = silver / "test-ds"

    # Documents
    docs = [_doc("d1", "Bệnh nhân bị bệnh tả cấp tính")]
    _write_jsonl(dataset / "documents" / "train.jsonl", docs)

    # Mentions
    mentions = [
        _mention("d1:m1", "d1", "Bệnh tả", 0, 8),
        _mention("d1:m2", "d1", "cấp tính", 9, 17),
    ]
    _write_jsonl(dataset / "canonical_mentions" / "train.jsonl", mentions)

    # Candidate links
    links = [
        _link(
            doc_id="d1",
            code_3char="A00",
            method="exact",
            score=1.0,
            mention_id="d1:m1",
            char_start=0,
            char_end=8,
        ),
        _link(doc_id="d1", code_3char="A00", method="tfidf", score=0.8),
        _link(doc_id="d1", code_3char="J00", method="bm25", score=0.7),
    ]
    _write_jsonl(dataset / "candidate_links" / "train.jsonl", links)

    # Rules
    all_rules = _REQUIRED_GLOBAL_RULES + (extra_rules or [])
    _write_jsonl(silver / _RULES_DIR, all_rules)

    return silver


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------


class TestWeakEvidenceSpan:
    def test_valid_span(self) -> None:
        span = WeakEvidenceSpan(
            mention_id="m1",
            char_start=0,
            char_end=8,
            text="bệnh tả",
            methods=["exact"],
            score=1.0,
        )
        assert span.mention_id == "m1"
        assert span.score == 1.0

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            WeakEvidenceSpan(
                mention_id="m1",
                char_start=0,
                char_end=8,
                text="test",
                methods=["exact"],
                score=1.0,
                unknown="field",
            )


class TestWeakLabelRecord:
    def test_valid_record(self) -> None:
        rec = WeakLabelRecord(
            doc_id="d1",
            code_3char="A00",
            label_type="positive",
            confidence=0.85,
            rank=1,
            support_methods=["exact"],
            support_rule_ids=[],
            evidence_spans=[
                WeakEvidenceSpan(
                    mention_id="m1",
                    char_start=0,
                    char_end=8,
                    text="test",
                    methods=["exact"],
                    score=1.0,
                ),
            ],
            created_at=_ts(),
        )
        assert rec.label_type == "positive"
        assert rec.code_3char == "A00"

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            WeakLabelRecord(
                doc_id="d1",
                code_3char="A00",
                label_type="positive",
                confidence=0.85,
                rank=1,
                support_methods=["exact"],
                support_rule_ids=[],
                evidence_spans=[],
                created_at=_ts(),
                extra="bad",
            )


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


class TestComputeMethodScore:
    def test_exact_method_weight(self) -> None:
        links = [
            _link(
                doc_id="d1",
                code_3char="A00",
                method="exact",
                score=0.9,
                mention_id="m1",
            ),
        ]
        score = compute_method_score(
            links,
            frozenset({"exact", "normalized", "fuzzy"}),
        )
        assert score == pytest.approx(0.9)

    def test_normalized_method_weight(self) -> None:
        links = [
            _link(
                doc_id="d1",
                code_3char="A00",
                method="normalized",
                score=0.8,
                mention_id="m1",
            ),
        ]
        score = compute_method_score(links, _MENTION_METHODS)
        assert score == pytest.approx(0.8 * 0.90)


class TestComputeRuleAdjustment:
    def test_include_note_bonus(self) -> None:
        rules = {
            "A00": [
                _rule(
                    rule_id="r1",
                    scope="code",
                    topic="include_note",
                    code_3char="A00",
                ),
            ],
        }
        adj, ids = compute_rule_adjustment("A00", rules, mention_score=0.5)
        assert adj == pytest.approx(0.10)
        assert "r1" in ids

    def test_exclude_note_penalty_when_weak(self) -> None:
        rules = {
            "A00": [
                _rule(
                    rule_id="r1",
                    scope="code",
                    topic="exclude_note",
                    code_3char="A00",
                ),
            ],
        }
        adj, _ = compute_rule_adjustment("A00", rules, mention_score=0.5)
        assert adj == pytest.approx(-0.25)

    def test_exclude_note_no_penalty_when_strong(self) -> None:
        rules = {
            "A00": [
                _rule(
                    rule_id="r1",
                    scope="code",
                    topic="exclude_note",
                    code_3char="A00",
                ),
            ],
        }
        adj, _ = compute_rule_adjustment("A00", rules, mention_score=0.95)
        assert adj == pytest.approx(0.0)

    def test_code_first_penalty_when_no_mention(self) -> None:
        rules = {
            "A00": [
                _rule(
                    rule_id="r1",
                    scope="code",
                    topic="code_first",
                    action="code_first",
                    code_3char="A00",
                ),
            ],
        }
        adj, _ = compute_rule_adjustment("A00", rules, mention_score=0.0)
        assert adj == pytest.approx(-0.10)


# ---------------------------------------------------------------------------
# Evidence span tests
# ---------------------------------------------------------------------------


class TestBuildEvidenceSpans:
    def test_deduplicates_and_merges(self) -> None:
        candidates = [
            _link(
                doc_id="d1",
                code_3char="A00",
                method="exact",
                score=1.0,
                mention_id="m1",
                char_start=0,
                char_end=8,
            ),
            _link(
                doc_id="d1",
                code_3char="A00",
                method="normalized",
                score=0.8,
                mention_id="m1",
                char_start=0,
                char_end=8,
            ),
        ]
        mentions = {"m1": _mention("m1", "d1", "Bệnh tả", 0, 8)}
        spans = build_evidence_spans(candidates, mentions)
        assert len(spans) == 1
        assert "exact" in spans[0].methods
        assert "normalized" in spans[0].methods
        # max(1.0*1.0, 0.9*0.8) = 1.0
        assert spans[0].score == 1.0

    def test_uses_normalized_text(self) -> None:
        candidates = [
            _link(
                doc_id="d1",
                code_3char="A00",
                method="exact",
                score=1.0,
                mention_id="m1",
                char_start=0,
                char_end=8,
            ),
        ]
        mentions = {
            "m1": MentionRecord(
                mention_id="m1",
                doc_id="d1",
                text="Bệnh Tả",
                normalized_text="bệnh tả",
                mention_type="disease",
                char_start=0,
                char_end=8,
                created_at=_ts(),
            ),
        }
        spans = build_evidence_spans(candidates, mentions)
        assert spans[0].text == "bệnh tả"


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateGlobalRules:
    def test_passes_with_all_required(self) -> None:
        validate_global_rules(_REQUIRED_GLOBAL_RULES)

    def test_fails_on_missing_topic(self) -> None:
        rules = [
            _rule(
                rule_id="g1",
                scope="global",
                topic="principal_diagnosis",
                action="prefer",
            ),
            _rule(
                rule_id="g2",
                scope="global",
                topic="symptom_fallback",
                action="fallback",
            ),
            # missing mortality_coding and official_3char_policy
        ]
        with pytest.raises(ValueError, match="Missing required global"):
            validate_global_rules(rules)


# ---------------------------------------------------------------------------
# Grouper tests
# ---------------------------------------------------------------------------


class TestGroupCandidates:
    def test_groups_by_doc_and_code(self) -> None:
        links = [
            _link(
                doc_id="d1",
                code_3char="A00",
                method="exact",
                score=1.0,
                mention_id="m1",
            ),
            _link(doc_id="d1", code_3char="A00", method="tfidf", score=0.5),
            _link(
                doc_id="d1",
                code_3char="B50",
                method="exact",
                score=0.9,
                mention_id="m2",
            ),
        ]
        groups = group_candidates(links)
        assert len(groups) == 2
        assert len(groups[("d1", "A00")]) == 2
        assert len(groups[("d1", "B50")]) == 1


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestAggregateWeakLabels:
    def test_mention_exact_match_emits_positive(self, tmp_path: Path) -> None:
        silver = _setup_silver(tmp_path)
        result = aggregate_weak_labels(
            dataset_name="test-ds",
            split="train",
            silver_dir=str(silver),
            rules_path=_rules_path(silver),
        )
        positive = [r for r in result if r.code_3char == "A00"]
        assert len(positive) == 1
        assert positive[0].label_type == "positive"

    def test_document_only_does_not_emit(self, tmp_path: Path) -> None:
        silver = _setup_silver(tmp_path)
        result = aggregate_weak_labels(
            dataset_name="test-ds",
            split="train",
            silver_dir=str(silver),
            rules_path=_rules_path(silver),
        )
        codes = {r.code_3char for r in result}
        # J00 only has bm25 (document-only)
        assert "J00" not in codes

    def test_exclude_note_suppresses_weak_label(self, tmp_path: Path) -> None:
        """Penalty of -0.25 on weak mention suppresses below threshold."""
        silver = tmp_path / "silver"
        dataset = silver / "test-ds"

        docs = [_doc("d1", "Bệnh nhân bị bệnh tả và viêm họng")]
        _write_jsonl(dataset / "documents" / "train.jsonl", docs)

        mentions = [
            _mention("d1:m1", "d1", "Bệnh tả", 0, 8),
            _mention("d1:m2", "d1", "viêm họng", 13, 22),
        ]
        _write_jsonl(
            dataset / "canonical_mentions" / "train.jsonl",
            mentions,
        )

        # A00: normalized 0.68 -> weighted = 0.612 -> mention_score = 0.612
        # With exclude -0.25: conf = 0.362 < 0.50
        # J00: exact 1.0 -> passes
        links = [
            _link(
                doc_id="d1",
                code_3char="A00",
                method="normalized",
                score=0.68,
                mention_id="d1:m1",
                char_start=0,
                char_end=8,
            ),
            _link(
                doc_id="d1",
                code_3char="J00",
                method="exact",
                score=1.0,
                mention_id="d1:m2",
                char_start=13,
                char_end=22,
            ),
        ]
        _write_jsonl(dataset / "candidate_links" / "train.jsonl", links)

        extra_rules = [
            _rule(rule_id="r_ex", scope="code", topic="exclude_note", code_3char="A00"),
        ]
        all_rules = _REQUIRED_GLOBAL_RULES + extra_rules
        _write_jsonl(silver / _RULES_DIR, all_rules)

        result = aggregate_weak_labels(
            dataset_name="test-ds",
            split="train",
            silver_dir=str(silver),
            rules_path=_rules_path(silver),
        )
        # A00 suppressed by exclude_note penalty
        a00 = [r for r in result if r.code_3char == "A00"]
        assert len(a00) == 0
        # J00 still passes
        j00 = [r for r in result if r.code_3char == "J00"]
        assert len(j00) == 1

    def test_include_note_preserves_label(self, tmp_path: Path) -> None:
        """include_note bonus helps marginal label meet threshold."""
        silver = tmp_path / "silver"
        dataset = silver / "test-ds"

        docs = [_doc("d1", "Bệnh nhân bị bệnh tả")]
        _write_jsonl(dataset / "documents" / "train.jsonl", docs)

        mentions = [_mention("d1:m1", "d1", "Bệnh tả", 0, 8)]
        _write_jsonl(
            dataset / "canonical_mentions" / "train.jsonl",
            mentions,
        )

        # fuzzy 0.55 -> weighted = 0.80 * 0.55 = 0.44 -> mention_score >= 0.40
        # Without bonus: conf = 0.44 < 0.50
        # With include_note +0.10: conf = 0.54 >= 0.50
        links = [
            _link(
                doc_id="d1",
                code_3char="A00",
                method="fuzzy",
                score=0.55,
                mention_id="d1:m1",
                char_start=0,
                char_end=8,
            ),
        ]
        _write_jsonl(dataset / "candidate_links" / "train.jsonl", links)

        extra_rules = [
            _rule(
                rule_id="r_inc",
                scope="code",
                topic="include_note",
                code_3char="A00",
            ),
        ]
        all_rules = _REQUIRED_GLOBAL_RULES + extra_rules
        _write_jsonl(silver / _RULES_DIR, all_rules)

        result = aggregate_weak_labels(
            dataset_name="test-ds",
            split="train",
            silver_dir=str(silver),
            rules_path=_rules_path(silver),
        )
        a00 = [r for r in result if r.code_3char == "A00"]
        assert len(a00) == 1

    def test_rank_ordering_deterministic(self, tmp_path: Path) -> None:
        """Two runs produce identical rank ordering."""
        silver = _setup_silver(tmp_path)
        kwargs = {
            "dataset_name": "test-ds",
            "split": "train",
            "silver_dir": str(silver),
            "rules_path": _rules_path(silver),
        }
        result1 = aggregate_weak_labels(**kwargs)
        result2 = aggregate_weak_labels(**kwargs)
        codes1 = [(r.doc_id, r.rank, r.code_3char) for r in result1]
        codes2 = [(r.doc_id, r.rank, r.code_3char) for r in result2]
        assert codes1 == codes2

    def test_missing_global_topic_fails(self, tmp_path: Path) -> None:
        silver = tmp_path / "silver"
        dataset = silver / "test-ds"

        docs = [_doc("d1", "test")]
        _write_jsonl(dataset / "documents" / "train.jsonl", docs)
        mentions = [_mention("d1:m1", "d1", "test", 0, 4)]
        _write_jsonl(
            dataset / "canonical_mentions" / "train.jsonl",
            mentions,
        )
        links = [
            _link(
                doc_id="d1",
                code_3char="A00",
                method="exact",
                score=1.0,
                mention_id="d1:m1",
                char_start=0,
                char_end=4,
            ),
        ]
        _write_jsonl(dataset / "candidate_links" / "train.jsonl", links)

        # Only partial global rules (missing required topics)
        partial_rules = [
            _rule(
                rule_id="g1",
                scope="global",
                topic="principal_diagnosis",
                action="prefer",
            ),
        ]
        _write_jsonl(silver / _RULES_DIR, partial_rules)

        with pytest.raises(ValueError, match="Missing required global"):
            aggregate_weak_labels(
                dataset_name="test-ds",
                split="train",
                silver_dir=str(silver),
                rules_path=_rules_path(silver),
            )

    def test_jsonl_writer_output(self, tmp_path: Path) -> None:
        """Output JSONL validates against WeakLabelRecord."""
        silver = _setup_silver(tmp_path)
        aggregate_weak_labels(
            dataset_name="test-ds",
            split="train",
            silver_dir=str(silver),
            rules_path=_rules_path(silver),
        )
        output_path = silver / "test-ds" / "weak_labels" / "train.jsonl"
        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as f:
            for raw_line in f:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                rec = WeakLabelRecord.model_validate_json(stripped)
                assert rec.label_type == "positive"
                assert 0.0 <= rec.confidence <= 1.0
                assert rec.rank >= 1
                assert len(rec.evidence_spans) > 0
                assert rec.code_3char[0].isupper()

    def test_manifest_generation(self, tmp_path: Path) -> None:
        silver = _setup_silver(tmp_path)
        aggregate_weak_labels(
            dataset_name="test-ds",
            split="train",
            silver_dir=str(silver),
            rules_path=_rules_path(silver),
        )
        manifest_path = silver / "test-ds" / "weak_labels" / "manifest.json"
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert data["dataset"] == "test-ds"
        assert "train" in data["record_count_by_split"]
