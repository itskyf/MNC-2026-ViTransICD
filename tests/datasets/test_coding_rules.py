"""Tests for coding rules extraction."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mnc.datasets.coding_rules import (
    extract_icd10_coding_rules,
    hash_text,
    normalize_rule_text,
)
from mnc.schemas.rule import RuleRecord


def test_rule_record_schema() -> None:
    """Test schema validation for one RuleRecord."""
    record = RuleRecord(
        rule_id="global:principal_diagnosis",
        scope="global",
        topic="principal_diagnosis",
        action="prefer",
        priority=90,
        evidence_text="Always prefer principal.",
        normalized_text="always prefer principal.",
        created_at=datetime.now(tz=UTC),
    )
    assert record.rule_id == "global:principal_diagnosis"

    with pytest.raises(ValidationError):
        RuleRecord(
            rule_id="global:principal_diagnosis",
            scope="global",
            topic="invalid_topic",
            action="prefer",
            priority=90,
            evidence_text="Text",
            normalized_text="text",
            created_at=datetime.now(tz=UTC),
        )


def test_hash_text() -> None:
    """Test stable ID generation."""
    assert hash_text("hello") == hash_text("hello")
    assert hash_text("hello") != hash_text("world")
    assert len(hash_text("hello")) == 8


def test_normalize_rule_text() -> None:
    """Test rule text normalization."""
    assert normalize_rule_text("This   is \n a test.") == "This is a test."


def test_extract_icd10_coding_rules(tmp_path: Path) -> None:
    """Test full extraction logic on mocked data."""
    intro_path = tmp_path / "intro.jsonl"
    ontology_path = tmp_path / "ontology.jsonl"
    bronze_path = tmp_path / "bronze.jsonl"
    output_dir = tmp_path / "output"

    intro_text = (
        '{"doc_id":"d1:principal_diagnosis","source":"pdf","language":"vi",'
        '"raw_text":"P rule","created_at":"2026-04-09T00:00:00"}\n'
        '{"doc_id":"d2:symptom_fallback","source":"pdf","language":"vi",'
        '"raw_text":"S rule","created_at":"2026-04-09T00:00:00"}\n'
        '{"doc_id":"d3:mortality_coding","source":"pdf","language":"vi",'
        '"raw_text":"M rule","created_at":"2026-04-09T00:00:00"}\n'
        '{"doc_id":"d4:official_3char_policy","source":"pdf","language":"vi",'
        '"raw_text":"3 policy","created_at":"2026-04-09T00:00:00"}\n'
    )
    intro_path.write_text(intro_text, encoding="utf-8")

    onto_text = (
        '{"code_3char":"A00","title_vi":"Bệnh tả","title_en":"Cholera",'
        '"search_text":"","created_at":"2026-04-09T00:00:00"}\n'
        '{"code_3char":"A01","title_vi":"Bệnh thương hàn","title_en":"Typhoid fever",'
        '"search_text":"","created_at":"2026-04-09T00:00:00"}\n'
    )
    ontology_path.write_text(onto_text, encoding="utf-8")

    bronze_text = (
        '{"doc_id":"p1","source":"pdf","language":"vi","raw_text":"'
        "A00 Bệnh tả\\nBao gồm: bệnh tả cổ điển\\nLoại trừ: viêm ruột\\n"
        'A01 Bệnh thương hàn\\nSử dụng mã thêm\\nCode first",'
        '"created_at":"2026-04-09T00:00:00","payload":{"page_no":1}}\n'
    )
    bronze_path.write_text(bronze_text, encoding="utf-8")

    rules = extract_icd10_coding_rules(
        intro_path,
        ontology_path,
        bronze_path,
        output_dir,
    )

    assert len(rules) > 4
    global_rules = [r for r in rules if r.scope == "global"]
    assert len(global_rules) == 4

    code_rules = [r for r in rules if r.scope == "code"]
    assert any(r.code_3char == "A00" and r.topic == "include_note" for r in code_rules)
    assert any(r.code_3char == "A00" and r.topic == "exclude_note" for r in code_rules)
    assert any(
        r.code_3char == "A01" and r.topic == "use_additional_code" for r in code_rules
    )
    assert any(r.code_3char == "A01" and r.topic == "code_first" for r in code_rules)

    ids = [r.rule_id for r in rules]
    assert len(ids) == len(set(ids))

    assert (output_dir / "rule_records.jsonl").exists()
    assert (output_dir / "heuristic_templates.json").exists()


def test_missing_global_topics(tmp_path: Path) -> None:
    """Test failure on missing required global topics."""
    intro_path = tmp_path / "intro.jsonl"
    ontology_path = tmp_path / "ontology.jsonl"
    bronze_path = tmp_path / "bronze.jsonl"
    output_dir = tmp_path / "output"

    intro_text = (
        '{"doc_id":"d1:principal_diagnosis","source":"pdf","language":"vi",'
        '"raw_text":"P rule","created_at":"2026-04-09T00:00:00"}\n'
    )
    intro_path.write_text(intro_text, encoding="utf-8")

    onto_text = (
        '{"code_3char":"A00","title_vi":"Bệnh tả","title_en":"Cholera",'
        '"search_text":"","created_at":"2026-04-09T00:00:00"}\n'
    )
    ontology_path.write_text(onto_text, encoding="utf-8")

    bronze_text = (
        '{"doc_id":"p1","source":"pdf","language":"vi",'
        '"raw_text":"A00 Bệnh tả\\nBao gồm: bệnh tả cổ điển",'
        '"created_at":"2026-04-09T00:00:00","payload":{"page_no":1}}\n'
    )
    bronze_path.write_text(bronze_text, encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required global topic"):
        extract_icd10_coding_rules(intro_path, ontology_path, bronze_path, output_dir)
