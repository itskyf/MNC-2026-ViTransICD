"""Tests for alias dictionary extraction."""

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from mnc.datasets.alias_dictionary import (
    build_icd10_alias_dictionary,
    extract_nos_form,
    normalize_alias,
)
from mnc.schemas.alias import AliasRecord


def test_alias_record_schema() -> None:
    """Test schema validation for one AliasRecord."""
    record = AliasRecord(
        alias_id="A00:test",
        code_3char="A00",
        alias="Test (NOS)",
        alias_norm="test",
        alias_type="title_en",
        language="en",
        match_level="exact",
        created_at=datetime.now(tz=UTC),
    )
    assert record.alias_id == "A00:test"

    with pytest.raises(ValidationError):
        AliasRecord(
            alias_id="A00:test",
            code_3char="A00",
            alias="Test",
            alias_norm="test",
            alias_type="invalid_type",
            language="en",
            created_at=datetime.now(tz=UTC),
        )


def test_normalize_alias() -> None:
    """Test alias normalization."""
    assert normalize_alias("Hello  World. ") == "hello world"
    assert normalize_alias("Cholera (NOS)") == "cholera (nos)"
    assert normalize_alias("Cholera;") == "cholera"


def test_extract_nos_form() -> None:
    """Test NOS/KXĐK extraction."""
    assert extract_nos_form("Something NOS something") == "NOS"
    assert extract_nos_form("Something kxđk") == "KXĐK"
    assert extract_nos_form("Nothing here") is None


def test_build_icd10_alias_dictionary(tmp_path: Path) -> None:
    """Test full extraction logic on mocked data."""
    ontology_path = tmp_path / "ontology.jsonl"
    bronze_path = tmp_path / "bronze.jsonl"
    output_dir = tmp_path / "output"

    onto_text = (
        '{"code_3char":"A00","title_vi":"Bệnh tả (Cholera)","title_en":"Cholera",'
        '"search_text":"","created_at":"2026-04-09T00:00:00"}\n'
        '{"code_3char":"A01","title_vi":"Bệnh thương hàn","title_en":"Typhoid fever",'
        '"search_text":"","created_at":"2026-04-09T00:00:00"}\n'
    )
    ontology_path.write_text(onto_text, encoding="utf-8")

    bronze_text = (
        '{"doc_id":"p1","source":"pdf","language":"vi",'
        '"raw_text":"A00 Bệnh tả\\nBao gồm: bệnh tả cổ điển\\nLoại trừ: viêm ruột\\n'
        'A01 Bệnh thương hàn\\n(bệnh thương hàn kxđk)\\n",'
        '"created_at":"2026-04-09T00:00:00","payload":{"page_no":1}}\n'
    )
    bronze_path.write_text(bronze_text, encoding="utf-8")

    records = build_icd10_alias_dictionary(ontology_path, bronze_path, output_dir)

    assert len(records) > 0
    ids = [r.alias_id for r in records]
    assert len(ids) == len(set(ids))

    # Parenthetical 'Cholera' will be deduplicated into 'title_en' since it matches.
    assert any(
        r.code_3char == "A00"
        and r.alias_norm == "cholera"
        and r.alias_type == "title_en"
        for r in records
    )

    assert any(
        r.code_3char == "A00"
        and r.alias == "bệnh tả cổ điển"
        and r.alias_type == "inclusion"
        for r in records
    )

    assert any(
        r.code_3char == "A01" and "kxđk" in r.alias_norm and r.alias_type == "nos_form"
        for r in records
    )

    assert not any("viêm ruột" in r.alias_norm for r in records)
    assert (output_dir / "alias_records.jsonl").exists()
    assert (output_dir / "alias_records.csv").exists()
