"""Tests for the VietMed-Sum adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnc.datasets.vietmed_sum import VietMedSumAdapter
from tests.conftest import FIXTURE_TS

if TYPE_CHECKING:
    from pathlib import Path

_VALID_RAW = {"transcript": "sốt cao", "summary": "bị sốt"}
_NUM_TRAIN_SAMPLES = 2


def _adapter() -> VietMedSumAdapter:
    return VietMedSumAdapter()


def test_dataset_name() -> None:
    """Adapter reports correct dataset name."""
    assert _adapter().dataset_name() == "vietmed-sum"


def test_discover_splits_finds_all(vietmed_dir: Path) -> None:
    """All three whole-dialogue splits are discovered."""
    splits = _adapter().discover_splits(vietmed_dir)
    names = [s[0] for s in splits]
    assert names == ["train_whole", "dev_whole", "test_whole"]


def test_discover_splits_skips_missing(tmp_path: Path) -> None:
    """Empty directory yields no splits."""
    splits = _adapter().discover_splits(tmp_path)
    assert splits == []


def test_iter_raw_samples_yields_dicts(vietmed_dir: Path) -> None:
    """Raw samples are dicts with expected keys."""
    split_path = vietmed_dir / "train_whole.parquet"
    samples = list(_adapter().iter_raw_samples(split_path))
    assert len(samples) == _NUM_TRAIN_SAMPLES
    assert "transcript" in samples[0]
    assert "summary" in samples[0]


def test_validate_accepts_valid() -> None:
    """Non-empty transcript and summary passes validation."""
    assert _adapter().validate_raw_sample(_VALID_RAW)


def test_validate_rejects_empty_transcript() -> None:
    """Empty transcript fails validation."""
    assert not _adapter().validate_raw_sample(
        {"transcript": "", "summary": "xyz"},
    )


def test_validate_rejects_missing_summary() -> None:
    """Missing summary key fails validation."""
    assert not _adapter().validate_raw_sample({"transcript": "abc"})


def test_to_de1_record_produces_valid_record() -> None:
    """Converted record has all required DE-1 fields populated."""
    record = _adapter().to_de1_record(
        _VALID_RAW,
        split="train_whole",
        created_at=FIXTURE_TS,
    )
    assert record.source == "vietmed-sum"
    assert record.language == "vi"
    assert record.raw_text == "sốt cao"
    assert record.split == "train_whole"
    assert record.source_record_id is not None
    assert record.payload["transcript"] == "sốt cao"
    assert record.payload["summary"] == "bị sốt"
    assert record.normalized_text == ""
    assert record.sentences == []


def test_source_record_id_is_deterministic() -> None:
    """Same content always produces the same source_record_id."""
    r1 = _adapter().to_de1_record(
        _VALID_RAW,
        split="train_whole",
        created_at=FIXTURE_TS,
    )
    r2 = _adapter().to_de1_record(
        _VALID_RAW,
        split="train_whole",
        created_at=FIXTURE_TS,
    )
    assert r1.source_record_id == r2.source_record_id
    assert r1.doc_id == r2.doc_id


def test_doc_id_includes_source_record_id() -> None:
    """doc_id embeds the derived source_record_id."""
    record = _adapter().to_de1_record(
        {"transcript": "abc", "summary": "xyz"},
        split="dev_whole",
        created_at=FIXTURE_TS,
    )
    assert record.doc_id.startswith("vietmed-sum:")
    assert record.source_record_id in record.doc_id
