"""Tests for the ViHealthQA adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mnc.datasets.vihealthqa import ViHealthQAAdapter
from tests.conftest import FIXTURE_TS

if TYPE_CHECKING:
    from pathlib import Path

_TRAIN_SAMPLE = {
    "id": 42,
    "question": "Q?",
    "answer": "A.",
    "link": "https://x.com",
}
_NUM_TRAIN_SAMPLES = 2


def _adapter() -> ViHealthQAAdapter:
    return ViHealthQAAdapter()


def test_dataset_name() -> None:
    """Adapter reports correct dataset name."""
    assert _adapter().dataset_name() == "vihealthqa"


def test_discover_splits_finds_all(vihealthqa_dir: Path) -> None:
    """All three CSV splits are discovered."""
    splits = _adapter().discover_splits(vihealthqa_dir)
    names = [s[0] for s in splits]
    assert names == ["train", "validation", "test"]


def test_val_renamed_to_validation(vihealthqa_dir: Path) -> None:
    """val.csv is mapped to the validation split name."""
    splits = _adapter().discover_splits(vihealthqa_dir)
    split_names = [s[0] for s in splits]
    assert "validation" in split_names
    assert "val" not in split_names


def test_iter_raw_samples_yields_dicts(vihealthqa_dir: Path) -> None:
    """Raw samples are dicts with expected keys."""
    split_path = vihealthqa_dir / "train.csv"
    samples = list(_adapter().iter_raw_samples(split_path))
    assert len(samples) == _NUM_TRAIN_SAMPLES
    assert "question" in samples[0]
    assert "answer" in samples[0]


def test_validate_accepts_valid() -> None:
    """Non-empty question and answer passes validation."""
    assert _adapter().validate_raw_sample({"question": "Q?", "answer": "A."})


def test_validate_rejects_empty_answer() -> None:
    """Empty answer fails validation."""
    assert not _adapter().validate_raw_sample(
        {"question": "Q?", "answer": ""},
    )


def test_validate_rejects_missing_question() -> None:
    """Missing question key fails validation."""
    assert not _adapter().validate_raw_sample({"answer": "A."})


def test_to_de1_record_produces_valid_record() -> None:
    """Converted record has all required DE-1 fields populated."""
    record = _adapter().to_de1_record(
        _TRAIN_SAMPLE,
        split="train",
        created_at=FIXTURE_TS,
    )
    assert record.source == "vihealthqa"
    assert record.language == "vi"
    assert "Q?" in record.raw_text
    assert "A." in record.raw_text
    assert record.split == "train"
    assert record.source_record_id == "42"
    assert record.doc_id == "vihealthqa:train:42"
    assert record.payload["question"] == "Q?"
    assert record.payload["link"] == "https://x.com"


def test_native_id_preserved() -> None:
    """Source id field is carried through to source_record_id."""
    record = _adapter().to_de1_record(
        {"id": 7, "question": "Q?", "answer": "A.", "link": ""},
        split="test",
        created_at=FIXTURE_TS,
    )
    assert record.source_record_id == "7"


def test_missing_link_defaults_to_empty() -> None:
    """Missing link field defaults to empty string in payload."""
    record = _adapter().to_de1_record(
        {"id": 1, "question": "Q?", "answer": "A."},
        split="train",
        created_at=FIXTURE_TS,
    )
    assert record.payload["link"] == ""
