"""Unit tests for DC-2: abbreviation normalization."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from mnc.datasets.abbrev import (
    abbrev_dataset,
    find_definitions,
    load_seed_dict,
    normalize_abbreviations,
)
from mnc.schemas.manifest import BronzeManifest
from mnc.schemas.mention import MentionRecord

if TYPE_CHECKING:
    from pathlib import Path


def _ts() -> datetime:
    return datetime.now(tz=UTC)


def _mention(
    text: str,
    mention_type: str = "abbreviation",
    doc_id: str = "doc:1",
    char_start: int = 0,
    char_end: int = 3,
) -> MentionRecord:
    return MentionRecord(
        mention_id=f"{doc_id}:m:{char_start}:{char_end}",
        doc_id=doc_id,
        text=text,
        normalized_text=text.lower(),
        mention_type=mention_type,  # type: ignore[arg-type]
        char_start=char_start,
        char_end=char_end,
        confidence=None,
        created_at=_ts(),
    )


# ---------------------------------------------------------------------------
# Seed dictionary
# ---------------------------------------------------------------------------


class TestSeedDictionary:
    def test_load_seed_dict(self) -> None:
        seed = load_seed_dict()
        assert isinstance(seed, dict)
        assert len(seed) > 0

    def test_known_expansion(self) -> None:
        seed = load_seed_dict()
        assert seed["HIV"] == "virus suy giảm miễn dịch ở người"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        seed = load_seed_dict(tmp_path / "nonexistent.json")
        assert seed == {}


# ---------------------------------------------------------------------------
# Definitional pattern extraction
# ---------------------------------------------------------------------------


class TestFindDefinitions:
    def test_full_form_paren_abbr(self) -> None:
        text = "virus suy giảm miễn dịch ở người (HIV) gây bệnh"
        defs = find_definitions(text)
        assert "HIV" in defs
        assert defs["HIV"] == "virus suy giảm miễn dịch ở người"

    def test_abbr_paren_full_form(self) -> None:
        text = "HIV (virus suy giảm miễn dịch ở người) gây bệnh"
        defs = find_definitions(text)
        assert "HIV" in defs

    def test_multiple_definitions_same_abbr_ambiguous(self) -> None:
        text = "virus X (HIV) và bệnh Y (HIV) khác nhau"
        defs = find_definitions(text)
        # Both expansions are unique, so ambiguous → excluded
        assert "HIV" not in defs

    def test_no_definitions_found(self) -> None:
        text = "Bệnh nhân bị viêm gan B cần điều trị."
        defs = find_definitions(text)
        assert isinstance(defs, dict)


# ---------------------------------------------------------------------------
# Abbreviation normalization
# ---------------------------------------------------------------------------


class TestNormalizeAbbreviations:
    def test_seed_expansion(self) -> None:
        mentions = [_mention("HIV")]
        seed = {"HIV": "virus suy giảm miễn dịch ở người"}
        result = normalize_abbreviations(mentions, "text about HIV", seed)
        assert result[0].normalized_text == "virus suy giảm miễn dịch ở người"

    def test_in_document_expansion(self) -> None:
        mentions = [_mention("HIV")]
        raw = "virus suy giảm miễn dịch ở người (HIV) gây bệnh"
        result = normalize_abbreviations(mentions, raw, {})
        assert result[0].normalized_text == "virus suy giảm miễn dịch ở người"

    def test_in_document_overrides_seed(self) -> None:
        mentions = [_mention("HIV")]
        raw = "definitional HIV (human immunodeficiency virus) here"
        seed = {"HIV": "seed expansion"}
        result = normalize_abbreviations(mentions, raw, seed)
        # In-document wins
        assert result[0].normalized_text == "human immunodeficiency virus"

    def test_unresolved_preserves_original(self) -> None:
        mentions = [_mention("XYZ", char_start=0, char_end=3)]
        result = normalize_abbreviations(mentions, "some text", {})
        assert result[0].normalized_text == "xyz"  # DC-1 normalized form

    def test_ambiguous_remains_unresolved(self) -> None:
        mentions = [_mention("HIV")]
        raw = "thing A (HIV) and thing B (HIV) are different"
        result = normalize_abbreviations(mentions, raw, {})
        assert result[0].normalized_text == "hiv"

    def test_non_abbreviation_mentions_unchanged(self) -> None:
        mentions = [
            _mention("viêm gan B", mention_type="disease", char_start=0, char_end=9),
        ]
        result = normalize_abbreviations(mentions, "some text", {"HIV": "expansion"})
        assert result[0].mention_type == "disease"
        assert result[0].normalized_text == "viêm gan b"

    def test_same_document_only(self) -> None:
        m1 = _mention("HIV", doc_id="doc:1")
        m2 = _mention("HIV", doc_id="doc:2")
        # Definition in doc:1 text only
        raw = "virus suy giảm miễn dịch ở người (HIV)"
        result1 = normalize_abbreviations([m1], raw, {})
        result2 = normalize_abbreviations([m2], "no definitions here", {})
        assert result1[0].normalized_text == "virus suy giảm miễn dịch ở người"
        assert result2[0].normalized_text == "hiv"  # unresolved

    def test_preserves_provenance(self) -> None:
        mentions = [_mention("HIV", char_start=10, char_end=13)]
        seed = {"HIV": "virus suy giảm miễn dịch ở người"}
        result = normalize_abbreviations(mentions, "text", seed)
        r = result[0]
        assert r.mention_id == mentions[0].mention_id
        assert r.doc_id == mentions[0].doc_id
        assert r.text == "HIV"
        assert r.char_start == 10
        assert r.char_end == 13


# ---------------------------------------------------------------------------
# DC-2 pipeline integration
# ---------------------------------------------------------------------------


class TestAbbrevDataset:
    def _setup_silver(self, tmp_path: Path) -> Path:
        """Create minimal silver directory with DC-1 outputs."""
        silver = tmp_path / "silver" / "test-ds"

        # Documents
        doc_dir = silver / "documents"
        doc_dir.mkdir(parents=True, exist_ok=True)
        with (doc_dir / "train.jsonl").open("w") as f:
            f.write(
                json.dumps(
                    {
                        "doc_id": "doc:1",
                        "source": "test",
                        "language": "vi",
                        "raw_text": "virus suy giảm miễn dịch ở người (HIV)",
                        "normalized_text": "virus suy giảm miễn dịch ở người (HIV)",
                        "retrieval_text": "",
                        "sentences": [],
                        "created_at": datetime.now(tz=UTC).isoformat(),
                    },
                )
                + "\n",
            )

        # Mentions
        ment_dir = silver / "mentions"
        ment_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(tz=UTC).isoformat()
        with (ment_dir / "train.jsonl").open("w") as f:
            f.write(
                json.dumps(
                    {
                        "mention_id": "doc:1:m:42:45",
                        "doc_id": "doc:1",
                        "text": "HIV",
                        "normalized_text": "hiv",
                        "mention_type": "abbreviation",
                        "char_start": 42,
                        "char_end": 45,
                        "confidence": None,
                        "created_at": ts,
                    },
                )
                + "\n",
            )

        return tmp_path / "silver"

    def test_manifest_generation(self, tmp_path: Path) -> None:
        silver_dir = self._setup_silver(tmp_path)
        manifest = abbrev_dataset("test-ds", silver_dir)
        assert isinstance(manifest, BronzeManifest)
        assert "train" in manifest.record_count_by_split

    def test_output_validates_against_mention_record(self, tmp_path: Path) -> None:
        silver_dir = self._setup_silver(tmp_path)
        abbrev_dataset("test-ds", silver_dir)

        out_file = silver_dir / "test-ds" / "canonical_mentions" / "train.jsonl"
        assert out_file.exists()
        with out_file.open() as f:
            rec = json.loads(f.readline())
            # Validate via MentionRecord
            MentionRecord.model_validate(rec)
