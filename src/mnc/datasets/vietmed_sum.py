"""VietMed-Sum adapter: ingest Vietnamese whole-dialogue splits."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import polars as pl

from mnc.schemas.document import DocumentRecord

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime
    from pathlib import Path

    from mnc.datasets.adapter import RawSample

_SPLIT_FILES: dict[str, str] = {
    "train_whole": "train_whole.parquet",
    "dev_whole": "dev_whole.parquet",
    "test_whole": "test_whole.parquet",
}


def _content_hash(transcript: str, summary: str) -> str:
    """Deterministic source_record_id derived from row content."""
    digest = hashlib.sha256(f"{transcript}|{summary}".encode()).hexdigest()
    return digest[:16]


def _coerce_row(row: dict[str, object]) -> RawSample:
    """Keep only JSON-compatible scalar values from a polars row."""
    return {
        k: v
        for k, v in row.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }


class VietMedSumAdapter:
    """Adapter for leduckhai/VietMed-Sum (whole-dialogue Vietnamese splits)."""

    def dataset_name(self) -> str:
        """Return the canonical dataset identifier."""
        return "vietmed-sum"

    def discover_splits(self, input_path: Path) -> list[tuple[str, Path]]:
        """Return (split_name, file_path) for each available whole-dialogue split."""
        found: list[tuple[str, Path]] = []
        for split_name, filename in _SPLIT_FILES.items():
            p = input_path / filename
            if p.exists():
                found.append((split_name, p))
        return found

    def iter_raw_samples(self, split_path: Path) -> Iterable[RawSample]:
        """Yield rows from a Parquet split as dicts."""
        df = pl.read_parquet(split_path)
        for row in df.iter_rows(named=True):
            yield _coerce_row(row)

    def validate_raw_sample(self, raw_sample: RawSample) -> bool:
        """A valid row must have non-empty transcript and summary strings."""
        transcript = raw_sample.get("transcript")
        summary = raw_sample.get("summary")
        return (
            isinstance(transcript, str)
            and bool(transcript.strip())
            and isinstance(summary, str)
        )

    def to_de1_record(
        self,
        raw_sample: RawSample,
        *,
        split: str,
        created_at: datetime,
    ) -> DocumentRecord:
        """Map a raw VietMed-Sum row to a DocumentRecord."""
        transcript = str(raw_sample["transcript"])
        summary = str(raw_sample["summary"])
        source_record_id = _content_hash(transcript, summary)
        return DocumentRecord(
            doc_id=f"vietmed-sum:{source_record_id}",
            source=self.dataset_name(),
            language="vi",
            raw_text=transcript,
            source_record_id=source_record_id,
            split=split,
            payload={"transcript": transcript, "summary": summary},
            created_at=created_at,
        )
