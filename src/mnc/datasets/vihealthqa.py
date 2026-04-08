"""ViHealthQA adapter: ingest train/val/test CSV splits."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from mnc.schemas.document import DocumentRecord

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime
    from pathlib import Path

    from mnc.datasets.adapter import RawSample

_SPLIT_MAP: dict[str, str] = {
    "train": "train.csv",
    "validation": "val.csv",
    "test": "test.csv",
}


class ViHealthQAAdapter:
    """Adapter for tarudesu/ViHealthQA (Vietnamese health Q&A)."""

    def dataset_name(self) -> str:
        """Return the canonical dataset identifier."""
        return "vihealthqa"

    def discover_splits(self, input_path: Path) -> list[tuple[str, Path]]:
        """Return (split_name, file_path) for each available CSV split."""
        found: list[tuple[str, Path]] = []
        for split_name, filename in _SPLIT_MAP.items():
            p = input_path / filename
            if p.exists():
                found.append((split_name, p))
        return found

    def iter_raw_samples(self, split_path: Path) -> Iterable[RawSample]:
        """Yield rows from a CSV split as dicts."""
        df = pl.read_csv(split_path)
        for row in df.iter_rows(named=True):
            yield _coerce_row(row)

    def validate_raw_sample(self, raw_sample: RawSample) -> bool:
        """A valid row must have non-empty question and answer strings."""
        question = raw_sample.get("question")
        answer = raw_sample.get("answer")
        return (
            isinstance(question, str)
            and bool(question.strip())
            and isinstance(answer, str)
            and bool(answer.strip())
        )

    def to_de1_record(
        self,
        raw_sample: RawSample,
        *,
        split: str,
        created_at: datetime,
    ) -> DocumentRecord:
        """Map a raw ViHealthQA row to a DocumentRecord."""
        question = str(raw_sample["question"])
        answer = str(raw_sample["answer"])
        link = raw_sample.get("link")
        source_id = str(raw_sample.get("id", ""))
        raw_text = f"{question}\n{answer}"
        return DocumentRecord(
            doc_id=f"vihealthqa:{split}:{source_id}",
            source=self.dataset_name(),
            language="vi",
            raw_text=raw_text,
            source_record_id=source_id,
            split=split,
            payload={
                "question": question,
                "answer": answer,
                "link": str(link) if isinstance(link, str) else "",
            },
            created_at=created_at,
        )


def _coerce_row(row: dict[str, object]) -> RawSample:
    """Keep only JSON-compatible scalar values from a polars row."""
    return {
        k: v
        for k, v in row.items()
        if isinstance(v, (str, int, float, bool)) or v is None
    }
