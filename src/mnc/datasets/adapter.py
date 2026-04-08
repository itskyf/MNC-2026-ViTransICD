"""Dataset adapter protocol for source-specific ingestion."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime
    from pathlib import Path

    from mnc.schemas.document import DocumentRecord

type RawSample = dict[str, str | int | float | bool | None]


@runtime_checkable
class DatasetAdapter(Protocol):
    """Uniform interface that each dataset adapter must expose.

    The adapter is responsible for source-specific parsing only.
    The core pipeline handles orchestration, validation, logging, and output.
    """

    def dataset_name(self) -> str:
        """Return the canonical dataset identifier."""
        ...

    def discover_splits(self, input_path: Path) -> list[tuple[str, Path]]:
        """Return (split_name, file_path) pairs for each available split."""
        ...

    def iter_raw_samples(self, split_path: Path) -> Iterable[RawSample]:
        """Yield raw row dicts from a single split file."""
        ...

    def to_de1_record(
        self,
        raw_sample: RawSample,
        *,
        split: str,
        created_at: datetime,
    ) -> DocumentRecord:
        """Convert a raw row into a DE-1 DocumentRecord."""
        ...

    def validate_raw_sample(self, raw_sample: RawSample) -> bool:
        """Return True if the raw row is structurally valid."""
        ...
