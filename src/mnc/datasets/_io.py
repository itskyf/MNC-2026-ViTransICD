"""Shared I/O utilities for bronze layer JSONL and manifest files."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pydantic import BaseModel

    from mnc.schemas.manifest import BronzeManifest


def write_jsonl(records: list[BaseModel], path: Path) -> int:
    """Write Pydantic records to a JSONL file. Returns count written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(rec.model_dump_json() + "\n")
    return len(records)


def write_manifest(manifest: BronzeManifest, path: Path) -> None:
    """Write a bronze manifest to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def now_utc() -> datetime:
    """Return current UTC datetime."""
    return datetime.now(tz=UTC)


def iter_jsonl(path: Path, model: type[BaseModel]) -> Iterator[tuple[int, BaseModel]]:
    """Yield ``(line_number, model_instance)`` from a JSONL file.

    Blank lines are skipped.  Validation errors propagate to the caller.
    """
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            yield line_num, model.model_validate_json(stripped)
