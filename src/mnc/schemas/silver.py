"""Silver record schema for weak supervision and training targets."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from datetime import datetime


class SilverRecord(BaseModel):
    """Weak supervision aggregation and model training target.

    Join key: ``doc_id`` → DocumentRecord. ``silver_labels`` and
    ``candidate_codes`` reference ``code_3char`` values.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    label_granularity: Literal["code_3char", "chapter"] = "code_3char"
    silver_labels: list[str]
    candidate_codes: list[str]
    confidence: float | None = None
    split: Literal["train", "val", "test"]
    created_at: datetime
