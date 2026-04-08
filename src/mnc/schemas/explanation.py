"""Explanation and evidence span schemas for explainability output."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from datetime import datetime


class EvidenceSpan(BaseModel):
    """A single evidence span within an explanation."""

    model_config = ConfigDict(extra="forbid")

    char_start: int
    char_end: int
    text: str
    score: float | None = None


class ExplanationRecord(BaseModel):
    """Explainability and demo output artifact.

    Join keys: ``doc_id`` → DocumentRecord, ``code_3char`` → OntologyCode.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    code_3char: str
    spans: list[EvidenceSpan]
    matched_label_text: str | None = None
    summary: str | None = None
    created_at: datetime
