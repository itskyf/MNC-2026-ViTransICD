"""Mention record schema for extraction and evidence tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from datetime import datetime


class MentionRecord(BaseModel):
    """Mention extraction and evidence tracking artifact.

    Join keys: ``doc_id`` → DocumentRecord, ``mention_id`` → CandidateLink.
    """

    model_config = ConfigDict(extra="forbid")

    mention_id: str
    doc_id: str
    text: str
    normalized_text: str
    mention_type: Literal[
        "disease",
        "symptom",
        "diagnosis",
        "procedure",
        "abbreviation",
        "other",
    ]
    char_start: int
    char_end: int
    confidence: float | None = None
    created_at: datetime
