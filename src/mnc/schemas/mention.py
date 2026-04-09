"""Mention record schema for extraction and evidence tracking."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

MentionType = Literal[
    "disease",
    "symptom",
    "diagnosis",
    "procedure",
    "abbreviation",
    "other",
]


class MentionRecord(BaseModel):
    """Mention extraction and evidence tracking artifact.

    Join keys: ``doc_id`` → DocumentRecord, ``mention_id`` → CandidateLink.
    """

    model_config = ConfigDict(extra="forbid")

    mention_id: str
    doc_id: str
    text: str
    normalized_text: str
    mention_type: MentionType
    char_start: int
    char_end: int
    confidence: float | None = None
    created_at: datetime
