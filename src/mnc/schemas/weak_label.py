"""Weak label schema for DC-4 silver-stage aggregation."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class WeakEvidenceSpan(BaseModel):
    """Evidence span from a mention-backed candidate link."""

    model_config = ConfigDict(extra="forbid")

    mention_id: str
    char_start: int
    char_end: int
    text: str
    methods: list[Literal["exact", "normalized", "fuzzy"]]
    score: float


class WeakLabelRecord(BaseModel):
    """Positive weak label for 3-character ICD code prediction."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    code_3char: str
    label_type: Literal["positive"]
    confidence: float
    rank: int
    support_methods: list[Literal["exact", "normalized", "fuzzy", "tfidf", "bm25"]]
    support_rule_ids: list[str]
    evidence_spans: list[WeakEvidenceSpan]
    created_at: datetime
