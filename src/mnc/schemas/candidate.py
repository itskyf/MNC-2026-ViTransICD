"""Candidate link schema for weak supervision and match auditing."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class CandidateLink(BaseModel):
    """Candidate generation and weak supervision artifact.

    Links a mention (or document) to an ontology code with a scored match method.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    mention_id: str | None = None
    code_3char: str
    method: Literal["exact", "normalized", "fuzzy", "tfidf", "bm25", "dense"]
    score: float
    char_start: int | None = None
    char_end: int | None = None
    created_at: datetime
