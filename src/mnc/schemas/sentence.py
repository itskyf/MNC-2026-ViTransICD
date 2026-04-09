"""Sentence span record schema for DC-1 sentence segmentation."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SentenceSpanRecord(BaseModel):
    """Sentence segmentation artifact produced by DC-1.

    Offsets reference ``DocumentRecord.normalized_text``.
    ``char_end`` is exclusive (Python-slice compatible).
    """

    model_config = ConfigDict(extra="forbid")

    sentence_id: str
    doc_id: str
    sentence_index: int
    text: str
    char_start: int
    char_end: int
    created_at: datetime
