"""Document record schema for ingestion and retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from datetime import datetime


class DocumentRecord(BaseModel):
    """Ingestion, normalization, retrieval, and explainability artifact.

    Join key: ``doc_id`` links to all downstream record types.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    raw_text: str
    normalized_text: str
    retrieval_text: str
    sentences: list[str]
    created_at: datetime
