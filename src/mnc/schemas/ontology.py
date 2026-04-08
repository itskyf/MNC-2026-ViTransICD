"""Ontology code schema for ICD-10 label lookup and candidate generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from datetime import datetime


class OntologyCode(BaseModel):
    """Canonical ICD-10 code entry for candidate generation and label lookup.

    Join key: ``code_3char`` links to candidates, silver labels, predictions,
    and explanations.
    """

    model_config = ConfigDict(extra="forbid")

    code_3char: str
    chapter_id: str | None = None
    title_vi: str
    title_en: str | None = None
    aliases: list[str] = []
    search_text: str
    created_at: datetime
