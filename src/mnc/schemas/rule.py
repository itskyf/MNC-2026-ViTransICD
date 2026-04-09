"""Coding rule schema for ICD-10."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RuleRecord(BaseModel):
    """A deterministic rule record extracted from ICD-10 PDF."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    scope: Literal["global", "code"]
    code_3char: str | None = None
    topic: Literal[
        "principal_diagnosis",
        "symptom_fallback",
        "mortality_coding",
        "official_3char_policy",
        "include_note",
        "exclude_note",
        "use_additional_code",
        "code_first",
        "general_note",
    ]
    action: Literal[
        "prefer",
        "fallback",
        "restrict",
        "allow",
        "require_additional_code",
        "code_first",
        "note",
    ]
    priority: int
    source_page_start: int | None = None
    source_page_end: int | None = None
    evidence_text: str
    normalized_text: str
    created_at: datetime
