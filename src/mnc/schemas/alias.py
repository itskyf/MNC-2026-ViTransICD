"""Alias dictionary schema for ICD-10."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class AliasRecord(BaseModel):
    """A deterministic alias record for an ICD-10 code."""

    model_config = ConfigDict(extra="forbid")

    alias_id: str
    code_3char: str
    alias: str
    alias_norm: str
    alias_type: Literal[
        "title_vi",
        "title_en",
        "parenthetical",
        "inclusion",
        "nos_form",
        "bilingual_variant",
    ]
    language: Literal["vi", "en", "mixed", "unknown"]
    match_level: Literal["exact", "normalized", "fuzzy_seed"] = "exact"
    source_page: int | None = None
    source_line: str | None = None
    created_at: datetime
