"""Prediction record schema for model inference and evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from datetime import datetime


class PredictionRecord(BaseModel):
    """Model inference and evaluation artifact.

    Join key: ``doc_id`` → DocumentRecord. ``predicted_codes`` reference
    ``code_3char`` values.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    model_name: str
    label_granularity: Literal["code_3char", "chapter"] = "code_3char"
    predicted_codes: list[str]
    scores: dict[str, float]
    created_at: datetime
