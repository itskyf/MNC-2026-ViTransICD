"""Document record schema for ingestion and retrieval."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, model_validator

type JsonValue = str | int | float | bool | None


class DocumentRecord(BaseModel):
    """Ingestion, normalization, retrieval, and explainability artifact.

    Join key: ``doc_id`` links to all downstream record types.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    source: str
    language: str
    raw_text: str
    source_record_id: str | None = None
    split: str | None = None
    payload: dict[str, JsonValue] | None = None
    normalized_text: str = ""
    retrieval_text: str = ""
    sentences: list[str] | None = None
    created_at: datetime

    @model_validator(mode="after")
    def _coalesce_defaults(self) -> "DocumentRecord":
        """Replace None defaults for mutable-typed fields."""
        if self.payload is None:
            self.payload = {}
        if self.sentences is None:
            self.sentences = []
        return self
