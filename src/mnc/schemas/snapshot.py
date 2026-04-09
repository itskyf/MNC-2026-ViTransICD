"""Bronze snapshot record schema for source-faithful dataset ingestion."""

from pydantic import BaseModel, ConfigDict

from mnc.schemas.document import JsonValue


class SnapshotRecord(BaseModel):
    """Source-faithful snapshot of one row from a public dataset.

    Each record represents **one source row**, not necessarily one document.
    """

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    source_split: str
    source_record_id: str
    payload: dict[str, JsonValue]
    source_format: str
    source_path: str
    ingest_version: str
    source_url: str | None = None
    language: str | None = None
    raw_checksum: str | None = None
