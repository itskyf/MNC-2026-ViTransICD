"""Bronze manifest schema for dataset artifacts."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BronzeManifest(BaseModel):
    """Dataset-level manifest for bronze layer snapshots and documents."""

    model_config = ConfigDict(extra="forbid")

    dataset: str
    input_splits: list[str]
    output_splits: list[str]
    record_count_by_split: dict[str, int]
    failed_count_by_split: dict[str, int]
    created_at: datetime
