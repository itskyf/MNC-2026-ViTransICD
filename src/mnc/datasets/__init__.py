"""Public dataset ingestion pipeline for ViTransICD."""

from __future__ import annotations

from mnc.datasets.adapter import DatasetAdapter, RawSample
from mnc.datasets.pipeline import Manifest, ingest
from mnc.datasets.vietmed_sum import VietMedSumAdapter
from mnc.datasets.vihealthqa import ViHealthQAAdapter

__all__ = [
    "DatasetAdapter",
    "Manifest",
    "RawSample",
    "ViHealthQAAdapter",
    "VietMedSumAdapter",
    "ingest",
]

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "vietmed-sum": VietMedSumAdapter,
    "vihealthqa": ViHealthQAAdapter,
}
