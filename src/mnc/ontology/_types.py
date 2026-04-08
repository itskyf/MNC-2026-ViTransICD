"""Shared type definitions for the ICD-10 ontology pipeline."""

from __future__ import annotations

from typing import TypedDict

type JsonValue = (
    str | int | float | bool | None | dict[str, JsonValue] | list[JsonValue]
)


class ManifestEntry(TypedDict):
    """One row in ``crawl_manifest.json``."""

    kind: str
    id: str
    lang: str
    endpoint: str
    http_status: int
    request_count: int
    timestamp: str


class ErrorEntry(TypedDict):
    """One failed-request record in ``crawl_errors.json``."""

    kind: str
    id: str
    lang: str
    error: str
    timestamp: str


class RawEnvelope(TypedDict):
    """Bronze raw JSON envelope written to disk by the crawler."""

    request_url: str
    endpoint_kind: str
    id: str
    lang: str
    http_status: int
    headers: dict[str, str]
    retrieved_at: str
    data: JsonValue


class DiscoverySummary(TypedDict):
    """Contents of ``discovery_summary.json``."""

    endpoint_map: dict[str, str]
    discovered_kinds: list[str]
    generated_at: str
