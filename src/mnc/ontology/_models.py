"""Shared data models for the ICD-10 ontology crawler and reporter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

LANG_VI: str = "vi"
LANG_DUAL: str = "dual"


@dataclass(frozen=True)
class ChildRef:
    """Reference to a child node discovered during crawl."""

    model: str
    node_id: str
    is_leaf: bool


@dataclass(frozen=True)
class RawResponseRecord:
    """Single bronze-level record saved to disk per API request."""

    url: str
    endpoint_kind: str
    node_id: str | None
    lang: str
    http_status: int
    request_headers: dict[str, str]
    response_headers: dict[str, str]
    retrieved_at: str
    payload: object


class ManifestEntry(TypedDict):
    """One row in the crawl manifest ledger."""

    url: str
    endpoint_kind: str
    node_id: str | None
    lang: str
    http_status: int
    retrieved_at: str
    saved_path: str
    content_hash: str


class ErrorEntry(TypedDict):
    """Record of a failed request."""

    url: str
    endpoint_kind: str
    node_id: str | None
    lang: str
    error_type: str
    error_message: str
    attempted_at: str
    attempt_number: int


class DiscoverySummary(TypedDict):
    """Schema inference and discovered node kinds from the crawl."""

    node_kinds: dict[str, int]
    endpoint_kinds: dict[str, int]
    total_requests: int
    total_errors: int
    languages: list[str]
    crawled_at_range: tuple[str, str]


@dataclass(frozen=True)
class RequestRef:
    """Identifies a single crawl request (endpoint + lang + node)."""

    endpoint_kind: str
    lang: str
    node_id: str | None


@dataclass
class CrawlState:
    """Mutable state bag carried through the BFS crawl.

    Bundles the manifest, errors, and discovery counters so that
    helper functions don't need 6+ positional arguments.
    """

    manifest: list[ManifestEntry] = field(default_factory=list)
    errors: list[ErrorEntry] = field(default_factory=list)
    node_kinds: dict[str, int] = field(default_factory=dict)
    endpoint_kinds: dict[str, int] = field(default_factory=dict)
    languages: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)


@dataclass
class CoverageStats:
    """Aggregated coverage statistics for the reporter."""

    by_kind_and_lang: dict[tuple[str, str], int] = field(default_factory=dict)
    total_by_kind: dict[str, int] = field(default_factory=dict)
    total_by_lang: dict[str, int] = field(default_factory=dict)
    grand_total: int = 0


@dataclass
class ConflictDetail:
    """Details of a conflicting duplicate across languages."""

    node_id: str
    node_kind: str
    field_name: str
    vi_value: str
    dual_value: str


@dataclass
class DuplicateReport:
    """Result of cross-language duplicate analysis."""

    total_shared_ids: int = 0
    identical_payloads: int = 0
    conflicts: list[ConflictDetail] = field(default_factory=list)


@dataclass
class ReportData:
    """Bundles all inputs for the markdown report generator.

    Avoids passing 6+ positional arguments to ``generate_markdown_report``.
    """

    base_dir: str
    coverage: CoverageStats
    max_depth: int
    duplicates: DuplicateReport
    errors: list[ErrorEntry]
    summary: DiscoverySummary | None
