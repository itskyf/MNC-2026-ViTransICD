"""Bronze crawler for the KCB/MoH ICD-10 Vietnamese ontology.

Recursively fetches all nodes from the public API using a BFS queue,
saving raw JSON responses with full metadata for downstream processing.

Usage::

    uv run -m mnc.ontology.crawler [output_dir]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
from collections import deque
from datetime import UTC, datetime
from pathlib import Path

import httpx
import tenacity

from mnc.ontology._models import (
    LANG_DUAL,
    LANG_VI,
    ChildRef,
    CrawlState,
    DiscoverySummary,
    ErrorEntry,
    ManifestEntry,
    RawResponseRecord,
    RequestRef,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://ccs.whiteneuron.com/api/ICD10"
_ROOT_SENTINEL = "__root__"
_SERVER_ERROR_THRESHOLD = 500
_REFERER = "https://icd.whiteneuron.com/"

# Regex to extract section range IDs (e.g. "A00-A09") from chapter HTML.
_SECTION_RE = re.compile(r"\b([A-Z]\d{2}-[A-Z]\d{2})\b")

# Maps a child node's model field to the API path pattern for fetching it.
# "data" → /api/ICD10/data/{model}?id=...&lang=...
# "tree" → /api/ICD10/tree/{model}?id=...&lang=...
_CHILD_FETCH_STRATEGY: dict[str, str] = {
    "chapter": "data",
    "section": "tree",
    "type": "tree",
    "disease": "data",
}

# How often (in successful requests) to flush the manifest to disk.
_FLUSH_INTERVAL = 10

# Polite delay between requests (seconds).
_REQUEST_DELAY = 0.5


def _now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(tz=UTC).isoformat()


def build_url(endpoint_kind: str, node_id: str | None, lang: str) -> str:
    """Construct the full API URL from endpoint kind, node ID, and language.

    Args:
        endpoint_kind: One of "root", "data_{model}", or "tree_{model}".
        node_id: The node identifier, or None for root requests.
        lang: Language code ("vi" or "dual").

    Returns:
        Fully-qualified URL string.
    """
    if endpoint_kind == "root":
        return f"{_BASE_URL}/root?lang={lang}"

    # endpoint_kind is "data_chapter", "tree_section", etc.
    parts = endpoint_kind.split("_", maxsplit=1)
    pattern = parts[0]  # "data" or "tree"
    model = parts[1]  # "chapter", "section", "type", "disease"
    return f"{_BASE_URL}/{pattern}/{model}?id={node_id}&lang={lang}"


def determine_endpoint_kind(child_model: str) -> str:
    """Map a child node's model name to the endpoint kind for fetching it.

    Args:
        child_model: The ``model`` field from an API node object.

    Returns:
        Endpoint kind string like ``"data_chapter"`` or ``"tree_section"``.
    """
    strategy = _CHILD_FETCH_STRATEGY.get(child_model, "data")
    return f"{strategy}_{child_model}"


def _content_hash(payload: object) -> str:
    """Return the SHA-256 hex digest of a JSON-serialised payload."""
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _file_stem(content_hash: str) -> str:
    """Return the first 12 hex characters for use as a filename."""
    return content_hash[:12]


def _dedup_key(endpoint_kind: str, lang: str, node_id: str | None) -> str:
    """Build a unique dedup/checkout key for a request."""
    return f"{endpoint_kind}:{lang}:{node_id or _ROOT_SENTINEL}"


def save_raw_response(base_dir: Path, record: RawResponseRecord) -> Path:
    """Save a raw response record to the partitioned directory layout.

    Directory structure::

        raw/endpoint=<kind>/lang=<lang>/id=<node_id>/<hash_prefix>.json

    Args:
        base_dir: Base output directory.
        record: The response record to persist.

    Returns:
        Path to the saved file.
    """
    chash = _content_hash(record.payload)
    dir_name = (
        f"endpoint={record.endpoint_kind}"
        f"/lang={record.lang}"
        f"/id={record.node_id or _ROOT_SENTINEL}"
    )
    out_dir = base_dir / "raw" / dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{_file_stem(chash)}.json"

    data = {
        "url": record.url,
        "endpoint_kind": record.endpoint_kind,
        "node_id": record.node_id,
        "lang": record.lang,
        "http_status": record.http_status,
        "request_headers": record.request_headers,
        "response_headers": dict(record.response_headers),
        "retrieved_at": record.retrieved_at,
        "payload": record.payload,
    }
    serialized = json.dumps(data, ensure_ascii=False, indent=2)
    out_path.write_text(serialized, encoding="utf-8")
    return out_path


def load_checkpoint(manifest_path: Path) -> set[str]:
    """Load an existing manifest to build a checkpoint set for dedup.

    Args:
        manifest_path: Path to ``crawl_manifest.json``.

    Returns:
        Set of dedup keys for all previously completed requests.
    """
    if not manifest_path.exists():
        return set()

    entries: list[ManifestEntry] = json.loads(
        manifest_path.read_text(encoding="utf-8"),
    )
    return {_dedup_key(e["endpoint_kind"], e["lang"], e["node_id"]) for e in entries}


def _refs_from_node_list(nodes: list[object]) -> list[ChildRef]:
    """Convert a list of API node objects to ChildRef instances."""
    refs: list[ChildRef] = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        model = node.get("model")
        node_id = node.get("id")
        is_leaf = node.get("is_leaf", False)
        if isinstance(model, str) and isinstance(node_id, str):
            refs.append(
                ChildRef(model=model, node_id=node_id, is_leaf=bool(is_leaf)),
            )
    return refs


def _find_deepest_expanded_children(
    tree_nodes: list[object],
) -> list[object] | None:
    """Walk the tree following ``expanded: true`` flags to the deepest level.

    Returns the children of the deepest expanded node, or None if no node
    is expanded.
    """
    for node in tree_nodes:
        if not isinstance(node, dict):
            continue
        if node.get("expanded") is True:
            nested = node.get("children")
            if isinstance(nested, list):
                # Recurse deeper to find an even deeper expanded node.
                deeper = _find_deepest_expanded_children(nested)
                if deeper is not None:
                    return deeper
                # This is the deepest — return its children.
                return nested
    return None


def _find_expanded_children(
    tree_nodes: list[object],
    target_id: str,
) -> list[object] | None:
    """Legacy: find children of the expanded target in a tree response.

    .. deprecated:: Use ``_find_deepest_expanded_children`` instead — the API
       expands the *path* to the target, not the target itself.
    """
    for node in tree_nodes:
        if not isinstance(node, dict):
            continue
        if node.get("id") == target_id and node.get("expanded") is True:
            return node.get("children", [])
        nested = node.get("children")
        if isinstance(nested, list):
            result = _find_expanded_children(nested, target_id)
            if result is not None:
                return result
    return None


def _extract_root_children(payload: dict[str, object]) -> list[ChildRef]:
    """Extract children from a root endpoint response."""
    nodes = payload.get("data")
    if isinstance(nodes, list):
        return _refs_from_node_list(nodes)
    return []


def _extract_data_children(
    payload: dict[str, object],
    endpoint_kind: str,
    node_id: str | None,
) -> list[ChildRef]:
    """Extract children from a data endpoint response.

    For chapters, the API doesn't return a ``children`` list — sections are
    embedded in the HTML.  This function parses the HTML to extract section IDs
    and returns them as ``ChildRef`` objects with ``model="section"``.
    """
    node = payload.get("data")
    if not isinstance(node, dict):
        return []

    # Try structured children first.
    child_list = node.get("children")
    if isinstance(child_list, list):
        return _refs_from_node_list(child_list)

    # Fallback: parse HTML for chapter → section discovery.
    if endpoint_kind == "data_chapter" and node_id is not None:
        inner = node.get("data")
        if isinstance(inner, dict):
            html = inner.get("html")
            if isinstance(html, str):
                return _extract_sections_from_html(html, node_id)

    return []


def _extract_sections_from_html(
    html: str,
    chapter_id: str,
) -> list[ChildRef]:
    """Parse chapter HTML to discover section range IDs.

    Section IDs look like ``A00-A09`` (two letter-digit pairs separated by a
    dash).  We exclude the chapter's own ID and any IDs outside its range.
    """
    candidates = set(_SECTION_RE.findall(html))
    # Remove the chapter ID itself (e.g. "A00-B99").
    candidates.discard(chapter_id)
    return [
        ChildRef(model="section", node_id=sid, is_leaf=False)
        for sid in sorted(candidates)
    ]


def _extract_tree_children(
    payload: dict[str, object],
) -> list[ChildRef]:
    """Extract children from a tree endpoint response.

    Tree endpoints return the full chapter tree with the path *to* the target
    expanded.  The children we want are those of the **deepest expanded** node,
    not the target itself (which is not marked as expanded).
    """
    tree_nodes = payload.get("children")
    if not isinstance(tree_nodes, list):
        return []
    deepest = _find_deepest_expanded_children(tree_nodes)
    if deepest is not None:
        return _refs_from_node_list(deepest)
    return _refs_from_node_list(tree_nodes)


def extract_children(
    payload: object,
    endpoint_kind: str,
    target_id: str | None,
) -> list[ChildRef]:
    """Parse an API response and extract child node references.

    Dispatches to the appropriate extractor based on endpoint kind.

    Args:
        payload: The parsed JSON response body.
        endpoint_kind: The endpoint kind that produced this response.
        target_id: The node ID that was requested (None for root).

    Returns:
        List of child node references to enqueue.
    """
    if not isinstance(payload, dict):
        return []

    if endpoint_kind == "root":
        return _extract_root_children(payload)
    if endpoint_kind.startswith("data_"):
        return _extract_data_children(payload, endpoint_kind, target_id)
    if endpoint_kind.startswith("tree_"):
        return _extract_tree_children(payload)
    return []


def _flush_manifests(
    manifest_path: Path,
    errors_path: Path,
    summary_path: Path,
    state: CrawlState,
) -> None:
    """Write all manifest files to disk."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(state.manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    errors_path.write_text(
        json.dumps(state.errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if state.manifest:
        timestamps = [e["retrieved_at"] for e in state.manifest]
        time_range: tuple[str, str] = (min(timestamps), max(timestamps))
    else:
        time_range = (_now_iso(), _now_iso())

    summary: DiscoverySummary = {
        "node_kinds": dict(sorted(state.node_kinds.items())),
        "endpoint_kinds": dict(sorted(state.endpoint_kinds.items())),
        "total_requests": len(state.manifest),
        "total_errors": len(state.errors),
        "languages": sorted(state.languages),
        "crawled_at_range": time_range,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def fetch_with_retry(
    client: httpx.AsyncClient,
    url: str,
) -> httpx.Response:
    """Fetch a single URL with tenacity retry on transient errors.

    Args:
        client: The httpx async client.
        url: The URL to fetch.

    Returns:
        The HTTP response.

    Raises:
        httpx.HTTPStatusError: If a non-retryable HTTP error occurs.
    """
    retryer = tenacity.AsyncRetrying(
        stop=tenacity.stop_after_attempt(5),
        wait=tenacity.wait_exponential(multiplier=1, min=2, max=60),
        retry=tenacity.retry_if_exception_type(
            (
                httpx.TimeoutException,
                httpx.ConnectError,
            ),
        ),
        reraise=True,
    )
    response = await retryer(client.get, url)

    # Raise for 5xx (transient), but NOT for 404 (permanent).
    if response.status_code >= _SERVER_ERROR_THRESHOLD:
        response.raise_for_status()

    return response


def _record_error(
    state: CrawlState,
    ref: RequestRef,
    url: str,
    exc: httpx.TimeoutException | httpx.ConnectError | httpx.HTTPStatusError,
) -> None:
    """Append an error entry and log it."""
    entry = ErrorEntry(
        url=url,
        endpoint_kind=ref.endpoint_kind,
        node_id=ref.node_id,
        lang=ref.lang,
        error_type=type(exc).__name__,
        error_message=str(exc),
        attempted_at=_now_iso(),
        attempt_number=1,
    )
    state.errors.append(entry)
    logger.warning("Error fetching %s: %s", url, exc)


def _update_node_kinds(
    node_kinds: dict[str, int],
    payload: object,
    endpoint_kind: str,
) -> None:
    """Track unique node model values found in API responses."""
    if not isinstance(payload, dict):
        return

    nodes: list[object] = []
    if endpoint_kind == "root":
        data = payload.get("data")
        if isinstance(data, list):
            nodes = data
    elif endpoint_kind.startswith("tree_"):
        nodes = payload.get("children", [])
    elif endpoint_kind.startswith("data_"):
        data = payload.get("data")
        if isinstance(data, dict):
            nodes = [data]

    for node in nodes:
        if isinstance(node, dict):
            model = node.get("model")
            if isinstance(model, str):
                node_kinds[model] = node_kinds.get(model, 0) + 1


def _save_and_record(
    base_dir: Path,
    state: CrawlState,
    ref: RequestRef,
    response: httpx.Response,
    retrieved_at: str,
) -> None:
    """Save a raw response and append to the manifest."""
    url = build_url(ref.endpoint_kind, ref.node_id, ref.lang)
    payload = response.json()
    record = RawResponseRecord(
        url=url,
        endpoint_kind=ref.endpoint_kind,
        node_id=ref.node_id,
        lang=ref.lang,
        http_status=response.status_code,
        request_headers=dict(response.request.headers),
        response_headers=dict(response.headers),
        retrieved_at=retrieved_at,
        payload=payload,
    )
    saved_path = save_raw_response(base_dir, record)

    _update_node_kinds(state.node_kinds, payload, ref.endpoint_kind)
    state.endpoint_kinds[ref.endpoint_kind] = (
        state.endpoint_kinds.get(ref.endpoint_kind, 0) + 1
    )

    chash = _content_hash(payload)
    rel_path = saved_path.relative_to(base_dir)
    state.manifest.append(
        ManifestEntry(
            url=url,
            endpoint_kind=ref.endpoint_kind,
            node_id=ref.node_id,
            lang=ref.lang,
            http_status=response.status_code,
            retrieved_at=retrieved_at,
            saved_path=str(rel_path),
            content_hash=chash,
        ),
    )


def _enqueue_children(
    state: CrawlState,
    queue: deque[RequestRef],
    payload: object,
    ref: RequestRef,
) -> None:
    """Discover children from a response and enqueue unvisited ones."""
    children = extract_children(payload, ref.endpoint_kind, ref.node_id)
    for child in children:
        child_endpoint = determine_endpoint_kind(child.model)
        child_key = _dedup_key(child_endpoint, ref.lang, child.node_id)
        if child_key not in state.completed:
            queue.append(RequestRef(child_endpoint, ref.lang, child.node_id))


async def _process_queue_item(
    base_dir: Path,
    state: CrawlState,
    queue: deque[RequestRef],
    client: httpx.AsyncClient,
    ref: RequestRef,
) -> int:
    """Fetch, save, and enqueue children for a single queue item.

    Returns 1 if a new request was made, 0 if skipped (checkpoint).
    """
    key = _dedup_key(ref.endpoint_kind, ref.lang, ref.node_id)
    if key in state.completed:
        return 0

    url = build_url(ref.endpoint_kind, ref.node_id, ref.lang)
    logger.info("Fetching %s", url)

    try:
        response = await fetch_with_retry(client, url)
    except (
        httpx.TimeoutException,
        httpx.ConnectError,
        httpx.HTTPStatusError,
    ) as exc:
        _record_error(state, ref, url, exc)
        return 0

    # Rate-limit politely.
    await asyncio.sleep(_REQUEST_DELAY)

    retrieved_at = _now_iso()
    _save_and_record(base_dir, state, ref, response, retrieved_at)
    state.completed.add(key)

    payload = response.json()
    _enqueue_children(state, queue, payload, ref)
    return 1


async def crawl(base_output_dir: Path) -> None:
    """Run the full BFS crawl from both roots.

    Args:
        base_output_dir: Root directory for all crawl output.
    """
    manifests_dir = base_output_dir / "manifests"
    manifest_path = manifests_dir / "crawl_manifest.json"
    errors_path = manifests_dir / "crawl_errors.json"
    summary_path = manifests_dir / "discovery_summary.json"

    # Load checkpoint for resume support.
    completed = load_checkpoint(manifest_path)
    logger.info("Checkpoint loaded: %d completed requests", len(completed))

    state = CrawlState(completed=completed)

    # Restore previous run's data if present.
    if manifest_path.exists():
        state.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if errors_path.exists():
        state.errors = json.loads(errors_path.read_text(encoding="utf-8"))
    if summary_path.exists():
        prev_summary: DiscoverySummary = json.loads(
            summary_path.read_text(encoding="utf-8"),
        )
        state.node_kinds.update(prev_summary.get("node_kinds", {}))

    # Seed BFS queue with both language roots.
    queue: deque[RequestRef] = deque()
    for lang in (LANG_VI, LANG_DUAL):
        queue.append(RequestRef("root", lang, None))
        state.languages.add(lang)

    flush_counter = 0

    headers = {"Referer": _REFERER}
    async with httpx.AsyncClient(timeout=30.0, headers=headers) as client:
        while queue:
            ref = queue.popleft()
            made_request = await _process_queue_item(
                base_output_dir,
                state,
                queue,
                client,
                ref,
            )
            flush_counter += made_request

            # Periodic flush.
            if flush_counter >= _FLUSH_INTERVAL:
                _flush_manifests(manifest_path, errors_path, summary_path, state)
                logger.info(
                    "Flushed manifest: %d requests, %d remaining",
                    len(state.manifest),
                    len(queue),
                )
                flush_counter = 0

    # Final flush.
    _flush_manifests(manifest_path, errors_path, summary_path, state)
    logger.info(
        "Crawl complete. %d requests, %d errors.",
        len(state.manifest),
        len(state.errors),
    )


def main(output_dir: str) -> None:
    """CLI entry point: parse args and run the async crawl.

    Args:
        output_dir: Base directory for crawl output.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    base = Path(output_dir).resolve()
    logger.info("Starting crawl. Output directory: %s", base)
    asyncio.run(crawl(base))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crawl ICD-10 ontology from the KCB API.",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="data/bronze/kcb_vn_icd10/",
        help="Base directory for crawl output.",
    )
    args = parser.parse_args()
    main(args.output_dir)
