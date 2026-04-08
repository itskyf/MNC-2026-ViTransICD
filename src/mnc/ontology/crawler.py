"""ICD-10 Bronze Crawler for icd.kcb.vn.

Recursively crawls the KCB ICD-10 API with dynamic endpoint discovery,
saving raw JSON responses for downstream processing.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from src.mnc.ontology._types import (
    ErrorEntry,
    JsonValue,
    ManifestEntry,
    RawEnvelope,
)

logger = logging.getLogger(__name__)

BASE_URL = "https://ccs.whiteneuron.com/api/ICD10"
LANGUAGES = ("vi", "dual")
ENDPOINT_CANDIDATES = ("childs", "data", "tree")

# Field names used to discover node IDs and kinds from JSON responses.
_ID_FIELDS = ("id", "ID", "Id", "code", "ID_mh")
_KIND_FIELDS = ("kind", "type", "level", "loai", "nodeType", "model")
_CHILD_FIELDS = ("children", "childs", "subItems", "items")

# HTTP status code thresholds.
_HTTP_OK = 200
_HTTP_NOT_FOUND = 404
_HTTP_SERVER_ERROR = 500


class CrawlError(Exception):
    """Raised when a request exhausts all retries."""


class ICDCrawler:
    """Async ICD-10 crawler with dynamic endpoint discovery.

    Attributes:
        base_url: API base URL.
        output_dir: Directory for bronze raw output and metadata files.
        concurrency: Maximum number of concurrent HTTP requests.
        limit: Optional cap on total requests. ``None`` means unlimited.
    """

    def __init__(
        self,
        base_url: str = BASE_URL,
        output_dir: Path | str = "data/bronze/kcb_vn_icd10",
        concurrency: int = 5,
        limit: int | None = None,
    ) -> None:
        """Initialise crawler state and concurrency primitives."""
        self.base_url = base_url.rstrip("/")
        self.output_dir = Path(output_dir)
        self.concurrency = concurrency
        self.limit = limit

        self._semaphore = asyncio.Semaphore(concurrency)
        self._request_count = 0
        self._manifest: list[ManifestEntry] = []
        self._errors: list[ErrorEntry] = []
        self._endpoint_map: dict[str, str] = {}
        self._visited: set[str] = set()
        self._queue: asyncio.Queue[tuple[str, str, str]] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def crawl(self) -> None:
        """Run the crawl: load state, seed roots, process, persist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()

        for lang in LANGUAGES:
            key = f"{lang}:root:root"
            if key not in self._visited:
                await self._queue.put((lang, "root", "root"))

        async with httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(30.0, connect=10.0),
            follow_redirects=True,
        ) as client:
            await self._process_queue(client)

        self._save_state()
        logger.info(
            "Crawl finished: %d requests, %d errors",
            self._request_count,
            len(self._errors),
        )

    # ------------------------------------------------------------------
    # State persistence (resumability)
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        manifest_path = self.output_dir / "crawl_manifest.json"
        if manifest_path.exists():
            with manifest_path.open() as f:
                self._manifest = json.load(f)
            for entry in self._manifest:
                key = f"{entry['lang']}:{entry['kind']}:{entry['id']}"
                self._visited.add(key)
                self._request_count += 1
            logger.info("Resumed from manifest: %d entries", len(self._manifest))

        errors_path = self.output_dir / "crawl_errors.json"
        if errors_path.exists():
            with errors_path.open() as f:
                self._errors = json.load(f)

        discovery_path = self.output_dir / "discovery_summary.json"
        if discovery_path.exists():
            with discovery_path.open() as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._endpoint_map = data.get("endpoint_map", {})

    def _save_state(self) -> None:
        with (self.output_dir / "crawl_manifest.json").open("w") as f:
            json.dump(self._manifest, f, indent=2, ensure_ascii=False)

        with (self.output_dir / "crawl_errors.json").open("w") as f:
            json.dump(self._errors, f, indent=2, ensure_ascii=False)

        discovery = {
            "endpoint_map": dict(sorted(self._endpoint_map.items())),
            "discovered_kinds": sorted(self._endpoint_map.keys()),
            "generated_at": datetime.now(UTC).isoformat(),
        }
        with (self.output_dir / "discovery_summary.json").open("w") as f:
            json.dump(discovery, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # Queue processing
    # ------------------------------------------------------------------

    async def _process_queue(self, client: httpx.AsyncClient) -> None:
        """Drain the work queue with bounded concurrency."""
        tasks: set[asyncio.Task[None]] = set()

        while True:
            if self.limit is not None and self._request_count >= self.limit:
                break

            # Reap completed tasks.
            done = {t for t in tasks if t.done()}
            for t in done:
                tasks.discard(t)
                exc = t.exception()
                if exc and not isinstance(exc, (CrawlError, asyncio.CancelledError)):
                    logger.error("Task error: %s", exc)

            # Fetch next work item.
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                if not tasks:
                    break
                _, tasks = await asyncio.wait(
                    tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                continue

            lang, kind, node_id = item

            async def _bounded(
                _lang: str = lang,
                _kind: str = kind,
                _nid: str = node_id,
            ) -> None:
                async with self._semaphore:
                    await self._process_item(client, _lang, _kind, _nid)

            tasks.add(asyncio.create_task(_bounded()))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Item processing
    # ------------------------------------------------------------------

    async def _process_item(
        self,
        client: httpx.AsyncClient,
        lang: str,
        kind: str,
        node_id: str,
    ) -> None:
        dedup_key = f"{lang}:{kind}:{node_id}"
        if dedup_key in self._visited:
            return
        self._visited.add(dedup_key)

        if self.limit is not None and self._request_count >= self.limit:
            return

        # Determine URL pattern.
        if kind == "root":
            url = "/root"
            params: dict[str, str] = {"lang": lang}
            endpoint_type = "root"
        else:
            endpoint_type = await self._probe_endpoint(client, kind, node_id, lang)
            if endpoint_type is None:
                self._errors.append(
                    ErrorEntry(
                        kind=kind,
                        id=node_id,
                        lang=lang,
                        error=f"No working endpoint for kind={kind}",
                        timestamp=datetime.now(UTC).isoformat(),
                    ),
                )
                logger.error("No working endpoint for kind=%s id=%s", kind, node_id)
                return
            url = f"/{endpoint_type}/{kind}"
            params = {"id": node_id, "lang": lang}

        # Fetch.
        try:
            payload, status_code, resp_headers = await self._fetch_with_retry(
                client,
                url,
                params,
            )
            self._request_count += 1
        except CrawlError as exc:
            self._errors.append(
                ErrorEntry(
                    kind=kind,
                    id=node_id,
                    lang=lang,
                    error=str(exc),
                    timestamp=datetime.now(UTC).isoformat(),
                ),
            )
            self._save_state()
            raise

        # Persist raw JSON.
        envelope = RawEnvelope(
            request_url=f"{self.base_url}{url}",
            endpoint_kind=endpoint_type,
            id=node_id,
            lang=lang,
            http_status=status_code,
            headers=resp_headers,
            retrieved_at=datetime.now(UTC).isoformat(),
            data=payload,
        )
        self._save_raw(envelope)

        # Manifest entry.
        self._manifest.append(
            ManifestEntry(
                kind=kind,
                id=node_id,
                lang=lang,
                endpoint=endpoint_type,
                http_status=status_code,
                request_count=self._request_count,
                timestamp=datetime.now(UTC).isoformat(),
            ),
        )

        # Enqueue children.
        for child_kind, child_id in self._extract_children(payload):
            child_key = f"{lang}:{child_kind}:{child_id}"
            if child_key not in self._visited:
                await self._queue.put((lang, child_kind, child_id))

        # Periodic checkpoint every 20 requests.
        if self._request_count % 20 == 0:
            self._save_state()

    # ------------------------------------------------------------------
    # Dynamic endpoint discovery (probing)
    # ------------------------------------------------------------------

    async def _probe_endpoint(
        self,
        client: httpx.AsyncClient,
        kind: str,
        node_id: str,
        lang: str,
    ) -> str | None:
        """Discover which endpoint pattern works for *kind*.

        Probes ``/childs/<kind>`` then ``/data/<kind>`` then ``/tree/<kind>``.
        Only an HTTP 200 is treated as a positive signal.  5xx responses are
        *not* used for control flow — the probe simply moves on to the next
        candidate.
        """
        if kind in self._endpoint_map:
            return self._endpoint_map[kind]

        for candidate in ENDPOINT_CANDIDATES:
            url = f"/{candidate}/{kind}"
            params = {"id": node_id, "lang": lang}
            try:
                resp = await client.get(url, params=params)
                if resp.status_code == _HTTP_OK:
                    self._endpoint_map[kind] = candidate
                    logger.info("Discovered: kind=%s -> /%s", kind, candidate)
                    return candidate
                # Non-200 (including 5xx): skip without assuming wrong endpoint.
            except httpx.RequestError:
                continue

        return None

    # ------------------------------------------------------------------
    # HTTP fetch with retry
    # ------------------------------------------------------------------

    async def _fetch_with_retry(
        self,
        client: httpx.AsyncClient,
        url: str,
        params: dict[str, str],
    ) -> tuple[JsonValue, int, dict[str, str]]:
        """Fetch with exponential backoff on transient errors.

        Returns:
            (parsed_json, http_status, response_headers)
        """
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(httpx.RequestError),
            stop=stop_after_attempt(3),
            wait=wait_random_exponential(multiplier=1, max=30),
            reraise=True,
        ):
            with attempt:
                resp = await client.get(url, params=params)
                if resp.status_code >= _HTTP_SERVER_ERROR:
                    msg = f"Server error {resp.status_code}"
                    raise httpx.RequestError(msg, request=resp.request)
                if resp.status_code == _HTTP_NOT_FOUND:
                    msg = f"404 Not Found: {url} params={params}"
                    raise CrawlError(msg)
                resp.raise_for_status()
                headers = dict(resp.headers)
                return resp.json(), resp.status_code, headers

        msg = "Retry exhausted"
        raise CrawlError(msg)  # pragma: no cover

    # ------------------------------------------------------------------
    # Child extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_nodes_from_list(
        items: list[JsonValue],
    ) -> list[tuple[str, str]]:
        """Extract ``(kind, id)`` pairs from a list of node dicts."""
        nodes: list[tuple[str, str]] = []
        for node in items:
            if not isinstance(node, dict):
                continue
            nid = ICDCrawler._get_id(node)
            nkind = ICDCrawler._get_kind(node)
            if nid and nkind:
                nodes.append((nkind, nid))
        return nodes

    @staticmethod
    def _extract_children(
        data: JsonValue,
    ) -> list[tuple[str, str]]:
        """Return ``[(child_kind, child_id), ...]`` from a response payload.

        Handles the KCB API envelope ``{"status": ..., "data": [...]}`` for
        both root and ``/childs/`` responses, plus ``{"children": [...]}``
        as a fallback pattern.
        """
        # Unwrap API envelope: {"status": "success", "data": <inner>}
        if isinstance(data, dict):
            inner = data.get("data")
            if isinstance(inner, list):
                data = inner

        if isinstance(data, list):
            return ICDCrawler._extract_nodes_from_list(data)

        if not isinstance(data, dict):
            return []

        # Dict with children in known fields (e.g. {"children": [...]}).
        children: list[tuple[str, str]] = []
        for field in _CHILD_FIELDS:
            child_list = data.get(field)
            if isinstance(child_list, list):
                children.extend(
                    ICDCrawler._extract_nodes_from_list(child_list),
                )
        return children

    @staticmethod
    def _get_id(node: dict[str, JsonValue]) -> str | None:
        for field in _ID_FIELDS:
            val = node.get(field)
            if val is not None:
                return str(val)
        return None

    @staticmethod
    def _get_kind(node: dict[str, JsonValue]) -> str | None:
        for field in _KIND_FIELDS:
            val = node.get(field)
            if val is not None:
                return str(val).lower()
        return None

    # ------------------------------------------------------------------
    # Raw persistence
    # ------------------------------------------------------------------

    def _save_raw(self, envelope: RawEnvelope) -> None:
        """Write a raw JSON envelope to the partitioned directory layout."""
        raw_dir = (
            self.output_dir
            / "raw"
            / f"endpoint={envelope['endpoint_kind']}"
            / f"lang={envelope['lang']}"
            / f"id={envelope['id']}"
        )
        raw_dir.mkdir(parents=True, exist_ok=True)

        content = json.dumps(envelope["data"], sort_keys=True, ensure_ascii=False)
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:12]

        raw_file = raw_dir / f"{content_hash}.json"
        with raw_file.open("w") as f:
            json.dump(envelope, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------
# CLI entry point
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the ICD-10 bronze crawler."""
    parser = argparse.ArgumentParser(description="ICD-10 Bronze Crawler")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default="data/bronze/kcb_vn_icd10",
        help="Base output directory (default: data/bronze/kcb_vn_icd10)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after N requests (default: unlimited)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Max concurrent requests (default: 5)",
    )
    parser.add_argument(
        "--base-url",
        default=BASE_URL,
        help=f"API base URL (default: {BASE_URL})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    crawler = ICDCrawler(
        base_url=args.base_url,
        output_dir=args.output_dir,
        concurrency=args.concurrency,
        limit=args.limit,
    )
    asyncio.run(crawler.crawl())


if __name__ == "__main__":
    main()
