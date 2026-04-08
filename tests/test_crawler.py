"""Unit tests for src/mnc/ontology/crawler.py.

All HTTP traffic is mocked via *respx* so tests run fully offline.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import httpx
import pytest
import pytest_asyncio
import respx

from mnc.ontology.crawler import BASE_URL, ICDCrawler, main

from .conftest import (
    SAMPLE_CHAPTER,
    SAMPLE_DISEASE,
    SAMPLE_ROOT_DUAL,
    SAMPLE_ROOT_VI,
    SAMPLE_SECTION,
    SAMPLE_TYPE,
    write_discovery,
    write_manifest,
)

if TYPE_CHECKING:
    from pathlib import Path


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest_asyncio.fixture()
async def crawler(tmp_path: Path) -> ICDCrawler:
    """Return a fresh ICDCrawler pointed at a temp directory."""
    return ICDCrawler(
        base_url=BASE_URL,
        output_dir=str(tmp_path / "data"),
        concurrency=3,
        limit=None,
    )


# -----------------------------------------------------------------------
# Endpoint probing
# -----------------------------------------------------------------------


class TestProbeEndpoint:
    """Test the dynamic endpoint discovery mechanism."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_probe_discovers_childs(self, crawler: ICDCrawler) -> None:
        """/childs/<kind> returns 200 -> cached as 'childs'."""
        respx.get(f"{BASE_URL}/childs/chapter?id=1&lang=vi").respond(
            200,
            json=SAMPLE_CHAPTER,
        )

        result = await crawler._probe_endpoint(
            httpx.AsyncClient(base_url=BASE_URL),
            "chapter",
            "1",
            "vi",
        )
        assert result == "childs"
        assert crawler._endpoint_map["chapter"] == "childs"

    @pytest.mark.asyncio
    @respx.mock
    async def test_probe_falls_back_to_data(self, crawler: ICDCrawler) -> None:
        """/childs/<kind> returns 404, /data/<kind> returns 200 -> cached as 'data'."""
        respx.get(f"{BASE_URL}/childs/disease?id=A00.0&lang=vi").respond(404)
        respx.get(f"{BASE_URL}/data/disease?id=A00.0&lang=vi").respond(
            200,
            json=SAMPLE_DISEASE,
        )

        result = await crawler._probe_endpoint(
            httpx.AsyncClient(base_url=BASE_URL),
            "disease",
            "A00.0",
            "vi",
        )
        assert result == "data"
        assert crawler._endpoint_map["disease"] == "data"

    @pytest.mark.asyncio
    @respx.mock
    async def test_probe_falls_back_to_tree(self, crawler: ICDCrawler) -> None:
        """/childs and /data return 404, /tree returns 200 -> cached as 'tree'."""
        respx.get(f"{BASE_URL}/childs/unknown?id=X&lang=vi").respond(404)
        respx.get(f"{BASE_URL}/data/unknown?id=X&lang=vi").respond(404)
        respx.get(f"{BASE_URL}/tree/unknown?id=X&lang=vi").respond(
            200,
            json={"id": "X"},
        )

        result = await crawler._probe_endpoint(
            httpx.AsyncClient(base_url=BASE_URL),
            "unknown",
            "X",
            "vi",
        )
        assert result == "tree"
        assert crawler._endpoint_map["unknown"] == "tree"

    @pytest.mark.asyncio
    @respx.mock
    async def test_probe_skips_500_gracefully(self, crawler: ICDCrawler) -> None:
        """A 500 from /childs should NOT be used for control flow.

        probe should silently skip to /data.
        """
        respx.get(f"{BASE_URL}/childs/flaky?id=X1&lang=vi").respond(500)
        respx.get(f"{BASE_URL}/data/flaky?id=X1&lang=vi").respond(
            200,
            json={"id": "X1"},
        )

        result = await crawler._probe_endpoint(
            httpx.AsyncClient(base_url=BASE_URL),
            "flaky",
            "X1",
            "vi",
        )
        assert result == "data"
        assert crawler._endpoint_map["flaky"] == "data"

    @pytest.mark.asyncio
    @respx.mock
    async def test_probe_returns_none_when_all_fail(self, crawler: ICDCrawler) -> None:
        """All candidates return non-200 -> None."""
        respx.get(f"{BASE_URL}/childs/unknown?id=Z&lang=vi").respond(404)
        respx.get(f"{BASE_URL}/data/unknown?id=Z&lang=vi").respond(404)
        respx.get(f"{BASE_URL}/tree/unknown?id=Z&lang=vi").respond(404)

        result = await crawler._probe_endpoint(
            httpx.AsyncClient(base_url=BASE_URL),
            "unknown",
            "Z",
            "vi",
        )
        assert result is None
        assert "unknown" not in crawler._endpoint_map

    @pytest.mark.asyncio
    @respx.mock
    async def test_probe_uses_cache(self, crawler: ICDCrawler) -> None:
        """Second call for the same kind returns cached result without HTTP."""
        crawler._endpoint_map["chapter"] = "childs"
        result = await crawler._probe_endpoint(
            httpx.AsyncClient(base_url=BASE_URL),
            "chapter",
            "1",
            "vi",
        )
        assert result == "childs"
        # No routes mocked -> no HTTP calls made.


# -----------------------------------------------------------------------
# Child extraction
# -----------------------------------------------------------------------


class TestExtractChildren:
    """Test _extract_children with various response shapes."""

    def test_root_envelope(self) -> None:
        """API envelope {"status": "success", "data": [...]} is unwrapped."""
        children = ICDCrawler._extract_children(SAMPLE_ROOT_VI)
        assert len(children) == 2
        assert children[0] == ("chapter", "1")
        assert children[1] == ("chapter", "2")

    def test_root_plain_list(self) -> None:
        """Plain list (no envelope) also works."""
        data = [
            {"model": "chapter", "id": "A00-B99"},
        ]
        children = ICDCrawler._extract_children(data)
        assert children == [("chapter", "A00-B99")]

    def test_non_root_envelope_data_field(self) -> None:
        """/childs/ returns {"status": "success", "data": [...]} — unwrapped."""
        children = ICDCrawler._extract_children(SAMPLE_CHAPTER)
        assert len(children) == 2
        assert children[0] == ("section", "A00-A09")

    def test_non_root_children_field_fallback(self) -> None:
        """Dict with "children" field (not "data") still works."""
        data = {
            "id": "X",
            "children": [
                {"model": "section", "id": "Y"},
            ],
        }
        children = ICDCrawler._extract_children(data)
        assert children == [("section", "Y")]

    def test_node_with_subitems_field(self) -> None:
        data = {
            "id": "X",
            "subItems": [
                {"id": "Y", "model": "section"},
            ],
        }
        children = ICDCrawler._extract_children(data)
        assert children == [("section", "Y")]

    def test_leaf_node_no_children(self) -> None:
        children = ICDCrawler._extract_children(SAMPLE_DISEASE)
        assert children == []

    def test_empty_root(self) -> None:
        children = ICDCrawler._extract_children([])
        assert children == []

    def test_non_dict_non_list(self) -> None:
        children = ICDCrawler._extract_children("not a dict")
        assert children == []


# -----------------------------------------------------------------------
# Full crawl integration (mocked)
# -----------------------------------------------------------------------


class TestCrawl:
    """End-to-end crawl with mocked HTTP."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_crawl_with_limit(self, tmp_path: Path) -> None:
        """Crawl respects --limit and produces correct artefacts."""
        data_dir = tmp_path / "data"

        # Mock root
        respx.get(f"{BASE_URL}/root").mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=SAMPLE_ROOT_VI if "lang=vi" in str(req.url) else SAMPLE_ROOT_DUAL,
            ),
        )
        # Mock /childs/ for non-leaf node kinds (first candidate).
        respx.get(url__startswith=f"{BASE_URL}/childs/chapter").respond(
            200,
            json=SAMPLE_CHAPTER,
        )
        respx.get(url__startswith=f"{BASE_URL}/childs/section").respond(
            200,
            json=SAMPLE_SECTION,
        )
        respx.get(url__startswith=f"{BASE_URL}/childs/type").respond(
            200,
            json=SAMPLE_TYPE,
        )
        # Disease is a leaf: /childs/ returns 500, /data/ returns 200.
        respx.get(url__startswith=f"{BASE_URL}/childs/disease").respond(500)
        respx.get(url__startswith=f"{BASE_URL}/data/disease").respond(
            200,
            json=SAMPLE_DISEASE,
        )

        crawler = ICDCrawler(
            base_url=BASE_URL,
            output_dir=str(data_dir),
            concurrency=3,
            limit=10,
        )
        await crawler.crawl()

        assert crawler._request_count == 10
        assert len(crawler._manifest) == 10

        # Verify manifest written.
        manifest = json.loads((data_dir / "crawl_manifest.json").read_text())
        assert len(manifest) == 10

        # Verify discovery summary — limit=10 is deep enough to reach sections.
        discovery = json.loads((data_dir / "discovery_summary.json").read_text())
        assert discovery["endpoint_map"]["chapter"] == "childs"
        assert discovery["endpoint_map"]["section"] == "childs"

    @pytest.mark.asyncio
    @respx.mock
    async def test_resumability(self, tmp_path: Path) -> None:
        """Second crawl skips already-visited nodes."""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)

        # Pre-seed only ch1:vi — roots are NOT pre-seeded so they re-enqueue children.
        write_manifest(
            data_dir,
            [
                {
                    "kind": "chapter",
                    "id": "1",
                    "lang": "vi",
                    "endpoint": "childs",
                    "http_status": 200,
                    "request_count": 1,
                    "timestamp": "...",
                },
            ],
        )
        write_discovery(data_dir, {"chapter": "childs"})

        # Mock roots to return chapters (will be fetched since not pre-seeded).
        respx.get(f"{BASE_URL}/root").mock(
            side_effect=lambda req: httpx.Response(
                200,
                json=SAMPLE_ROOT_VI if "lang=vi" in str(req.url) else SAMPLE_ROOT_DUAL,
            ),
        )
        respx.get(url__startswith=f"{BASE_URL}/childs/chapter").respond(
            200,
            json=SAMPLE_CHAPTER,
        )
        respx.get(url__startswith=f"{BASE_URL}/childs/section").respond(404)
        respx.get(url__startswith=f"{BASE_URL}/data/section").respond(404)
        respx.get(url__startswith=f"{BASE_URL}/tree/section").respond(
            200,
            json=SAMPLE_SECTION,
        )

        crawler = ICDCrawler(
            base_url=BASE_URL,
            output_dir=str(data_dir),
            concurrency=3,
            limit=8,
        )
        await crawler.crawl()

        # 1 pre-seeded entry + new requests (roots + chapters).
        assert crawler._request_count > 1
        manifest = json.loads((data_dir / "crawl_manifest.json").read_text())
        # ch1:vi should appear exactly once (pre-seeded, not re-fetched).
        ch1_vi_entries = [
            e
            for e in manifest
            if e["kind"] == "chapter" and e["id"] == "1" and e["lang"] == "vi"
        ]
        assert len(ch1_vi_entries) == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_raw_files_structure(self, tmp_path: Path) -> None:
        """Verify raw JSON files are saved in the expected directory layout."""
        data_dir = tmp_path / "data"

        respx.get(f"{BASE_URL}/root").respond(200, json=SAMPLE_ROOT_VI)
        respx.get(url__startswith=f"{BASE_URL}/childs/chapter").respond(
            200,
            json=SAMPLE_CHAPTER,
        )

        crawler = ICDCrawler(
            base_url=BASE_URL,
            output_dir=str(data_dir),
            concurrency=2,
            limit=3,
        )
        await crawler.crawl()

        raw = data_dir / "raw"
        assert raw.exists()

        # Check root file.
        root_files = list(
            (raw / "endpoint=root" / "lang=vi" / "id=root").glob("*.json"),
        )
        assert len(root_files) == 1
        envelope = json.loads(root_files[0].read_text())
        assert envelope["endpoint_kind"] == "root"
        assert envelope["lang"] == "vi"
        assert envelope["http_status"] == 200
        assert "data" in envelope

    @pytest.mark.asyncio
    @respx.mock
    async def test_error_recording(self, tmp_path: Path) -> None:
        """Failed requests are recorded in crawl_errors.json."""
        data_dir = tmp_path / "data"

        # Root returns empty to avoid further crawling.
        respx.get(f"{BASE_URL}/root").respond(200, json=[])
        # All probes fail for a non-existent kind.
        respx.get(url__startswith=f"{BASE_URL}/childs/missing").respond(404)
        respx.get(url__startswith=f"{BASE_URL}/data/missing").respond(404)
        respx.get(url__startswith=f"{BASE_URL}/tree/missing").respond(404)

        crawler = ICDCrawler(
            base_url=BASE_URL,
            output_dir=str(data_dir),
            concurrency=2,
            limit=None,
        )
        await crawler.crawl()

        errors_file = data_dir / "crawl_errors.json"
        # Only root requests were made (empty response, no children).
        assert crawler._request_count == 2  # vi + dual
        assert errors_file.exists()


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


class TestCLI:
    def test_main_importable(self) -> None:
        assert callable(main)
