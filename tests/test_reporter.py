"""Unit tests for src/mnc/ontology/reporter.py."""

# ruff: noqa: D101, D102, D103, PLR2004, S324, SLF001, TC001, TC003

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.mnc.ontology._types import JsonValue
from src.mnc.ontology.reporter import ICDReporter

from .conftest import (
    SAMPLE_CHAPTER,
    make_raw_envelope,
    write_discovery,
    write_errors,
    write_manifest,
)

# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    d = tmp_path / "reporter_data"
    d.mkdir()
    return d


def _write_raw(data_dir: Path, envelope: dict[str, JsonValue]) -> None:
    """Write a raw envelope to the expected path layout."""
    ek = str(envelope["endpoint_kind"])
    lang = str(envelope["lang"])
    nid = str(envelope["id"])
    content = json.dumps(envelope["data"], sort_keys=True)

    h = hashlib.md5(content.encode()).hexdigest()[:12]
    raw_dir = data_dir / "raw" / f"endpoint={ek}" / f"lang={lang}" / f"id={nid}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{h}.json").write_text(json.dumps(envelope, ensure_ascii=False))


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestAggregateCoverage:
    def test_counts_by_kind_and_lang(self) -> None:
        manifest = [
            {"kind": "root", "id": "root", "lang": "vi"},
            {"kind": "root", "id": "root", "lang": "dual"},
            {"kind": "chapter", "id": "1", "lang": "vi"},
            {"kind": "chapter", "id": "1", "lang": "dual"},
            {"kind": "chapter", "id": "2", "lang": "vi"},
        ]
        result = ICDReporter._aggregate_coverage(manifest)
        assert result["root"]["vi"] == 1
        assert result["root"]["dual"] == 1
        assert result["chapter"]["vi"] == 2
        assert result["chapter"]["dual"] == 1

    def test_deduplication(self) -> None:
        """Same (lang, kind, id) should be counted once."""
        manifest = [
            {"kind": "chapter", "id": "1", "lang": "vi"},
            {"kind": "chapter", "id": "1", "lang": "vi"},
        ]
        result = ICDReporter._aggregate_coverage(manifest)
        assert result["chapter"]["vi"] == 1


class TestComputeMaxDepth:
    def test_depth_from_manifest(self) -> None:
        manifest = [
            {"kind": "root", "id": "root", "lang": "vi"},
            {"kind": "chapter", "id": "1", "lang": "vi"},
            {"kind": "section", "id": "A00", "lang": "vi"},
            {"kind": "type", "id": "A00", "lang": "vi"},
            {"kind": "disease", "id": "A00.0", "lang": "vi"},
        ]
        assert ICDReporter._compute_max_depth(manifest) == 4

    def test_depth_zero_when_only_root(self) -> None:
        manifest = [{"kind": "root", "id": "root", "lang": "vi"}]
        assert ICDReporter._compute_max_depth(manifest) == 0

    def test_depth_ignores_duplicate_kinds(self) -> None:
        manifest = [
            {"kind": "root", "id": "root", "lang": "vi"},
            {"kind": "chapter", "id": "1", "lang": "vi"},
            {"kind": "chapter", "id": "2", "lang": "vi"},
            {"kind": "section", "id": "A00", "lang": "vi"},
        ]
        assert ICDReporter._compute_max_depth(manifest) == 2


class TestCountByKind:
    def test_cross_lang_dedup(self) -> None:
        manifest = [
            {"kind": "chapter", "id": "1", "lang": "vi"},
            {"kind": "chapter", "id": "1", "lang": "dual"},
            {"kind": "chapter", "id": "2", "lang": "vi"},
        ]
        result = ICDReporter._count_by_kind(manifest)
        assert result["chapter"] == 2  # ids 1 and 2 (deduped across langs)


class TestFullReport:
    def test_generates_markdown(self, data_dir: Path) -> None:
        write_manifest(
            data_dir,
            [
                {
                    "kind": "root",
                    "id": "root",
                    "lang": "vi",
                    "endpoint": "root",
                    "http_status": 200,
                    "request_count": 1,
                    "timestamp": "...",
                },
                {
                    "kind": "root",
                    "id": "root",
                    "lang": "dual",
                    "endpoint": "root",
                    "http_status": 200,
                    "request_count": 2,
                    "timestamp": "...",
                },
                {
                    "kind": "chapter",
                    "id": "1",
                    "lang": "vi",
                    "endpoint": "data",
                    "http_status": 200,
                    "request_count": 3,
                    "timestamp": "...",
                },
                {
                    "kind": "chapter",
                    "id": "1",
                    "lang": "dual",
                    "endpoint": "data",
                    "http_status": 200,
                    "request_count": 4,
                    "timestamp": "...",
                },
            ],
        )
        write_errors(data_dir, [])
        write_discovery(data_dir, {"chapter": "data"})

        # Write raw files.
        _write_raw(
            data_dir,
            make_raw_envelope(
                endpoint_kind="data",
                node_id="1",
                lang="vi",
                data=SAMPLE_CHAPTER,
            ),
        )
        _write_raw(
            data_dir,
            make_raw_envelope(
                endpoint_kind="data",
                node_id="1",
                lang="dual",
                data=SAMPLE_CHAPTER,
            ),
        )

        reporter = ICDReporter(data_dir=data_dir)
        out = reporter.generate_report()
        assert out.endswith("coverage_report.md")

        content = (data_dir / "coverage_report.md").read_text()
        assert "ICD-10 Bronze Coverage Report" in content
        assert "chapter" in content
        assert "Endpoint Discovery" in content

    def test_empty_data(self, data_dir: Path) -> None:
        """Reporter handles missing files gracefully."""
        reporter = ICDReporter(data_dir=data_dir)
        reporter.generate_report()
        content = (data_dir / "coverage_report.md").read_text()
        assert "No data" in content
