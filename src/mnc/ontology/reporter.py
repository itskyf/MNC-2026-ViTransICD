"""ICD-10 Bronze Reporter — local analytics over crawled JSON data.

Reads bronze raw JSONs and ``discovery_summary.json``, then generates a
structured ``coverage_report.md``.  This script makes **no** HTTP requests.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from mnc.ontology._types import (
    DiscoverySummary,
    ErrorEntry,
    JsonValue,
    ManifestEntry,
    RawEnvelope,
)

logger = logging.getLogger(__name__)

_KNOWN_KINDS = {"chapter", "section", "type", "disease"}

_MAX_DUPLICATE_ROWS = 50
_MAX_ERROR_ROWS = 20


class ICDReporter:
    """Analyse local bronze data and produce a coverage report."""

    def __init__(self, data_dir: Path | str = "data/bronze/kcb_vn_icd10") -> None:
        """Initialise reporter with the bronze data directory."""
        self.data_dir = Path(data_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_report(self) -> str:
        """Read data, compute stats, write ``coverage_report.md``, return its path."""
        discovery_raw = self._read_json(self.data_dir / "discovery_summary.json")
        discovery: DiscoverySummary
        if isinstance(discovery_raw, dict):
            discovery = cast("DiscoverySummary", discovery_raw)
        else:
            discovery = DiscoverySummary(
                endpoint_map={},
                discovered_kinds=[],
                generated_at="",
            )

        manifest_raw = self._read_json(self.data_dir / "crawl_manifest.json")
        manifest: list[ManifestEntry]
        if isinstance(manifest_raw, list):
            manifest = cast("list[ManifestEntry]", manifest_raw)
        else:
            manifest = []

        errors_raw = self._read_json(self.data_dir / "crawl_errors.json")
        errors: list[ErrorEntry]
        if isinstance(errors_raw, list):
            errors = cast("list[ErrorEntry]", errors_raw)
        else:
            errors = []

        raw_dir = self.data_dir / "raw"
        raw_envelopes = self._read_all_raw(raw_dir)

        coverage = self._aggregate_coverage(manifest)
        duplicates = self._detect_duplicates(raw_envelopes)
        max_depth = self._compute_max_depth(manifest)
        kind_counts = self._count_by_kind(manifest)

        lines: list[str] = []
        self._render_header(lines, manifest, errors, max_depth)
        self._render_discovery_section(lines, discovery)
        self._render_node_counts_section(lines, kind_counts)
        self._render_coverage_section(lines, coverage)
        self._render_duplicates_section(lines, duplicates)
        self._render_errors_section(lines, errors)
        report = "\n".join(lines)

        out_path = self.data_dir / "coverage_report.md"
        out_path.write_text(report, encoding="utf-8")
        logger.info("Report written to %s", out_path)
        return str(out_path)

    # ------------------------------------------------------------------
    # Data readers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_json(path: Path) -> JsonValue:
        if not path.exists():
            return {}
        with path.open() as f:
            return json.load(f)

    @staticmethod
    def _read_all_raw(raw_dir: Path) -> list[RawEnvelope]:
        """Walk ``raw/`` and collect all envelope JSONs."""
        envelopes: list[RawEnvelope] = []
        if not raw_dir.exists():
            return envelopes
        for fp in sorted(raw_dir.rglob("*.json")):
            with fp.open() as f:
                envelopes.append(cast("RawEnvelope", json.load(f)))
        return envelopes

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_coverage(
        manifest: list[ManifestEntry],
    ) -> dict[str, dict[str, int]]:
        """Count unique nodes per (kind, lang)."""
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        seen: set[str] = set()
        for entry in manifest:
            key = f"{entry['lang']}:{entry['kind']}:{entry['id']}"
            if key not in seen:
                seen.add(key)
                counts[entry["kind"]][entry["lang"]] += 1
        return dict(counts)

    @staticmethod
    def _detect_duplicates(
        envelopes: list[RawEnvelope],
    ) -> list[dict[str, str]]:
        """Find node IDs with conflicting payloads across languages."""
        id_data: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
        for env in envelopes:
            node_id = env.get("id", "")
            lang = env.get("lang", "")
            data_str = json.dumps(env.get("data"), sort_keys=True)
            data_hash = hash(data_str)
            id_data[node_id][lang].add(data_hash)

        duplicates: list[dict[str, str]] = []
        for node_id, langs in id_data.items():
            if len(langs) > 1:
                all_hashes: set[int] = set()
                for lang_hashes in langs.values():
                    all_hashes |= lang_hashes
                if len(all_hashes) > 1:
                    duplicates.append(
                        {"id": node_id, "languages": ", ".join(sorted(langs))},
                    )
        return duplicates

    @staticmethod
    def _count_by_kind(manifest: list[ManifestEntry]) -> dict[str, int]:
        """Count total unique nodes per kind across all languages."""
        seen: set[str] = set()
        counts: dict[str, int] = defaultdict(int)
        for entry in manifest:
            key = f"{entry['kind']}:{entry['id']}"
            if key not in seen:
                seen.add(key)
                counts[entry["kind"]] += 1
        return dict(counts)

    @staticmethod
    def _compute_max_depth(manifest: list[ManifestEntry]) -> int:
        """Heuristic max tree depth from manifest order.

        The manifest is produced by BFS traversal, so the first occurrence of
        each new ``kind`` represents a deeper level.  We count the number of
        distinct non-root kinds encountered in order.
        """
        seen_kinds: list[str] = []
        for entry in manifest:
            kind = entry["kind"]
            if kind != "root" and kind not in seen_kinds:
                seen_kinds.append(kind)
        return len(seen_kinds)

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    def _render_header(
        self,
        lines: list[str],
        manifest: list[ManifestEntry],
        errors: list[ErrorEntry],
        max_depth: int,
    ) -> None:
        """Render title and overview section."""
        w = lines.append
        w("# ICD-10 Bronze Coverage Report")
        w("")
        w(f"Generated: {datetime.now(UTC).isoformat()}")
        w("")
        w("## Overview")
        w("")
        w(f"- **Total requests**: {len(manifest)}")
        w(f"- **Errors**: {len(errors)}")
        w(f"- **Max tree depth**: {max_depth}")
        w("")

    def _render_discovery_section(
        self,
        lines: list[str],
        discovery: DiscoverySummary,
    ) -> None:
        """Render endpoint discovery table."""
        w = lines.append
        w("## Endpoint Discovery")
        w("")
        emap = discovery.get("endpoint_map", {})
        if emap:
            w("| Node Kind | Endpoint Pattern |")
            w("|-----------|-----------------|")
            for kind in sorted(emap):
                w(f"| {kind} | `/{emap[kind]}/{kind}` |")
        else:
            w("_No endpoint discovery data._")
        w("")

    def _render_node_counts_section(
        self,
        lines: list[str],
        kind_counts: dict[str, int],
    ) -> None:
        """Render node counts by kind table."""
        w = lines.append
        w("## Node Counts by Kind")
        w("")
        if kind_counts:
            w("| Kind | Count |")
            w("|------|-------|")
            for kind in sorted(kind_counts):
                w(f"| {kind} | {kind_counts[kind]} |")
            unknown = {k: v for k, v in kind_counts.items() if k not in _KNOWN_KINDS}
            if unknown:
                w("")
                w(f"Unknown kinds: {', '.join(sorted(unknown))}")
        else:
            w("_No data._")
        w("")

    def _render_coverage_section(
        self,
        lines: list[str],
        coverage: dict[str, dict[str, int]],
    ) -> None:
        """Render coverage-by-language table."""
        w = lines.append
        w("## Coverage by Language")
        w("")
        if coverage:
            all_langs = sorted({lang for langs in coverage.values() for lang in langs})
            header = "| Kind | " + " | ".join(all_langs) + " |"
            sep = "|------|" + "|".join(["-------" for _ in all_langs]) + " |"
            w(header)
            w(sep)
            for kind in sorted(coverage):
                row = f"| {kind} |"
                for lang in all_langs:
                    row += f" {coverage[kind].get(lang, 0)} |"
                w(row)
        else:
            w("_No data._")
        w("")

    def _render_duplicates_section(
        self,
        lines: list[str],
        duplicates: list[dict[str, str]],
    ) -> None:
        """Render duplicates-across-languages section."""
        w = lines.append
        w("## Duplicates Across Languages")
        w("")
        if duplicates:
            w(
                f"Found **{len(duplicates)}** node(s) with conflicting "
                f"data across languages.",
            )
            w("")
            w("| ID | Languages |")
            w("|----|-----------|")
            for dup in duplicates[:_MAX_DUPLICATE_ROWS]:
                w(f"| {dup['id']} | {dup['languages']} |")
            if len(duplicates) > _MAX_DUPLICATE_ROWS:
                w(f"_... and {len(duplicates) - _MAX_DUPLICATE_ROWS} more._")
        else:
            w("No conflicting duplicates found across languages.")
        w("")

    def _render_errors_section(
        self,
        lines: list[str],
        errors: list[ErrorEntry],
    ) -> None:
        """Render errors summary section."""
        w = lines.append
        w("## Errors")
        w("")
        if errors:
            w(f"Total failed requests: **{len(errors)}**")
            w("")
            for err in errors[:_MAX_ERROR_ROWS]:
                kind_val = err.get("kind", "?")
                id_val = err.get("id", "?")
                lang_val = err.get("lang", "?")
                error_val = err.get("error", "unknown")
                w(f"- `{kind_val}/{id_val}` ({lang_val}): {error_val}")
            if len(errors) > _MAX_ERROR_ROWS:
                w(f"_... and {len(errors) - _MAX_ERROR_ROWS} more._")
        else:
            w("No errors recorded.")
        w("")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry point for the ICD-10 bronze reporter."""
    parser = argparse.ArgumentParser(description="ICD-10 Bronze Reporter")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/bronze/kcb_vn_icd10",
        help="Base data directory (default: data/bronze/kcb_vn_icd10)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    reporter = ICDReporter(data_dir=args.data_dir)
    reporter.generate_report()


if __name__ == "__main__":
    main()
