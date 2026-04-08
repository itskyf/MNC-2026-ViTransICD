"""Coverage analyzer and markdown reporter for crawled ICD-10 data.

Reads local bronze JSON files and generates a structured coverage report.
Makes **no** HTTP requests.

Usage::

    uv run -m mnc.ontology.reporter [data_dir]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from tabulate import tabulate

from mnc.ontology._models import (
    ConflictDetail,
    CoverageStats,
    DiscoverySummary,
    DuplicateReport,
    ErrorEntry,
    ManifestEntry,
    ReportData,
)

logger = logging.getLogger(__name__)

_HTTP_OK = 200

# Fields to ignore during cross-language comparison (they legitimately differ).
_COMPARE_SKIP_FIELDS = frozenset({"html"})

# Maximum rows to render in conflict/error tables before truncation.
_TABLE_ROW_LIMIT = 50


def load_manifest(base_dir: Path) -> list[ManifestEntry]:
    """Load the crawl manifest from disk.

    Args:
        base_dir: Base crawl output directory.

    Returns:
        List of manifest entries.
    """
    path = base_dir / "manifests" / "crawl_manifest.json"
    if not path.exists():
        logger.error("Manifest not found: %s", path)
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_errors(base_dir: Path) -> list[ErrorEntry]:
    """Load crawl errors from disk.

    Args:
        base_dir: Base crawl output directory.

    Returns:
        List of error entries.
    """
    path = base_dir / "manifests" / "crawl_errors.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def load_discovery_summary(base_dir: Path) -> DiscoverySummary | None:
    """Load the discovery summary from disk.

    Args:
        base_dir: Base crawl output directory.

    Returns:
        Discovery summary dict, or None if not found.
    """
    path = base_dir / "manifests" / "discovery_summary.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw_records(raw_dir: Path) -> dict[str, dict[str, object]]:
    """Load all raw JSON files, keyed by ``'{endpoint_kind}:{lang}:{node_id}'``.

    Args:
        raw_dir: Path to the ``raw/`` directory.

    Returns:
        Mapping from dedup key to the parsed JSON record.
    """
    records: dict[str, dict[str, object]] = {}
    if not raw_dir.exists():
        return records

    for json_path in sorted(raw_dir.rglob("*.json")):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid JSON: %s", json_path)
            continue

        ekind = data.get("endpoint_kind")
        lang = data.get("lang")
        nid = data.get("node_id")
        if isinstance(ekind, str) and isinstance(lang, str):
            key = f"{ekind}:{lang}:{nid or '__root__'}"
            records[key] = data

    return records


def _derive_kind(ekind: str) -> str:
    """Derive the node kind from an endpoint kind string."""
    return "root" if ekind == "root" else ekind.split("_", maxsplit=1)[1]


def aggregate_coverage(manifest: list[ManifestEntry]) -> CoverageStats:
    """Count nodes by kind and language from the manifest.

    Args:
        manifest: List of manifest entries.

    Returns:
        Aggregated coverage statistics.
    """
    stats = CoverageStats()

    for entry in manifest:
        if entry["http_status"] != _HTTP_OK:
            continue

        kind = _derive_kind(entry["endpoint_kind"])
        lang = entry["lang"]
        key = (kind, lang)
        stats.by_kind_and_lang[key] = stats.by_kind_and_lang.get(key, 0) + 1
        stats.total_by_kind[kind] = stats.total_by_kind.get(kind, 0) + 1
        stats.total_by_lang[lang] = stats.total_by_lang.get(lang, 0) + 1
        stats.grand_total += 1

    return stats


def _extract_root_nodes(
    payload: dict[str, object],
    all_nodes: set[str],
) -> None:
    """Extract node IDs from a root response."""
    nodes = payload.get("data")
    if isinstance(nodes, list):
        for node in nodes:
            if isinstance(node, dict) and isinstance(node.get("id"), str):
                all_nodes.add(node["id"])


def _extract_tree_adjacency(
    payload: dict[str, object],
    adjacency: dict[str, list[str]],
    all_nodes: set[str],
    children_set: set[str],
) -> None:
    """Extract parent→child adjacency from a tree response."""
    tree_children = payload.get("children")
    if isinstance(tree_children, list):
        _walk_tree_nodes(tree_children, adjacency, all_nodes, children_set)


def _walk_tree_nodes(
    nodes: list[object],
    adjacency: dict[str, list[str]],
    all_nodes: set[str],
    children_set: set[str],
) -> None:
    """Recursively extract parent→child relationships from a tree node list."""
    for node in nodes:
        if not isinstance(node, dict):
            continue
        nid = node.get("id")
        if not isinstance(nid, str):
            continue
        all_nodes.add(nid)
        child_nodes = node.get("children")
        if isinstance(child_nodes, list):
            for child in child_nodes:
                if isinstance(child, dict) and isinstance(child.get("id"), str):
                    cid = child["id"]
                    adjacency.setdefault(nid, []).append(cid)
                    children_set.add(cid)
            _walk_tree_nodes(child_nodes, adjacency, all_nodes, children_set)


def compute_max_depth(raw_records: dict[str, dict[str, object]]) -> int:
    """Reconstruct the tree and compute maximum nesting depth.

    Builds a parent→children adjacency map from tree endpoint responses,
    then BFS-walks from root nodes to find the deepest path.

    Args:
        raw_records: Mapping from dedup key to parsed JSON record.

    Returns:
        Maximum depth (root = level 0).
    """
    adjacency: dict[str, list[str]] = {}
    all_nodes: set[str] = set()
    children_set: set[str] = set()

    for record in raw_records.values():
        ekind = record.get("endpoint_kind")
        if not isinstance(ekind, str):
            continue

        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue

        if ekind == "root":
            _extract_root_nodes(payload, all_nodes)
        elif ekind.startswith("tree_"):
            _extract_tree_adjacency(payload, adjacency, all_nodes, children_set)

    if not all_nodes:
        return 0

    # Root nodes are those that never appear as a child.
    roots = all_nodes - children_set
    if not roots:
        roots = all_nodes  # Fallback if adjacency is incomplete.

    depth_map: dict[str, int] = dict.fromkeys(roots, 0)
    queue = list(roots)

    while queue:
        current = queue.pop(0)
        current_depth = depth_map[current]
        for child in adjacency.get(current, []):
            depth_map[child] = max(depth_map.get(child, 0), current_depth + 1)
            queue.append(child)

    return max(depth_map.values()) if depth_map else 0


def detect_duplicates(
    raw_records: dict[str, dict[str, object]],
) -> DuplicateReport:
    """Find same-ID nodes across vi and dual languages with conflicting payloads.

    Compares the structured data fields (excluding ``html``) of each node that
    exists in both languages.

    Args:
        raw_records: Mapping from dedup key to parsed JSON record.

    Returns:
        Duplicate analysis report.
    """
    # Group records by node_id across languages.
    by_node_id: dict[str, dict[str, dict[str, object]]] = {}
    for record in raw_records.values():
        nid = record.get("node_id")
        lang = record.get("lang")
        if isinstance(nid, str) and isinstance(lang, str) and nid:
            by_node_id.setdefault(nid, {})[lang] = record

    report = DuplicateReport()

    for nid, lang_map in by_node_id.items():
        vi_rec = lang_map.get("vi")
        dual_rec = lang_map.get("dual")
        if vi_rec is None or dual_rec is None:
            continue

        report.total_shared_ids += 1

        vi_payload = vi_rec.get("payload")
        dual_payload = dual_rec.get("payload")
        if not isinstance(vi_payload, dict) or not isinstance(dual_payload, dict):
            continue

        conflicts = _compare_payloads(nid, vi_payload, dual_payload)
        if conflicts:
            report.conflicts.extend(conflicts)
        else:
            report.identical_payloads += 1

    return report


def _compare_payloads(
    node_id: str,
    vi_payload: dict[str, object],
    dual_payload: dict[str, object],
) -> list[ConflictDetail]:
    """Compare two payloads field-by-field, returning conflicts."""
    conflicts: list[ConflictDetail] = []

    vi_data = vi_payload.get("data")
    dual_data = dual_payload.get("data")

    if isinstance(vi_data, dict) and isinstance(dual_data, dict):
        all_keys = set(vi_data.keys()) | set(dual_data.keys())
        for field_name in sorted(all_keys):
            if field_name in _COMPARE_SKIP_FIELDS:
                continue
            vi_val = vi_data.get(field_name)
            dual_val = dual_data.get(field_name)
            if vi_val != dual_val:
                model = vi_data.get("model", dual_data.get("model", "unknown"))
                conflicts.append(
                    ConflictDetail(
                        node_id=node_id,
                        node_kind=str(model) if model else "unknown",
                        field_name=field_name,
                        vi_value=str(vi_val),
                        dual_value=str(dual_val),
                    ),
                )

    return conflicts


def _render_summary_section(data: ReportData, lines: list[str]) -> None:
    """Render the summary metrics table."""
    w = lines.append
    w("## Summary\n")
    total_errors = len(data.errors)
    langs = ", ".join(sorted(data.coverage.total_by_lang.keys())) or "none"
    w(
        tabulate(
            [
                ("Total successful requests", data.coverage.grand_total),
                ("Total errors", total_errors),
                ("Languages", langs),
                ("Max tree depth", data.max_depth),
            ],
            headers=["Metric", "Value"],
            tablefmt="github",
        ),
    )
    w("")


def _render_kind_coverage(data: ReportData, lines: list[str]) -> None:
    """Render the coverage-by-node-kind table."""
    w = lines.append
    w("## Coverage by Node Kind\n")
    all_kinds = sorted(data.coverage.total_by_kind.keys())
    all_langs = sorted(data.coverage.total_by_lang.keys())

    kind_rows = []
    for kind in all_kinds:
        row = [kind]
        row.extend(
            data.coverage.by_kind_and_lang.get((kind, lang), 0) for lang in all_langs
        )
        row.append(data.coverage.total_by_kind[kind])
        kind_rows.append(row)

    kind_headers = ["Node Kind", *all_langs, "Total"]
    w(tabulate(kind_rows, headers=kind_headers, tablefmt="github"))
    w("")


def _render_endpoint_coverage(data: ReportData, lines: list[str]) -> None:
    """Render the coverage-by-endpoint-kind table."""
    w = lines.append
    w("## Coverage by Endpoint Kind\n")
    if data.summary:
        ep_rows = sorted(data.summary["endpoint_kinds"].items())
        w(tabulate(ep_rows, headers=["Endpoint", "Requests"], tablefmt="github"))
    else:
        w("_No discovery summary available._")
    w("")


def _render_conflicts(data: ReportData, lines: list[str]) -> None:
    """Render the cross-language duplicate analysis and conflict details."""
    w = lines.append
    w("## Cross-Language Duplicate Analysis\n")
    w(
        tabulate(
            [
                ("Total shared IDs", data.duplicates.total_shared_ids),
                ("Identical structured payloads", data.duplicates.identical_payloads),
                ("Conflicting payloads", len(data.duplicates.conflicts)),
            ],
            headers=["Metric", "Count"],
            tablefmt="github",
        ),
    )
    w("")

    if data.duplicates.conflicts:
        w("### Conflicts\n")
        conflict_rows = [
            (c.node_id, c.node_kind, c.field_name, c.vi_value, c.dual_value)
            for c in data.duplicates.conflicts[:_TABLE_ROW_LIMIT]
        ]
        w(
            tabulate(
                conflict_rows,
                headers=["Node ID", "Kind", "Field", "vi Value", "dual Value"],
                tablefmt="github",
            ),
        )
        remaining = len(data.duplicates.conflicts) - _TABLE_ROW_LIMIT
        if remaining > 0:
            w(f"\n_...and {remaining} more conflicts._")
        w("")


def _render_errors(data: ReportData, lines: list[str]) -> None:
    """Render the errors table."""
    if not data.errors:
        return
    w = lines.append
    w("## Errors\n")
    err_rows = [
        (e["url"][:80], e["error_type"], e["error_message"][:60])
        for e in data.errors[:_TABLE_ROW_LIMIT]
    ]
    w(
        tabulate(
            err_rows,
            headers=["URL", "Error Type", "Message"],
            tablefmt="github",
        ),
    )
    remaining = len(data.errors) - _TABLE_ROW_LIMIT
    if remaining > 0:
        w(f"\n_...and {remaining} more errors._")
    w("")


def _render_discovery_summary(data: ReportData, lines: list[str]) -> None:
    """Render the discovery summary section."""
    if not data.summary:
        return
    w = lines.append
    w("## Discovery Summary\n")
    w("### Discovered Node Kinds\n")
    nk_rows = sorted(data.summary["node_kinds"].items())
    w(tabulate(nk_rows, headers=["Node Kind", "Count"], tablefmt="github"))
    w("")

    w("### Crawl Duration\n")
    started, completed = data.summary["crawled_at_range"]
    w(f"- Started: {started}")
    w(f"- Completed: {completed}")
    w("")


def generate_markdown_report(data: ReportData) -> str:
    """Render the full coverage report as markdown.

    Args:
        data: Bundled report inputs.

    Returns:
        Markdown string.
    """
    lines: list[str] = []
    w = lines.append

    w("# ICD-10 Ontology Crawl Coverage Report\n")
    w(f"Generated: {datetime.now(tz=UTC).isoformat()}")
    w(f"Base directory: `{data.base_dir}`\n")

    _render_summary_section(data, lines)
    _render_kind_coverage(data, lines)
    _render_endpoint_coverage(data, lines)

    # Tree depth.
    w("## Tree Depth Analysis\n")
    w(f"- Maximum depth: **{data.max_depth}** levels\n")

    _render_conflicts(data, lines)
    _render_errors(data, lines)
    _render_discovery_summary(data, lines)

    return "\n".join(lines)


def main(data_dir: str) -> None:
    """CLI entry point: read local data, generate the markdown report.

    Args:
        data_dir: Base directory containing crawled data.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    base = Path(data_dir).resolve()

    manifest = load_manifest(base)
    if not manifest:
        logger.error("No manifest entries found. Run the crawler first.")
        sys.exit(1)

    errors = load_errors(base)
    summary = load_discovery_summary(base)
    raw_records = load_raw_records(base / "raw")

    coverage = aggregate_coverage(manifest)
    max_depth = compute_max_depth(raw_records)
    duplicates = detect_duplicates(raw_records)

    data = ReportData(
        base_dir=str(base),
        coverage=coverage,
        max_depth=max_depth,
        duplicates=duplicates,
        errors=errors,
        summary=summary,
    )
    report = generate_markdown_report(data)

    report_path = base / "coverage_report.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate coverage report from crawled ICD-10 data.",
    )
    parser.add_argument(
        "data_dir",
        nargs="?",
        default="data/bronze/kcb_vn_icd10/",
        help="Base directory containing crawled data.",
    )
    args = parser.parse_args()
    main(args.data_dir)
