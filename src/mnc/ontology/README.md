# ICD-10 Ontology Crawler & Reporter

Two-part pipeline for the official Vietnamese ICD-10 ontology served by the KCB/MoH browser.

## Setup

```bash
uv sync --dev
```

## Crawler — `crawler.py`

Recursively crawls the KCB ICD-10 API with **dynamic endpoint discovery** (probing `/childs/<kind>`, `/data/<kind>`, and `/tree/<kind>`), saving raw bronze JSON.

```bash
# Full crawl (unlimited)
uv run python -m src.mnc.ontology.crawler

# Crawl with a request limit
uv run python -m src.mnc.ontology.crawler --limit 30

# Custom output directory
uv run python -m src.mnc.ontology.crawler path/to/output --limit 100

# Adjust concurrency
uv run python -m src.mnc.ontology.crawler --concurrency 3
```

### Key features

| Feature | Detail |
|---------|--------|
| Endpoint discovery | Probes `/childs/<kind>` then `/data/<kind>` then `/tree/<kind>` per node kind; caches first 200 |
| API envelope | Handles `{"status": "success", "data": [...]}` wrapper transparently |
| Resumability | Reads `crawl_manifest.json` on startup to skip visited nodes |
| Raw output | `<base>/raw/endpoint=<kind>/lang=<lang>/id=<id>/<hash>.json` |
| Metadata files | `crawl_manifest.json`, `crawl_errors.json`, `discovery_summary.json` |

### Discovered API patterns

| Node kind | Working endpoint | Response shape |
|-----------|-----------------|----------------|
| root | `GET /root?lang=<lang>` | `{"status": "success", "data": [chapters...]}` |
| chapter | `/childs/chapter?id=<id>&lang=<lang>` | `{"status": "success", "data": [sections...]}` |
| section | `/childs/section?id=<id>&lang=<lang>` | `{"status": "success", "data": [types...]}` |
| type | `/childs/type?id=<id>&lang=<lang>` | `{"status": "success", "data": [diseases...]}` |
| disease | `/data/disease?id=<id>&lang=<lang>` | `{"status": "success", "data": {details}}` (leaf) |

## Reporter — `reporter.py`

Standalone processor that reads local bronze JSON and generates `coverage_report.md`.

```bash
uv run python -m src.mnc.ontology.reporter
# or with custom data dir
uv run python -m src.mnc.ontology.reporter path/to/data
```

## Testing

```bash
uv run pytest tests/ -v
```
