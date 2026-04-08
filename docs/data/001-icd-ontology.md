# ICD-10 Ontology by KCB

Task: Build a two-part data pipeline (Ingestion + Analytics) for the official Vietnamese ICD-10 ontology used by the KCB/MoH browser.

Known public API patterns:

- GET <https://ccs.whiteneuron.com/api/ICD10/root?lang=vi>
- GET <https://ccs.whiteneuron.com/api/ICD10/root?lang=dual>
- GET <https://ccs.whiteneuron.com/api/ICD10/data/chapter?id=><CHAPTER_ID>&lang=<LANG>
- GET <https://ccs.whiteneuron.com/api/ICD10/tree/section?id=><SECTION_ID>&lang=<LANG>
- GET <https://ccs.whiteneuron.com/api/ICD10/tree/type?id=><TYPE_ID>&lang=<LANG>
- GET <https://ccs.whiteneuron.com/api/ICD10/data/disease?id=><DISEASE_ID>&lang=<LANG>

Important context:

- The public KCB ICD-10 site is a hierarchical browser.
- Likely hierarchy: chapter -> section -> type -> disease, but do NOT hard-code that there are only four node kinds or four levels.
- ICD-10 may contain deeper descendants in some branches.

## PART 1: THE BRONZE CRAWLER

1. Crawl recursively and exhaustively from both roots (/root?lang=vi and /root?lang=dual).
2. Fetch payloads for chapters, sections, types, diseases, and any newly discovered child collections recursively. Do not assume disease is always terminal.
3. Save BRONZE raw data exactly as returned:
   - one raw JSON file per request.
   - preserve request URL, endpoint kind, id, lang, HTTP status, headers, retrieved_at.
4. Output layout for the crawler:

```text
data/bronze/
  moh_vn_icd10/
    raw/
      endpoint=<endpoint_kind>/lang=<lang>/id=<node_id>/<timestamp_or_hash>.json
    manifests/
      crawl_manifest.json (ledger: one row per request)
      crawl_errors.json (failed requests)
      discovery_summary.json (schema inference per endpoint, list of discovered node kinds)
```

5. Resilience: retry transient failures with exponential backoff, rate-limit politely, checkpoint progress, deduplicate repeated requests.

## PART 2: THE ANALYZER & REPORTER

6. Create a separate reporting script that processes the downloaded local JSON data:
   - Read the bronze raw JSONs and discovery_summary.json.
   - Aggregate coverage counts by node kind and language.
   - Detect duplicate ids with conflicting payloads across languages.
   - Calculate total chapters, sections, types, diseases, unknown kinds, and max tree depth.
   - Output these findings into a rich, structured `coverage_report.md`.

## DELIVERABLES & DOCUMENTATION

7. The agent must output the following files:
   - `crawler.py`: The ingestion script.
   - `reporter.py`: The analytics and markdown generation script.
   - `README.md`: Clear, step-by-step instructions explaining how to run the crawler first, and then how to run the reporter.

Implementation notes:

- STRICT SEPARATION OF CONCERNS: `crawler` must strictly handle HTTP requests, traversal, and saving JSONs. It must NOT generate markdown or text summaries. `reporter` must strictly read local files and generate the Markdown report. It must NOT make any HTTP requests.
- Prefer a recursive/queue-based crawler over a fixed-depth script.
