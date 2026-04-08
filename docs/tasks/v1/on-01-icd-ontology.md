# ICD-10 Ontology by icd.kcb.vn

Task: Build a two-part data pipeline (Ingestion + Analytics) for the official Vietnamese ICD-10 ontology used by the KCB/MoH browser.

Known public API patterns:

- GET `https://ccs.whiteneuron.com/api/ICD10/root?lang=vi`
- GET `https://ccs.whiteneuron.com/api/ICD10/root?lang=dual`
- GET `https://ccs.whiteneuron.com/api/ICD10/data/chapter?id=<CHAPTER_ID>&lang=<LANG>`
- GET `https://ccs.whiteneuron.com/api/ICD10/tree/section?id=<SECTION_ID>&lang=<LANG>`
- GET `https://ccs.whiteneuron.com/api/ICD10/tree/type?id=<TYPE_ID>&lang=<LANG>`
- GET `https://ccs.whiteneuron.com/api/ICD10/data/disease?id=<DISEASE_ID>&lang=<LANG>`

Important context:

- The public KCB ICD-10 site is a hierarchical browser.
- Likely hierarchy: chapter -\> section -\> type -\> disease. Do not hard-code that there are only four node kinds or four levels.
- ICD-10 may contain deeper descendants in some branches.

## PART 1: THE BRONZE CRAWLER

- Crawl recursively and exhaustively from both roots (/root?lang=vi and /root?lang=dual).
- Implement an asynchronous queue with a concurrency limit using asyncio.Semaphore.
- Fetch payloads for all discovered node kinds. Do not assume disease is always terminal.
- Extract child nodes strictly from structured JSON fields. Do not use ad-hoc regex to parse HTML strings for child IDs, as this captures chapter IDs as sections and causes server errors.
- Implement a dynamic endpoint discovery (probing) mechanism. When encountering a newly discovered node kind, do not hardcode its fetch strategy. Instead, probe available endpoint patterns (e.g., `/data/<model>` and `/tree/<model>`). Cache the successful pattern (HTTP 200) for all subsequent requests of that model kind. Do not use HTTP 500 errors for control flow.
- Save BRONZE raw data exactly as returned. Write one raw JSON file per request.
- Preserve request URL, endpoint kind, id, lang, HTTP status, headers, and retrieved\_at inside each saved JSON.
- Organize raw output as `<BASE_OUTPUT_DIR>/raw/endpoint=<endpoint_kind>/lang=<lang>/id=<node_id>/<hash>.json`.
- Maintain a `crawl_manifest.json` as a flat ledger with one row per request.
- Maintain a `crawl_errors.json` for failed requests that exhaust all retries.
- Maintain a `discovery_summary.json` containing schema inference, discovered node kinds, and the inferred endpoint strategy for each kind.
- Ensure resilience by retrying transient failures with exponential backoff, checkpointing progress, and deduplicating visited nodes.

## PART 2: THE ANALYZER & REPORTER

- Create a separate script that processes the downloaded local JSON data.
- Read the bronze raw JSONs and discovery\_summary.json.
- Aggregate coverage counts by node kind and language.
- Detect duplicate ids with conflicting payloads across languages.
- Calculate total chapters, sections, types, diseases, unknown kinds, and max tree depth.
- Output these findings into a rich, structured `coverage_report.md`.

## DELIVERABLES & DOCUMENTATION

- Output `crawler.py` for the ingestion script.
- Output `reporter.py` for the analytics and markdown generation script.
- Output `README.md` with clear instructions explaining how to run the crawler first, and then the reporter.
- Enforce strict separation of concerns. The crawler must strictly handle HTTP requests and JSON persistence.
- The reporter must strictly read local files and generate Markdown. It must not make HTTP requests.
- Accept a positional argument for the base data directory defaulting to `data/bronze/kcb_vn_icd10/`.
- Implement all code inside `src/mnc/ontology`.
