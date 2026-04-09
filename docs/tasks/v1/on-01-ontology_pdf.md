# ON-01: Parse Official ICD-10 PDF (Primary)

## Objective

Implement a bronze-stage parser that reads the official ICD-10 PDF from the provided URL and emits source-faithful `DocumentRecord` artifacts.

This task exists to make the PDF content available to downstream agents in a schema-valid, auditable format.

## Source

Use only this PDF as the source:

`https://soyte.laichau.gov.vn/upload/20762/20191121/ICD-10--tap-1_b55f6.pdf`

Do not implement any MoH website fallback.

## Scope

Implement only the bronze ingestion/parsing layer for the official PDF.

Produce page-level `DocumentRecord` outputs from the PDF text.

Preserve page fidelity and page numbering for downstream ontology parsing.

Write outputs under the required data layout:

`data/bronze/icd10_official_pdf/primary/`

## Out of Scope

Do not parse ICD hierarchy, chapters, blocks, 3-character codes, or 4-character codes into ontology structures.

Do not emit `OntologyCode`, `CandidateLink`, `SilverRecord`, `PredictionRecord`, `MentionRecord`, or `ExplanationRecord`.

Do not normalize text beyond minimal whitespace cleanup needed for readable extraction.

Do not create any storage layer outside `data/{bronze,silver,gold}/...`.

Do not scrape or patch from any website.

## Required Schema

Use only models already defined in `src/mnc/schemas/`.

For this task, the output artifact type is `DocumentRecord` from `src/mnc/schemas/document.py`.

The implementation must populate fields exactly as defined by the schema.

## Input Contract

The parser must accept the official PDF as either:

* a direct URL, defaulting to the source above
* a local PDF path for reproducible offline reruns

The parser must treat the PDF as the single source of truth.

## Output Contract

Emit one `DocumentRecord` per PDF page.

Write the records to:

* `data/bronze/icd10_official_pdf/primary/document_records.jsonl`

Parquet output is optional. If implemented, write to:

* `data/bronze/icd10_official_pdf/primary/document_records.parquet`

Each output row must validate against `DocumentRecord`.

## Record Granularity

Use page-level granularity, not whole-document granularity.

Rationale:

* preserves provenance
* keeps page boundaries auditable
* makes downstream ontology parsing easier
* avoids custom page-offset metadata outside the schema

## Field Mapping

Populate `DocumentRecord` fields as follows:

* `doc_id`: stable page key in the format `icd10_official_pdf:page:{page_no}`
* `source`: `icd10_official_pdf`
* `language`: `vi`
* `raw_text`: extracted text for that page, source-faithful except minimal whitespace cleanup
* `source_record_id`: page number as string
* `split`: `None`
* `payload`: flat scalar metadata only
* `normalized_text`: empty string
* `retrieval_text`: empty string
* `sentences`: empty list
* `created_at`: UTC timestamp at artifact creation time

## Payload Requirements

`payload` must remain compatible with `dict[str, JsonValue]`.

Do not store nested objects or arrays.

Recommended keys:

* `page_no`
* `pdf_url`
* `extractor`
* `is_empty_page`

Optional keys:

* `pdf_sha256`
* `total_pages`

## Extraction Rules

Extract text page by page.

Keep page order exactly as in the PDF.

Apply only minimal cleanup:

* normalize line endings
* strip leading and trailing whitespace
* collapse repeated blank lines if needed
* preserve Vietnamese diacritics
* do not rewrite content semantically

If a page yields no text, still emit a `DocumentRecord` with empty `raw_text` and `payload["is_empty_page"] = True`.

## Parser Behavior

Use `pymupdf` as the PDF text extraction library.

You must use the skill `get-api-docs` to retrieve the documentation and correct usage patterns for `pymupdf` before implementing the text extraction logic.

The parser must fail loudly if the PDF cannot be opened or if zero pages are extracted.

## Storage Layout

Use only the following bronze directory:

`data/bronze/icd10_official_pdf/primary/`

Allowed artifacts for this task:

* `document_records.jsonl`
* `document_records.parquet` if implemented

Do not create `data/raw/` or any extra storage layer.

## Idempotency

The job must be deterministic for the same input PDF.

Re-running the job must produce the same `doc_id` values and the same page count.

`created_at` may differ across runs.

## Validation

Validate every record with `DocumentRecord` before writing.

Reject records with schema violations.

Minimum validation checks:

* page count greater than zero
* `doc_id` unique across output
* `source_record_id` unique across output
* page numbering contiguous from 1 to total pages
* all records have `source = "icd10_official_pdf"`
* all records have `language = "vi"`

## Logging

Log at least:

* source path or URL
* total pages discovered
* total records written
* number of empty pages
* output path

Do not persist custom log artifacts into another data layer.

## Suggested Module Structure

Keep implementation small and task-focused.

Recommended pieces:

* PDF loader
* page text extractor
* `DocumentRecord` builder
* JSONL writer
* optional Parquet writer
* CLI entrypoint

## Suggested Public API

The Coding Agent can expose a minimal API like this:

```python
def parse_icd10_official_pdf(
    pdf_source: str,
    output_dir: str = "data/bronze/icd10_official_pdf/primary",
) -> list[DocumentRecord]:
    """Parse the official ICD-10 PDF into page-level DocumentRecord artifacts."""
```

## Testing

Use `pytest`.

Keep tests simple and focused.

Required unit tests:

* schema validation test for one generated `DocumentRecord`
* page-level `doc_id` formatting test
* empty-page handling test
* writer test for JSONL output
* parser smoke test on a small local sample PDF or a mocked extractor output
* idempotent key generation test for repeated runs on the same page numbers

Do not require network access in unit tests.

If the implementation supports URL download, isolate that logic and mock it in tests.

## Acceptance Criteria

The task is complete when all conditions below are satisfied:

* parser reads the official PDF from the provided source or local equivalent
* output is written only under `data/bronze/icd10_official_pdf/primary/`
* output records are valid `DocumentRecord` instances
* one record is emitted per PDF page
* page order and page numbering are preserved
* no MoH web fallback exists in code
* no ontology or silver/gold logic is mixed into this task
* `pytest` unit tests exist and pass

## Non-Functional Requirements

Keep the implementation simple, deterministic, and auditable.

Prefer clear functions over premature abstraction.

Avoid introducing dependencies unrelated to PDF parsing or schema serialization.

## Deliverable

A bronze parser that converts the official ICD-10 PDF into page-level `DocumentRecord` JSONL artifacts for downstream ontology parsing.
