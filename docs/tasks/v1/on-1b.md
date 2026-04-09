# ON-1b: Extract Intro Guidance and Official 3-Char Policy from PDF

## Objective

Implement a silver-stage extractor that reads the bronze page-level ICD-10 PDF records from ON-1 and emits structured intro-guidance artifacts for downstream rule extraction and report generation.

The extractor must capture the official guidance for:

* principal diagnosis
* symptom fallback
* mortality coding
* official 3-character national default policy

## Dependency

This task depends on ON-1.

Use only this upstream artifact as input:

* `data/bronze/icd10_official_pdf/primary/document_records.jsonl`

Treat ON-1 output as the single source of truth for this task. Do not re-download the PDF inside ON-1b.

## Scope

Implement only intro-section guidance extraction from the official ICD-10 PDF content already parsed by ON-1.

Emit schema-valid records that preserve the exact supporting text and provide normalized text for downstream rule parsing.

Write outputs only under:

* `data/silver/icd10_official_pdf/intro_guidance/`

## Out of Scope

Do not parse ontology codes, chapter hierarchy, blocks, or 4-character entries.

Do not emit `OntologyCode`, `MentionRecord`, `CandidateLink`, `SilverRecord`, `PredictionRecord`, or `ExplanationRecord`.

Do not create a custom guidance schema.

Do not create a new data layer outside `data/{bronze,silver,gold}/...`.

Do not implement MoH website fallback or external scraping.

## Required Schema

Use only models already defined in `src/mnc/schemas/`.

For this task, the output artifact type is `DocumentRecord` from `src/mnc/schemas/document.py`.

This task intentionally uses `DocumentRecord` because no dedicated guidance schema exists in the repository.

## Input Contract

Read `DocumentRecord` rows from:

* `data/bronze/icd10_official_pdf/primary/document_records.jsonl`

Required assumptions about input:

* records are page-level
* `doc_id` format is `icd10_official_pdf:page:{page_no}`
* `source` is `icd10_official_pdf`
* `language` is `vi`

The extractor must fail loudly if the input file is missing, invalid, empty, or contains no intro guidance candidates.

## Output Contract

Emit one `DocumentRecord` per required guidance topic.

Write records to:

* `data/silver/icd10_official_pdf/intro_guidance/document_records.jsonl`

Parquet output is optional. If implemented, write to:

* `data/silver/icd10_official_pdf/intro_guidance/document_records.parquet`

Each output row must validate against `DocumentRecord`.

## Record Granularity

Use topic-level granularity.

Emit exactly one record for each required topic:

* `principal_diagnosis`
* `symptom_fallback`
* `mortality_coding`
* `official_3char_policy`

If evidence for one topic appears on multiple pages, concatenate the source-faithful excerpts in page order using `\n\n`.

If any required topic cannot be found, fail loudly.

## Topic Detection Rules

Use deterministic pattern-based extraction over the ON-1 page text.

Prioritize exact or near-exact official wording from the intro pages.

Use page-order scanning and explicit keyword triggers to identify candidate passages.

Keep extraction rules small, explicit, and auditable.

Do not paraphrase in `raw_text`.

Do not use an LLM-based extractor in this task.

## Normalization Rules

Apply only minimal normalization for `normalized_text` and `retrieval_text`:

* normalize line endings
* strip leading and trailing whitespace
* collapse repeated blank lines
* collapse repeated spaces
* preserve Vietnamese diacritics
* preserve source meaning

Do not rewrite policy content semantically.

## Field Mapping

Populate `DocumentRecord` fields as follows:

* `doc_id`: `icd10_official_pdf:intro_guidance:{topic}`
* `source`: `icd10_official_pdf`
* `language`: `vi`
* `raw_text`: concatenated source-faithful excerpt(s) for the topic
* `source_record_id`: `{topic}`
* `split`: `None`
* `payload`: flat scalar metadata only
* `normalized_text`: minimally normalized guidance text
* `retrieval_text`: same as `normalized_text`
* `sentences`: sentence split of `normalized_text`
* `created_at`: UTC timestamp at artifact creation time

## Payload Requirements

`payload` must remain compatible with `dict[str, JsonValue]`.

Do not store nested objects or arrays.

Required payload keys:

* `topic`
* `page_start`
* `page_end`
* `page_span`
* `extractor`
* `match_method`

Recommended payload values:

* `topic`: one of the four required topic IDs
* `page_start`: first page number containing supporting text
* `page_end`: last page number containing supporting text
* `page_span`: compact string such as `5-7`
* `extractor`: `rule_based_intro_guidance`
* `match_method`: `keyword_window`

## Storage Layout

Use only:

* `data/silver/icd10_official_pdf/intro_guidance/document_records.jsonl`
* `data/silver/icd10_official_pdf/intro_guidance/document_records.parquet` if implemented

Do not create extra artifacts outside this scope.

## Suggested Module Structure

Keep implementation small and task-focused.

Recommended pieces:

* bronze reader
* intro-page candidate selector
* topic matcher
* excerpt consolidator
* `DocumentRecord` builder
* JSONL writer
* optional Parquet writer
* CLI entrypoint

## Suggested Public API

```python
def extract_intro_guidance(
    input_path: str = "data/bronze/icd10_official_pdf/primary/document_records.jsonl",
    output_dir: str = "data/silver/icd10_official_pdf/intro_guidance",
) -> list"""Extract required intro guidance topics from ON-1 page records."""
```

## Validation

Validate every output record with `DocumentRecord` before writing.

Minimum validation checks:

* exactly 4 records are produced
* all `doc_id` values are unique
* all `source_record_id` values are unique
* all records have `source = "icd10_official_pdf"`
* all records have `language = "vi"`
* all required topics are present exactly once
* `raw_text` is non-empty for every topic
* `normalized_text` is non-empty for every topic
* `payload["topic"]` matches the topic in `doc_id`
* `page_start <= page_end`

## Logging

Log at least:

* input path
* number of bronze pages read
* matched page span per topic
* number of output records written
* output path

Do not persist custom logs as data artifacts.

## Testing

Use `pytest`.

Keep tests simple and focused.

Required unit tests:

* schema validation test for one generated guidance `DocumentRecord`
* topic `doc_id` formatting test
* missing-topic failure test
* multi-page excerpt consolidation test
* sentence splitting smoke test
* JSONL writer test
* extractor smoke test on mocked ON-1 page records
* deterministic output key test for repeated runs on the same input

Do not require network access in unit tests.

Use small mocked `DocumentRecord` inputs instead of the real PDF.

## Acceptance Criteria

The task is complete when all conditions below are satisfied:

* extractor reads only ON-1 bronze page records
* output is written only under `data/silver/icd10_official_pdf/intro_guidance/`
* output records are valid `DocumentRecord` instances
* exactly 4 topic-level records are emitted
* all required topics are captured
* source-faithful excerpts are preserved in `raw_text`
* normalized text is populated for downstream rule parsing
* no custom schema is introduced
* `pytest` unit tests exist and pass
