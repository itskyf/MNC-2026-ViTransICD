# DE-3 Spec: Parse source formats to bronze

## Goal

Implement the first structured bronze transformation from DE-2 snapshot records into source-faithful bronze documents.

This task converts dataset-specific snapshot payloads into the existing bronze document schema defined in `src/mnc/schemas/`.

The output must remain source-faithful. Do not normalize text, segment sentences, extract mentions, generate labels, or add retrieval views here.

## Scope

Parse DE-2 snapshot JSONL files and write bronze document records for the supported datasets:

* VietMed-Sum
* ViHealthQA

Use only schema classes already defined in `src/mnc/schemas/` for both input validation and output validation.

This task is a light structural transformation only.

## Non-goals

The following are explicitly out of scope for DE-3:

* Text normalization
* Lowercasing
* Vietnamese diacritic removal
* Sentence or phrase segmentation
* Mention extraction
* Abbreviation expansion
* Ontology linking
* Candidate generation
* Silver label construction
* Deduplication across datasets
* Data quality repair beyond light structural validation

Notes from the draft such as `norm_text`, `retrieval_text`, and segmentation belong to DC-1, not DE-3.

## Input contract

Read only DE-2 outputs.

The input records must be validated against the existing snapshot schema in `src/mnc/schemas/`. Do not redefine a parallel input model.

Expected input source:

* One JSONL file per dataset split produced by DE-2
* One manifest file per dataset produced by DE-2

Expected input semantics:

* Source-faithful snapshot
* Original dataset fields preserved in `payload`
* Stable `source_record_id`
* Split-level lineage preserved

## Output contract

Write only records that validate against the existing bronze document schema in `src/mnc/schemas/`.

Do not add new schema fields for this task unless the current schema package truly lacks a bronze document model. If a new schema is unavoidable, add it under `src/mnc/schemas/` and keep it minimal.

The output record must remain source-faithful and preserve provenance.

## Bronze mapping rules

### Common rules

For every valid input snapshot record:

* `source` must be the canonical dataset name from the input schema.
* `source_record_id` must be carried over unchanged from DE-2.
* `split` must be carried over unchanged from DE-2.
* `language` must be copied from input if present.
* `payload` must preserve the original source fields required by the dataset.
* `created_at` must be set once per run and reused for all records in that run.
* `doc_id` must be deterministic and stable across reruns.

Do not mutate source field values except for safe string coercion needed to satisfy the schema.

### VietMed-Sum

Input payload fields expected from DE-2:

* `transcript`
* `summary`

Output mapping:

* `raw_text = payload["transcript"]`
* `payload` must preserve both `transcript` and `summary`
* `doc_id = "vietmed-sum:{source_record_id}"`

Validation rules:

* `transcript` must be a non-empty string
* `summary` must be a string
* reject the record if required fields are missing or invalid

### ViHealthQA

Input payload fields expected from DE-2:

* `id`
* `question`
* `answer`
* `link`

Output mapping:

* `raw_text = question + "\n" + answer`
* `payload` must preserve `question`, `answer`, and `link`
* preserve `id` if the bronze schema supports it via `payload`
* `doc_id = "vihealthqa:{split}:{source_record_id}"`

Validation rules:

* `question` must be a non-empty string
* `answer` must be a non-empty string
* `link` may be empty or null
* reject the record if required fields are missing or invalid

## Validation policy

Apply only light validation.

A record is valid if:

* it passes input schema validation
* required dataset-specific payload fields exist
* required text fields are strings
* required text fields are non-empty where specified
* the output bronze record passes output schema validation

Fail fast on malformed files.

Skip invalid records and log them to an error file with enough context for debugging.

## Output layout

Use a separate output root from DE-2 to avoid mixing snapshot artifacts with parsed bronze documents.

Recommended layout:

* `data/bronze_docs/vietmed-sum/train_whole.jsonl`
* `data/bronze_docs/vietmed-sum/dev_whole.jsonl`
* `data/bronze_docs/vietmed-sum/test_whole.jsonl`
* `data/bronze_docs/vietmed-sum/manifest.json`
* `data/bronze_docs/vietmed-sum/errors.jsonl`
* `data/bronze_docs/vihealthqa/train.jsonl`
* `data/bronze_docs/vihealthqa/validation.jsonl`
* `data/bronze_docs/vihealthqa/test.jsonl`
* `data/bronze_docs/vihealthqa/manifest.json`
* `data/bronze_docs/vihealthqa/errors.jsonl`

If the repository already has a standardized bronze parsed path, use that path instead. Do not overwrite DE-2 snapshot files in place.

## Manifest requirements

Write one manifest per dataset.

Minimum manifest fields:

* `dataset`
* `input_root`
* `output_root`
* `written_splits`
* `record_count_by_split`
* `failed_count_by_split`
* `total_rows`
* `successful_rows`
* `failed_rows`
* `created_at`

If a manifest schema already exists in `src/mnc/schemas/`, use it. Otherwise reuse the current internal manifest model already used by dataset pipeline code.

## Implementation requirements

Keep the implementation small, deterministic, and dataset-aware.

Recommended structure:

* `src/mnc/datasets/de3_parse.py`
* `src/mnc/datasets/de3_cli.py`

Implementation rules:

* Reuse existing dataset adapter patterns where practical.
* Read DE-2 JSONL records locally only.
* Validate input and output using Pydantic models from `src/mnc/schemas/`.
* Write JSONL outputs with UTF-8 encoding.
* Keep record order stable within each split.
* Use deterministic `doc_id` construction only.
* Do not call the network.
* Do not introduce heavy preprocessing logic.

## Error handling

Write invalid records to `errors.jsonl`.

Each error entry should include:

* `dataset`
* `split`
* `source_record_id`
* `error`

The pipeline must continue after record-level validation failures, but must stop on unreadable input files or schema import failures.

## Acceptance criteria

The task is complete when all conditions below are true:

* DE-3 reads DE-2 snapshot JSONL files for both datasets.
* Every written bronze record validates against the output schema in `src/mnc/schemas/`.
* VietMed-Sum bronze records use `transcript` as `raw_text`.
* ViHealthQA bronze records use `question + "\n" + answer` as `raw_text`.
* Source-faithful fields are preserved in `payload`.
* Split names remain unchanged from DE-2.
* A manifest is written for each dataset.
* An error log is written for each dataset.
* No normalization, segmentation, or mention extraction is applied.

## Minimal pytest coverage

Unit tests are mandatory and must use `pytest`.

Keep tests small and local. No network calls. No large fixtures.

Required tests:

* one happy-path test for VietMed-Sum parsing
* one happy-path test for ViHealthQA parsing
* one invalid payload test that produces an error entry
* one schema validation test for output records
* one deterministic `doc_id` test
* one manifest generation test

## Test expectations

Each test should verify only the task scope.

Minimum assertions:

* correct output filenames per split
* correct `raw_text` mapping for each dataset
* preserved source fields in `payload`
* stable `doc_id`
* invalid records are excluded from output and written to `errors.jsonl`
* manifest counts match written records

## Suggested deliverables

* `src/mnc/datasets/de3_parse.py`
* `src/mnc/datasets/de3_cli.py`
* `tests/datasets/test_de3_parse_vietmed_sum.py`
* `tests/datasets/test_de3_parse_vihealthqa.py`
* `tests/datasets/test_de3_parse_errors.py`
* `tests/datasets/test_de3_parse_manifest.py`

## One implementation rule

If a field is not defined in `src/mnc/schemas/`, do not invent it for DE-3.
