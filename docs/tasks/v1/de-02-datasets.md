# DE-2 Spec: Ingest Public Datasets

## Goal

Implement a unified ingestion pipeline for public Vietnamese medical datasets.

The pipeline must read raw source files, convert each sample into the DE-1 schema defined in `src/mnc/schemas`, and save reproducible ingestion artifacts for downstream tasks.

This task is only about ingestion. It must not perform text normalization, sentence segmentation, ontology linking, weak supervision, or model training.

## References

- DE-1 schema: `src/mnc/schemas`
- Data schema document: `docs/data/v1/de-01-schemas.md`

## Scope

The ingestion pipeline must support public datasets used in this project, including the currently selected sources such as VietMed-Sum and PET/CT.

The implementation should be source-extensible. Adding a new dataset should require a new adapter, not changes to the core ingestion flow.

## Inputs

- A dataset identifier
- A local input path or configured raw data directory
- Optional split information if provided by the source
- Optional source-specific metadata

## Outputs

- One DE-1-compliant record per ingested sample
- A line-delimited JSON or JSON file for ingested records
- A manifest file summarizing ingestion results
- An error file for skipped or failed samples
- Deterministic record IDs and content hashes when possible

## Required behavior

- Load raw public dataset files from disk
- Parse source-specific fields with a dataset adapter
- Map parsed samples into the DE-1 schema from `src/mnc/schemas`
- Preserve source provenance
- Preserve the original raw text needed by downstream tasks
- Preserve split if the source provides it
- Skip invalid rows safely and log them
- Continue ingestion when individual rows fail
- Produce stable output for repeated runs on the same input

## Non-goals

- No text normalization
- No sentence segmentation
- No ontology parsing
- No mention extraction
- No silver labeling
- No train/dev/test re-splitting beyond source-provided split

## Dataset adapter contract

Each dataset adapter should expose a small, uniform interface.

- `dataset_name() -> str`
- `iter_raw_samples(input_path) -> Iterable[object]`
- `to_de1_record(raw_sample) -> BaseModel`
- `validate_raw_sample(raw_sample) -> bool`

The adapter is responsible for source-specific parsing only.

The core pipeline is responsible for orchestration, validation, logging, and writing outputs.

## Minimum fields to populate

Populate all required DE-1 fields from `src/mnc/schemas`.

At minimum, each ingested record should preserve:

- record identity
- source dataset name
- source sample identifier if available
- language
- raw text
- split if available
- minimal metadata needed for provenance and debugging

Do not invent labels or ontology fields at this stage.

## File layout

Suggested output layout:

- `data/bronze/<dataset_name>/records.jsonl`
- `data/bronze/<dataset_name>/manifest.json`
- `data/bronze/<dataset_name>/errors.jsonl`

## Error handling

- Invalid samples must not stop the full run
- Every failed sample must be logged with dataset name, sample identifier if available, and error message
- The final manifest must include total rows, successful rows, failed rows, and output paths

## Determinism

- Re-running ingestion on the same input must produce the same record content except for allowed runtime metadata such as timestamps
- Record IDs should be deterministic when the source provides stable identifiers
- If the source has no stable identifier, derive one deterministically from content

## CLI or entrypoint

Provide one runnable entrypoint for ingestion.

Example behavior:

- accept dataset name
- accept input path
- accept output directory
- run the correct adapter
- write records, manifest, and errors
- print a short summary

## Acceptance criteria

- The pipeline ingests at least the initial target datasets required by the project
- Every output record validates against the DE-1 schema from `src/mnc/schemas`
- The pipeline writes records, manifest, and error logs
- Failed rows are isolated and logged without crashing the run
- The pipeline is deterministic on repeated runs
- The implementation is covered by pytest

## Pytest requirements

Tests must be added under `tests/`.

Minimum required test cases:

- test that each supported adapter can read a small fixture input
- test that each adapter returns DE-1-valid records
- test that required provenance fields are populated
- test that split is preserved when present
- test that missing optional fields do not crash ingestion
- test that malformed rows are logged and skipped
- test that the pipeline continues after row-level failures
- test that repeated runs on the same fixture produce identical records
- test that manifest counts are correct
- test that output files are created in the expected locations

## Test fixtures

Use tiny local fixtures only.

Each fixture should contain:

- one valid sample
- one sample with missing optional fields
- one malformed sample

Do not use network access in tests.

## Implementation notes

- Keep adapters small and explicit
- Keep the core pipeline schema-driven
- Validate records at the boundary before writing output
- Prefer pure functions for row mapping to simplify testing
- Keep source-specific logic out of the shared pipeline

## Definition of done

- ingestion code implemented
- adapters added for the selected public datasets
- outputs written in the expected bronze location
- pytest suite added and passing
- documentation updated with supported datasets and run command
