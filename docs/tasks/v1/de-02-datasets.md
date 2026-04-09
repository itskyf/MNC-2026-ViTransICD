# DE-2: Snapshot public datasets to bronze

## Goal

Snapshot public datasets into immutable, source‑faithful bronze records.

This task reflects raw source ingestion only.

## Datasets

* [leduckhai/VietMed-Sum](https://huggingface.co/datasets/leduckhai/VietMed-Sum)
* [tarudesu/ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA)

## Layer & Storage

* Layer: bronze
* Output root: `data/bronze/<dataset_name>/snapshots/`
* One dataset per directory, grouped under `snapshots`
* One JSONL per split
* Manifest per dataset

## Scope

* Capture raw dataset rows exactly as provided by the source.
* Preserve original fields without semantic interpretation.
* Attach minimal ingestion metadata for lineage.

## Non-goals

* No document construction.
* No text normalization.
* No sentence segmentation.
* No field renaming for semantics.
* No doc\_id interpretation beyond source identity.

## Input

* Raw local dataset files downloaded previously.

## Output

* Snapshot records validating against a bronze snapshot schema in `src/mnc/schemas/`.

## Required record semantics

Each output record represents **one source row**, not necessarily one document.

Required fields:

* dataset\_name
* source\_split
* source\_record\_id
* payload
* source\_format
* source\_path
* ingest\_version

Optional fields:

* source\_url
* language
* raw\_checksum

## Output layout

* `data/bronze/<dataset_name>/snapshots/<split>.jsonl`
* `data/bronze/<dataset_name>/snapshots/manifest.json`

## Validation policy

* Fail fast on unreadable files.
* Skip records failing schema validation.
* Do not repair or coerce semantics.

## Acceptance criteria

* All output records are source‑faithful snapshots.
* No document‑level assumptions are introduced.
* Output validates against schemas in `src/mnc/schemas/`.
