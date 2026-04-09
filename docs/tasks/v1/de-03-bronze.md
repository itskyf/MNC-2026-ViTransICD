
# DE-3 Spec: Parse source formats to bronze documents

## Goal

Convert bronze snapshot records into bronze **document‑level records**.

This task defines what constitutes a “document” per dataset, without any semantic transformation.

## Datasets

* [leduckhai/VietMed-Sum](https://huggingface.co/datasets/leduckhai/VietMed-Sum)
* [tarudesu/ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA)

## Layer & Storage

* Layer: bronze
* Output root: `data/bronze/<dataset_name>/documents/`
* DE‑3 outputs MUST coexist with DE‑2 outputs under the same dataset directory (separated by `snapshots/` and `documents/` groups).
* Do not introduce a new storage layer like `data/raw/`.
* Do not overwrite DE‑2 snapshot files.

## Scope

* Read DE‑2 snapshot JSONL files.
* Construct document‑level records using schemas from `src/mnc/schemas/`.
* Perform light structural validation only.

## Non-goals

* No normalization.
* No lowercasing.
* No removal of Vietnamese diacritics.
* No sentence or phrase segmentation.
* No retrieval view construction.
* No mention extraction.
* No ontology linking.
* No label generation.

These belong to DC‑1 or later.

## Input

* DE‑2 bronze snapshot records.
* Validated against existing snapshot schema in `src/mnc/schemas/`.

## Output

* Bronze document records validating against `DocumentRecord` (or the existing bronze document schema in `src/mnc/schemas/`).

## Document construction rules

General rules:

* One input snapshot record → one document.
* doc\_id must be deterministic and stable.
* source\_record\_id must be preserved.
* split must be preserved.
* created\_at must be injected once per run.

### VietMed-Sum

Dataset: [leduckhai/VietMed-Sum](https://huggingface.co/datasets/leduckhai/VietMed-Sum)

Input payload fields:

* transcript
* summary

Mapping:

* raw\_text = transcript
* payload retains transcript and summary
* doc\_id = "vietmed-sum:{source\_record\_id}"

Validation:

* transcript must be non‑empty string
* summary must be string

### ViHealthQA

Dataset: [tarudesu/ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA)

Input payload fields:

* question
* answer
* link
* id

Mapping:

* raw\_text = question + "\n" + answer
* payload retains question, answer, link
* doc\_id = "vihealthqa:{split}:{source\_record\_id}"

Validation:

* question non‑empty string
* answer non‑empty string

## Output layout

Under the existing bronze dataset directory (`data/bronze/<dataset_name>/`):

* DE‑2 snapshot files remain untouched in `snapshots/`.
* DE‑3 writes parsed document files using a grouped subdirectory:
  * `documents/<split>.jsonl`
  * `documents/manifest.json`

Do not invent a new root layer name (like `data/raw/`). Do not flatten files into `data/bronze/<dataset_name>/`.

## Manifest

Write a dataset‑level manifest containing:

* dataset
* input\_splits
* output\_splits
* record\_count\_by\_split
* failed\_count\_by\_split
* created\_at

Reuse an existing manifest schema if one exists.

## Error handling

* Invalid records are skipped.
* Write error entries to `errors.jsonl`.
* Pipeline must continue after record‑level failures.

## Acceptance criteria

* All output records validate against document schema.
* DE‑2 artifacts remain intact.
* No semantic transformation is applied.
* doc\_id is deterministic.
* Bronze directory layout uses group subdirectories under `data/bronze/<dataset_name>` (e.g., `snapshots`, `documents`).

## Minimal pytest coverage

Required:

* VietMed‑Sum happy path
* ViHealthQA happy path
* Invalid record error logging
* Schema validation failure
* Deterministic doc\_id
* Manifest generation

All tests local, no network.
