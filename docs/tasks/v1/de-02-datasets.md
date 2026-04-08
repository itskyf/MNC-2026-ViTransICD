# DE-2 Spec: Snapshot public datasets to raw

## Goal

* Implement a minimal ingestion task that snapshots the two public datasets into source-faithful records.
* Keep this task as raw snapshot semantics, but persist outputs under `data/bronze/<dataset_name>/...` because that is the current repo convention.
* Do not do text normalization, stripping, segmentation, mention extraction, or label construction here.

## Scope

* Dataset 1: [leduckhai/VietMed-Sum](https://huggingface.co/datasets/leduckhai/VietMed-Sum). It is a Hugging Face dataset for medical conversation summarization, stored as Parquet, with many predefined splits and at least `transcript` and `summary` fields visible in the viewer.
* Dataset 2: [tarudesu/ViHealthQA](https://huggingface.co/datasets/tarudesu/ViHealthQA). It is a Hugging Face dataset for Vietnamese medical QA, stored as CSV, with `train.csv`, `val.csv`, `test.csv`, and core fields `id`, `question`, `answer`, `link`.

## Non-goals

* No semantic cleanup.
* No lowercasing.
* No whitespace normalization beyond what is required to read the file safely.
* No deduplication across datasets.
* No train/val/test reshuffling.
* No ontology linking.
* No silver labels.

## Implementation rules

* Use Pydantic schemas from `src/mnc/schemas/`.
* Write one bronze snapshot per dataset split.
* Preserve source field names as much as possible.
* Add only minimal ingestion metadata needed for lineage and reproducibility.
* Fail fast on unreadable files or schema-invalid records.
* Keep the code small and deterministic.

## Output layout

* `data/bronze/vietmed_sum/<split>.jsonl`
* `data/bronze/vietmed_sum/manifest.json`
* `data/bronze/vihealthqa/<split>.jsonl`
* `data/bronze/vihealthqa/manifest.json`

## Required schema shape

* Use an existing Pydantic model from `src/mnc/schemas/` if one already matches bronze snapshot records.
* If no exact model exists, add a small bronze snapshot schema under `src/mnc/schemas/` and keep it generic.

### Required fields

* `dataset_name: str`
* `source_split: str`
* `source_record_id: str`
* `payload: dict[str, str | int | float | bool | None]`
* `source_format: str`
* `source_path: str`
* `ingest_version: str`

### Optional fields

* `source_url: str | None`
* `language: str | None`
* `raw_checksum: str | None`

## Dataset mapping

### vietmed_sum

* Read Hugging Face Parquet split files. The dataset exposes many split variants such as `train.train_30s`, `train.train_whole`, `test.test_full_dialogue`, and similar English variants. <https://huggingface.co/datasets/leduckhai/VietMed-Sum/tree/main/data>
* For the 12-hour plan, ingest only Vietnamese splits.
* For the 12-hour plan, ingest only these split variants: `train_whole`, `dev_whole`, `test_whole`.
* Preserve at least `transcript` and `summary` in `payload`, because those fields are visible in the dataset viewer. <https://huggingface.co/datasets/leduckhai/VietMed-Sum>
* `source_record_id` can be a deterministic hash of the raw row if no stable ID exists.

### vihealthqa

* Read `train.csv`, `val.csv`, `test.csv`. <https://huggingface.co/datasets/tarudesu/ViHealthQA/tree/main>
* Preserve `id`, `question`, `answer`, and `link` in `payload`. Those fields are visible in the dataset viewer and dataset card. <https://huggingface.co/datasets/tarudesu/ViHealthQA>, <https://huggingface.co/datasets/tarudesu/ViHealthQA/viewer/default?p=1>
* Use source `id` as `source_record_id`.
* Map `val.csv` to bronze split name `validation` to avoid ambiguity.

## Manifest content

* `dataset_name`
* `source_url`
* `source_format`
* `written_splits`
* `record_count_by_split`
* `error_count`
* `ingest_version`

## Acceptance criteria

* The task reads local raw files for both datasets and writes JSONL files under `data/bronze/<dataset_name>/`.
* Every output record validates against a Pydantic schema in `src/mnc/schemas/`.
* `vietmed_sum` writes only the selected Vietnamese `*_whole` splits.
* `vihealthqa` writes `train`, `validation`, and `test`.
* A manifest is produced for each dataset.
* No normalization or semantic transformation is applied.

## Minimal pytest coverage

* One happy-path test for `vietmed_sum`.
* One happy-path test for `vihealthqa`.
* One schema validation failure test.
* One manifest generation test.
* One deterministic `source_record_id` test for rows without native IDs.

## Test expectations

* Use small local fixture files only.
* Do not call the network in tests.
* Check that the output path and filenames are correct.
* Check that preserved source fields remain unchanged in `payload`.
* Check that `val.csv` becomes `validation`.

## Suggested deliverables

* `src/mnc/datasets/de2_snapshot.py`
* `src/mnc/schemas/bronze.py` (nếu cần)
* `tests/datasets/test_de2_snapshot_vietmed_sum.py`
* `tests/datasets/test_de2_snapshot_vihealthqa.py`
* `tests/datasets/test_de2_snapshot_manifest.py`
