# BM-2: TF-IDF/BM25 baseline over bilingual PDF ontology

## Objective

Implement lexical retrieval baselines over the official bilingual PDF ontology using TF-IDF and BM25.

This task must produce two baseline variants:

* BM2-TFIDF
* BM2-BM25

## Scope

In scope:

* Read silver documents from DC-1
* Read ON-2 ontology
* Build lexical indexes over bilingual ontology text
* Query each document
* Produce ranked `PredictionRecord` outputs for TF-IDF and BM25
* Optionally compute metrics when target labels are provided

Out of scope:

* Reranking
* Neural retrieval
* New persisted schema types
* Mention-level candidate persistence

## Required Data Structures

Use only existing schemas:

* `src/mnc/schemas/prediction.py::PredictionRecord`
* `src/mnc/schemas/silver.py::SilverRecord` for optional evaluation input
* `src/mnc/schemas/ontology.py::OntologyCode`

Use ON-2 fields only:

* `code_3char`
* `title_vi`
* `title_en`
* `aliases`
* `search_text`

## Input Contract

Required:

* `data/silver/<dataset_name>/documents/<split>.jsonl`
* `data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl`

Optional:

* `--targets-path` for evaluation

The query text must come from `DocumentRecord.retrieval_text`.

If `retrieval_text` is empty, fallback to `normalized_text`.

## Output Layout

Write under the gold layer:

* `data/gold/<dataset_name>/bm_2_tfidf_bm25/<split>.tfidf.predictions.jsonl`
* `data/gold/<dataset_name>/bm_2_tfidf_bm25/<split>.bm25.predictions.jsonl`
* `data/gold/<dataset_name>/bm_2_tfidf_bm25/<split>.tfidf.metrics.json` only when targets are provided
* `data/gold/<dataset_name>/bm_2_tfidf_bm25/<split>.bm25.metrics.json` only when targets are provided
* `data/gold/<dataset_name>/bm_2_tfidf_bm25/manifest.json`
* `data/gold/<dataset_name>/bm_2_tfidf_bm25/errors.jsonl` only when record-level failures occur

## Index Construction Policy

Build one document per ICD code using deterministic ontology text composition.

Recommended index text per code:

* `code_3char`
* `title_vi`
* `title_en` if present
* unique aliases
* `search_text`

Use a single joined string per code.

Keep one row per unique `code_3char`.

## Retrieval Policy

TF-IDF variant:

* use document `retrieval_text` as query
* compute cosine similarity against ontology index
* normalize similarities to `[0.0, 1.0]`

BM25 variant:

* tokenize query and index text deterministically
* compute BM25 score per code
* normalize scores to `[0.0, 1.0]`

Prediction logic:

* rank codes by score
* emit top-k codes
* populate `scores` with all non-zero code scores or the top-N score map
* use deterministic ordering for tie cases

Recommended defaults:

* `model_name = "bm2_tfidf"` for TF-IDF
* `model_name = "bm2_bm25"` for BM25
* `label_granularity = "code_3char"`
* `top_k = 5`
* `top_n_scores = 100`

## Library Choice

Preferred:

* `scikit-learn` for TF-IDF
* a small local BM25 helper in code, or `rank_bm25` if already available in the environment

Do not require external services.

## Evaluation Policy

If `--targets-path` is provided:

* read targets as `SilverRecord`
* compute metrics through TE-2
* write one metrics JSON per variant

If targets are missing:

* skip evaluation
* still write both prediction files

## Code Location

Create:

* `src/mnc/baselines/tfidf_bm25.py`
* `src/mnc/baselines/_bm25.py`
* `src/mnc/baselines/_vectorizer.py`

Update if needed:

* `src/mnc/baselines/__init__.py`

Tests:

* `tests/baselines/test_tfidf_bm25.py`

## Suggested Public API

```python
def run_tfidf_bm25_baselines(
    dataset_name: str,
    split: str,
    silver_dir: str = "data/silver",
    gold_dir: str = "data/gold",
    ontology_path: str = "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
    targets_path: str | None = None,
    top_k: int = 5,
) -> dict[str, list[PredictionRecord]]:
    """Run TF-IDF and BM25 ICD baselines for one dataset split."""
```

## CLI Requirements

```bash
uv run -m mnc.baselines.tfidf_bm25 --dataset vietmed-sum --split train
uv run -m mnc.baselines.tfidf_bm25 --dataset vihealthqa --split test
```

CLI arguments:

* `--dataset`
* `--split`
* `--silver-dir`
* `--gold-dir`
* `--ontology-path`
* `--targets-path`
* `--top-k`

## Error Handling

* Fail fast on missing required inputs.
* Continue on per-document failures and log them.
* Validate every emitted `PredictionRecord`.
* Keep index creation deterministic.

## Testing

Use `pytest`.

Minimum tests:

* TF-IDF predicts the correct code on a tiny hand-crafted example
* BM25 predicts the correct code on a tiny hand-crafted example
* `retrieval_text` fallback to `normalized_text` works
* output validates against `PredictionRecord`
* metrics files are written when targets are provided
* predictions are written when targets are absent
* deterministic ordering holds across repeated runs
* manifest generation works

## Acceptance Criteria

BM-2 is complete when:

* both TF-IDF and BM25 prediction files are written under `data/gold/<dataset_name>/bm_2_tfidf_bm25/`
* outputs validate against `PredictionRecord`
* bilingual ontology fields from ON-2 are used for indexing
* TE-2 metrics are used when targets are provided
* pytest coverage exists and passes

***

## Shared Implementation Notes

### File Placement

Create or modify only these areas:

* `src/mnc/datasets/candidate_generation.py`
* `src/mnc/datasets/_candidate_rank.py`
* `src/mnc/datasets/_lexical_index.py`
* `src/mnc/baselines/rule_based.py`
* `src/mnc/baselines/_rule_scoring.py`
* `src/mnc/baselines/tfidf_bm25.py`
* `src/mnc/baselines/_bm25.py`
* `src/mnc/baselines/_vectorizer.py`
* `tests/datasets/test_candidate_generation.py`
* `tests/baselines/test_rule_based.py`
* `tests/baselines/test_tfidf_bm25.py`

Do not refactor unrelated modules.

### Determinism Rules

* Do not use random seeds in ranking logic.
* Sort outputs deterministically.
* Break score ties by `code_3char` ascending.
* Keep manifest counts stable for the same input.

### Worktree Notes

Allowed workflow:

* copy required `data/silver/...` and `data/gold/...` inputs from `main` into the current worktree
* generate new artifacts inside the worktree only
* keep canonical storage layout unchanged
* do not read or write files in another worktree

### Minimal Definition of Done

Done means:

* code exists in the specified locations
* outputs are written only to the specified data paths
* all persisted rows validate against existing schemas
* required pytest tests exist and pass locally
