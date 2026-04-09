# BM-1: Rule-based baseline using official PDF notes

## Objective

Implement a deterministic rule-based baseline that predicts ICD-10 3-character codes from document text and extracted mentions using official PDF-derived ontology evidence.

This baseline must be simple, auditable, and note-aware.

## Scope

In scope:

* Read silver documents and mentions from DC-1
* Read ON-2 ontology
* Optionally read ON-2b aliases
* Optionally read ON-3 rules for exclusion-style pruning and note-aware backoff
* Produce ranked predictions as `PredictionRecord`
* Optionally compute metrics when target labels are provided

Out of scope:

* Model training
* Neural scoring
* Dense retrieval
* New persisted schema types

## Required Data Structures

Use only existing schemas:

* `src/mnc/schemas/prediction.py::PredictionRecord`
* `src/mnc/schemas/silver.py::SilverRecord` for optional evaluation input
* `src/mnc/schemas/ontology.py::OntologyCode`

Do not introduce a custom baseline prediction schema.

## Input Contract

Required:

* `data/silver/<dataset_name>/documents/<split>.jsonl`
* `data/silver/<dataset_name>/mentions/<split>.jsonl`
* `data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl`

Optional:

* `data/silver/icd10_official_pdf/alias_dictionary/alias_records.jsonl`
* `data/silver/icd10_official_pdf/coding_rules/rule_records.jsonl`
* `--targets-path` for evaluation

If optional artifacts are missing, the baseline must still run with titles and aliases available in ON-2.

## Output Layout

Write under the gold layer:

* `data/gold/<dataset_name>/bm_1_rule_based/<split>.predictions.jsonl`
* `data/gold/<dataset_name>/bm_1_rule_based/<split>.metrics.json` only when targets are provided
* `data/gold/<dataset_name>/bm_1_rule_based/manifest.json`
* `data/gold/<dataset_name>/bm_1_rule_based/errors.jsonl` only when record-level failures occur

## Rule Engine Policy

Use deterministic matching only.

Evidence sources in descending priority:

1. exact mention match to official title
2. exact mention match to official alias
3. normalized mention match
4. document-level lexical support from document retrieval text
5. optional inclusion-style support from ON-2b alias text if present
6. optional exclusion-style pruning from ON-3 if present

Prediction logic:

* aggregate evidence per code at document level
* rank codes by total rule score
* emit top-k ranked codes
* always predict at 3-character granularity
* do not emit chapter predictions in BM-1

Recommended default rule weights:

* exact title match: `1.00`
* exact alias match: `0.95`
* normalized match: `0.85`
* document lexical support: `0.60`
* inclusion-style support: `0.75`
* exclusion pruning: remove candidate or apply strong negative filter
* repeated independent evidence: add small deterministic bonus such as `+0.05` once per additional evidence source, capped

If rule notes are unavailable:

* skip note-aware pruning
* do not fail

## Score Output Policy

Populate `PredictionRecord.scores` with document-level code scores after aggregation.

Populate `PredictionRecord.predicted_codes` with the top-k codes sorted by descending score.

Recommended defaults:

* `model_name = "bm_1_rule_based"`
* `label_granularity = "code_3char"`
* `k = 5`

## Evaluation Policy

If `--targets-path` is provided:

* load targets as `SilverRecord`
* compute metrics via TE-2 evaluator
* write metrics JSON next to predictions

If targets are missing:

* skip evaluation
* still write predictions

## Code Location

Create:

* `src/mnc/baselines/rule_based.py`
* `src/mnc/baselines/_rule_scoring.py`

Update if needed:

* `src/mnc/baselines/__init__.py`

Tests:

* `tests/baselines/test_rule_based.py`

Reuse TE-2 evaluator instead of re-implementing metrics.

## Suggested Public API

```python
def run_rule_based_baseline(
    dataset_name: str,
    split: str,
    silver_dir: str = "data/silver",
    gold_dir: str = "data/gold",
    ontology_path: str = "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
    alias_path: str | None = "data/silver/icd10_official_pdf/alias_dictionary/alias_records.jsonl",
    rules_path: str | None = "data/silver/icd10_official_pdf/coding_rules/rule_records.jsonl",
    targets_path: str | None = None,
    top_k: int = 5,
) -> list"""Run the BM-1 rule-based ICD baseline for one dataset split."""
```

## CLI Requirements

```bash
uv run -m mnc.baselines.rule_based --dataset vietmed-sum --split train
uv run -m mnc.baselines.rule_based --dataset vihealthqa --split test
```

CLI arguments:

* `--dataset`
* `--split`
* `--silver-dir`
* `--gold-dir`
* `--ontology-path`
* `--alias-path`
* `--rules-path`
* `--targets-path`
* `--top-k`

## Error Handling

* Fail fast on missing required inputs.
* Continue on per-document failures and log them.
* Validate every emitted `PredictionRecord`.
* Raise clear errors for invalid target files in evaluation mode.

## Testing

Use `pytest`.

Minimum tests:

* exact mention match ranks the correct code first
* alias match ranks the correct code first
* repeated evidence increases score deterministically
* optional exclusion pruning removes a conflicting code when ON-3 rules are present
* output validates against `PredictionRecord`
* metrics file is written when targets are provided
* predictions are still written when targets are absent
* manifest generation works

## Acceptance Criteria

BM-1 is complete when:

* predictions are written under `data/gold/<dataset_name>/bm_1_rule_based/`
* output rows validate against `PredictionRecord`
* the baseline runs with required inputs only
* optional note-aware pruning is supported when ON-3 exists
* metrics are computed through TE-2 when targets are provided
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
