# DC-3: Candidate ICD generation (PDF-first ontology)

## Objective

Generate ICD-10 3-character candidate links for each document using the official PDF-derived ontology as the first source of truth.

The output must support downstream weak supervision and baseline models.

## Scope

In scope:

* Read silver documents and canonical mentions.
* Read ON-2 ontology.
* Optionally read ON-2b alias records if present.
* Generate mention-level candidates from mention text.
* Generate document-level candidates from document retrieval text.
* Merge, rank, deduplicate, and persist `CandidateLink` records.

Out of scope:

* Weak-label aggregation.
* Final prediction selection.
* Training features beyond lexical candidate generation.
* New persisted schema types.

## Required Data Structure

Use only existing schema:

* `src/mnc/schemas/candidate.py::CandidateLink`

Use `mention_id=None` for document-level retrieval candidates.

Method mapping must stay within the existing enum:

* `exact`
* `normalized`
* `fuzzy`
* `tfidf`
* `bm25`

Channel semantics:

* `exact`: exact title or alias match
* `normalized`: normalized alias or normalized phrase match
* `fuzzy`: fuzzy lexical match over ontology text
* `tfidf`: document-level TF-IDF retrieval
* `bm25`: document-level BM25 retrieval

## Input Contract

For each split:

* Read `DocumentRecord` from `data/silver/<dataset_name>/documents/<split>.jsonl`
* Read `MentionRecord` from `data/silver/<dataset_name>/canonical_mentions/<split>.jsonl`
* If canonical mentions are missing, fail loudly for DC-3
* Read `OntologyCode` from `data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl`

Optional:

* Read alias records from ON-2b if present
* Read rule records from ON-3 if present only for exclusion-style pruning

## Output Layout

Write only under the silver layer:

* `data/silver/<dataset_name>/candidate_links/<split>.jsonl`
* `data/silver/<dataset_name>/candidate_links/manifest.json`
* `data/silver/<dataset_name>/candidate_links/errors.jsonl` only when record-level failures occur

Do not overwrite upstream artifacts.

## Candidate Generation Policy

Use a PDF-first retrieval order.

Priority order:

1. Exact title match against `OntologyCode.title_vi`, `title_en`, and `aliases`
2. Normalized exact match against normalized mention text
3. Optional exact match against ON-2b alias dictionary if available
4. Fuzzy lexical match against ontology titles and aliases
5. TF-IDF retrieval over ontology `search_text`
6. BM25 retrieval over ontology `search_text`

If ON-3 exclusion rules are available:

* prune candidates whose matched evidence directly conflicts with explicit exclusion notes
* pruning only removes candidates
* pruning must never invent new candidates

If ON-3 is unavailable:

* skip pruning

## Ranking and Merge Rules

Emit candidates per document only.

Implementation rule:

* build mention-level candidates first
* build document-level retrieval candidates second
* merge by `(doc_id, code_3char)`
* keep the best score per method
* keep at most one output row per `(doc_id, mention_id, code_3char, method)`

Scoring policy:

* scores must be floats in `[0.0, 1.0]`
* exact matches should score higher than normalized matches
* normalized matches should score higher than fuzzy matches
* fuzzy matches should score higher than retrieval matches when the lexical evidence is stronger
* TF-IDF and BM25 scores must be normalized before writing

Recommended default top-k:

* mention-level: 10 per mention per method
* document-level retrieval: 20 per document per method
* final merged unique codes per document: 50

## Offset Policy

For mention-derived candidates:

* preserve `char_start` and `char_end` from the source mention

For document-level retrieval candidates:

* set `mention_id=None`
* set `char_start=None`
* set `char_end=None`

## Code Location

Create:

* `src/mnc/datasets/candidate_generation.py`
* `src/mnc/datasets/_candidate_rank.py`
* `src/mnc/datasets/_lexical_index.py`

Update if needed:

* `src/mnc/datasets/__init__.py`

Tests:

* `tests/datasets/test_candidate_generation.py`

Do not create a new top-level package.

## Suggested Public API

```python
def generate_icd_candidates(
    dataset_name: str,
    split: str,
    silver_dir: str = "data/silver",
    ontology_path: str = "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
    alias_path: str | None = "data/silver/icd10_official_pdf/alias_dictionary/alias_records.jsonl",
    rules_path: str | None = "data/silver/icd10_official_pdf/coding_rules/rule_records.jsonl",
) -> list"""Generate PDF-first ICD candidate links for one dataset split."""
```

## CLI Requirements

```bash
uv run -m mnc.datasets.candidate_generation --dataset vietmed-sum
uv run -m mnc.datasets.candidate_generation --dataset vihealthqa
uv run -m mnc.datasets.candidate_generation --all
```

CLI arguments:

* `--dataset`
* `--all`
* `--split`
* `--silver-dir`
* `--ontology-path`
* `--alias-path`
* `--rules-path`

## Error Handling

* Fail fast on missing required inputs.
* Continue on record-level validation failures.
* Write `errors.jsonl` only when at least one record fails.
* Validate every emitted `CandidateLink`.
* Do not silently coerce invalid offsets.

## Testing

Use `pytest`.

Minimum tests:

* exact title match produces `CandidateLink`
* normalized alias match produces `CandidateLink`
* TF-IDF retrieval produces document-level candidates with `mention_id=None`
* BM25 retrieval produces document-level candidates with `mention_id=None`
* duplicate candidate rows are merged deterministically
* candidate offsets are preserved for mention-derived matches
* missing canonical mention input fails loudly
* manifest generation works
* output validates against `CandidateLink`

## Acceptance Criteria

DC-3 is complete when:

* candidate links are written under `data/silver/<dataset_name>/candidate_links/`
* all output rows validate against `CandidateLink`
* exact, normalized, fuzzy, TF-IDF, and BM25 channels are supported
* mention offsets are preserved when available
* output ordering is deterministic
* required pytest coverage exists and passes

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
