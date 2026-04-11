# Technical Specification: DC-1 and DC-2

## Title

DC-1: Normalize text, segment, and extract mentions  
DC-2: Abbreviation normalization

## Purpose

Implement the first semantic transformation stage on top of bronze documents.

DC-1 must:

* read bronze document records
* normalize text for NLP and retrieval
* segment documents into sentence spans
* extract mention candidates with raw-text offsets

DC-2 must:

* normalize abbreviation mentions into canonical surface forms
* preserve provenance from DC-1
* produce canonical mentions for downstream candidate generation

## Dependencies

* DE-3 output is required as input.
* DC-2 depends on DC-1 output.
* Do not depend on ON-2b. Ontology-driven normalization is out of scope for DC-2 and belongs to DC-2b.

## Non-goals

* No ICD candidate generation.
* No ontology alias matching.
* No weak supervision.
* No model training features beyond `normalized_text`, `retrieval_text`, sentence spans, and mention records.
* No new data layer outside `data/{bronze,silver,gold}/...`.
* No network calls in tests.

## Input

For each dataset:

* `data/bronze/<dataset_name>/documents/<split>.jsonl`

Input records must validate against `src/mnc/schemas/document.py::DocumentRecord`.

## Output Layout

Write only under the silver layer.

For DC-1:

* `data/silver/<dataset_name>/documents/<split>.jsonl`
* `data/silver/<dataset_name>/sentence_spans/<split>.jsonl`
* `data/silver/<dataset_name>/mentions/<split>.jsonl`
* `data/silver/<dataset_name>/documents/manifest.json`
* `data/silver/<dataset_name>/sentence_spans/manifest.json`
* `data/silver/<dataset_name>/mentions/manifest.json`
* `data/silver/<dataset_name>/mentions/errors.jsonl` only if record-level failures occur

For DC-2:

* `data/silver/<dataset_name>/canonical_mentions/<split>.jsonl`
* `data/silver/<dataset_name>/canonical_mentions/manifest.json`
* `data/silver/<dataset_name>/canonical_mentions/errors.jsonl` only if record-level failures occur

Do not overwrite bronze artifacts.

## Data Contracts

### 1. Silver DocumentRecord

Reuse `src/mnc/schemas/document.py::DocumentRecord`.

DC-1 must populate:

* `normalized_text`
* `retrieval_text`
* `sentences`

DC-1 must preserve:

* `doc_id`
* `source`
* `language`
* `raw_text`
* `source_record_id`
* `split`
* `payload`

Field semantics:

* `normalized_text`: normalized text for NLP and alignment
* `retrieval_text`: normalized text optimized for lexical retrieval
* `sentences`: ordered sentence texts in normalized-text order

Normalization rules:

* preserve Vietnamese diacritics
* normalize Unicode to NFKC
* normalize line endings to `\n`
* collapse repeated spaces and tabs
* trim leading and trailing whitespace
* normalize common punctuation spacing
* remove zero-width and other invisible separator characters
* do not remove medically meaningful punctuation inside abbreviations or lab values
* do not strip digits
* do not transliterate Vietnamese characters

Retrieval text rules:

* start from `normalized_text`
* lowercase
* collapse whitespace
* replace most punctuation with spaces
* keep alphanumeric content
* keep Vietnamese diacritics
* do not stem
* do not remove stopwords in this task

### 2. SentenceSpanRecord

Add a new schema at `src/mnc/schemas/sentence.py`.

```python
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class SentenceSpanRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sentence_id: str
    doc_id: str
    sentence_index: int
    text: str
    char_start: int
    char_end: int
    created_at: datetime
````

Field semantics:

* `text` is the sentence text from `normalized_text`
* `char_start` and `char_end` are offsets into `normalized_text`
* `char_end` is exclusive
* `sentence_id` must be deterministic: `"{doc_id}:s:{sentence_index}"`

### 3. MentionRecord for DC-1

Reuse `src/mnc/schemas/mention.py::MentionRecord`.

Field semantics for DC-1:

* `text`: raw surface form from `raw_text`
* `normalized_text`: mention-level normalized form derived from the matched surface form
* `mention_type`: one of existing enum values
* `char_start` and `char_end`: offsets into `raw_text`
* `char_end` is exclusive
* `mention_id` must be deterministic: `"{doc_id}:m:{char_start}:{char_end}"`

Confidence policy:

* set `confidence` only if the extractor exposes a stable score
* otherwise set `None`

### 4. MentionRecord for DC-2

Reuse the same `MentionRecord` schema and write canonicalized results into `canonical_mentions`.

Field semantics for DC-2:

* preserve `mention_id`
* preserve `doc_id`
* preserve `text`
* preserve `char_start`
* preserve `char_end`
* update `normalized_text` to the canonical expansion if resolved
* keep `normalized_text` from DC-1 if unresolved
* set `mention_type="abbreviation"` only when the extracted surface is an abbreviation mention; otherwise preserve the original type

This avoids introducing an extra downstream schema.

## Code Location

Implement in the dataset pipeline package.

Create:

* `src/mnc/datasets/normalize.py`
* `src/mnc/datasets/abbrev.py`
* `src/mnc/datasets/_text.py`
* `src/mnc/datasets/_mentions.py`
* `src/mnc/schemas/sentence.py`

Update:

* `src/mnc/schemas/__init__.py`
* `src/mnc/datasets/__init__.py` if needed

Tests:

* `tests/datasets/test_normalize.py`
* `tests/datasets/test_abbrev.py`

If a small local resource file is needed for seed abbreviation mappings, store it in:

* `src/mnc/resources/abbreviations/vi_medical_abbrev_seed.json`

Do not create a new top-level package.

## Library Choice

Prefer established libraries over custom implementations.

Recommended dependencies:

* Python `re` for normalization rules and abbreviation patterns
* [underthesea](https://github.com/undertheseanlp/underthesea) for Vietnamese sentence segmentation
* [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) for lightweight string similarity only if needed for abbreviation resolution tie-breaks
* `pydantic` for schema validation
* `pytest` for unit tests

Implementation rule:

* keep extraction mostly deterministic and rule-based
* avoid training a model in DC-1 or DC-2
* avoid depending on external APIs

## DC-1 Functional Requirements

### A. Read and validate bronze documents

For each split:

* read `data/bronze/<dataset_name>/documents/<split>.jsonl`
* validate each line as `DocumentRecord`
* skip invalid records
* log record-level failures to `errors.jsonl` only for the failing scope
* continue processing after record-level failures

### B. Normalize document text

Implement a pure function in `src/mnc/datasets/_text.py`:

```python
def normalize_document_text(text: str) -> str:
    """Return normalized text for NLP."""
```

Implement a pure function in `src/mnc/datasets/_text.py`:

```python
def build_retrieval_text(normalized_text: str) -> str:
    """Return retrieval-oriented text."""
```

Behavior:

* input is `raw_text`
* output must be deterministic
* empty `raw_text` is invalid for DC-1 and should be counted as failure

### C. Segment sentences

Implement sentence segmentation against `normalized_text`.

Requirements:

* use a library-first approach
* return ordered sentence spans with exact offsets in `normalized_text`
* populate `DocumentRecord.sentences` from the sentence span texts
* if sentence segmentation fails, fallback to one-sentence-per-document using the full `normalized_text`
* never emit overlapping sentence spans
* never emit out-of-range spans

Implement:

```python
def segment_sentences(text: str) -> list[SentenceSpanRecord]:
    """Segment normalized text into sentence spans."""
```

### D. Extract mention candidates

Implement rule-based mention extraction from `raw_text`.

Extraction targets:

* disease-like phrases
* symptom-like phrases
* diagnosis-like phrases
* procedure-like phrases
* abbreviation-like phrases
* fallback `other` only when a detected candidate does not fit the main labels

Minimum extraction strategy:

* detect spans triggered by common clinical cue patterns
* detect noun-phrase-like spans adjacent to cue patterns
* detect uppercase and mixed-case medical abbreviations
* deduplicate identical spans within a document
* keep the longest non-overlapping span when two candidates share the same start
* never emit zero-length spans

Boundary trimming rules:

* collapse consecutive duplicate words (e.g. "bệnh bệnh" → "bệnh")
* trim trailing Vietnamese function words (up to 3 rounds)
* strip trailing single-character noise from speech transcripts
* maximum capture length: 25 characters for disease cues, 20 for symptom/diagnosis/procedure cues

Mention normalization for DC-1:

* normalize the extracted surface form with the same mention-level text normalization rules as document normalization
* do not expand abbreviations in DC-1

Implement:

```python
def extract_mentions(doc: DocumentRecord) -> list[MentionRecord]:
    """Extract mention candidates with raw-text offsets."""
```

### E. Write DC-1 outputs

For each split:

* write enriched silver `DocumentRecord`
* write `SentenceSpanRecord`
* write `MentionRecord`
* write one manifest per scope

Manifest schema:

* reuse `src/mnc/schemas/manifest.py::BronzeManifest` for now
* dataset field is `<dataset_name>`
* input and output split fields must reflect processed splits
* record counts must be per split
* failed counts must be per split

## DC-2 Functional Requirements

### A. Read and validate DC-1 mentions

For each split:

* read `data/silver/<dataset_name>/mentions/<split>.jsonl`
* validate each line as `MentionRecord`
* process only mention records from DC-1
* skip invalid rows and log them

### B. Abbreviation normalization logic

Implement deterministic abbreviation expansion in `src/mnc/datasets/abbrev.py`.

Priority order:

* explicit local seed dictionary match
* in-document definitional pattern match
* simple normalized-form match
* optional fuzzy tie-break only within already shortlisted candidates
* fallback to original DC-1 `normalized_text`

Supported definitional patterns:

* `full form (ABBR)`
* `ABBR (full form)`
* repeated local reuse of an abbreviation after first definition in the same document

Resolution rules:

* operate within the same document only
* prefer the nearest earlier full-form definition
* keep output deterministic
* do not invent expansions without evidence
* do not use ontology aliases in DC-2
* do not expand common ambiguous abbreviations when multiple expansions are equally plausible; keep the original normalized form

Implement:

```python
def normalize_abbreviations(
    mentions: list[MentionRecord],
    raw_text: str,
) -> list[MentionRecord]:
    """Return canonical mentions with abbreviation expansion when resolved."""
```

### C. Canonical mention policy

When abbreviation resolution succeeds:

* write the expanded canonical phrase to `normalized_text`

When abbreviation resolution fails:

* keep DC-1 `normalized_text`

Preserve:

* `mention_id`
* `doc_id`
* `text`
* `char_start`
* `char_end`
* `confidence`

### D. Write DC-2 outputs

For each split:

* write canonical mention records to `data/silver/<dataset_name>/canonical_mentions/<split>.jsonl`
* write manifest
* keep DC-1 outputs unchanged

## Determinism Rules

The pipeline must be deterministic across repeated runs over the same input.

Use deterministic IDs:

* `sentence_id = "{doc_id}:s:{sentence_index}"`
* `mention_id = "{doc_id}:m:{char_start}:{char_end}"`

Do not use random seeds for extraction logic.

## CLI Requirements

Add CLI entry points consistent with existing dataset modules.

DC-1:

```bash
uv run -m mnc.datasets.normalize --dataset vietmed-sum
uv run -m mnc.datasets.normalize --dataset vihealthqa
uv run -m mnc.datasets.normalize --all
```

DC-2:

```bash
uv run -m mnc.datasets.abbrev --dataset vietmed-sum
uv run -m mnc.datasets.abbrev --dataset vihealthqa
uv run -m mnc.datasets.abbrev --all
```

CLI arguments:

* `--dataset`
* `--all`
* `--bronze-dir` for DC-1 with default `data/bronze`
* `--silver-dir` for DC-1 and DC-2 with default `data/silver`

## Error Handling

* Fail fast on missing input directories.
* Continue on record-level validation or extraction failures.
* Write `errors.jsonl` only when at least one record fails.
* Do not silently coerce invalid offsets.
* Validate all emitted records with Pydantic before writing.

## Testing

Unit tests are mandatory and must use `pytest`.

Minimum DC-1 tests:

* document text normalization happy path
* retrieval text generation happy path
* sentence segmentation returns valid ordered spans
* sentence fallback path when segmenter fails
* mention extraction happy path for disease-like text
* abbreviation-looking surface is extracted as a mention candidate
* deterministic `mention_id`
* deterministic `sentence_id`
* manifest generation
* invalid input record is skipped and logged

Minimum DC-2 tests:

* seed dictionary expansion happy path
* `full form (ABBR)` expansion
* `ABBR (full form)` expansion
* unresolved abbreviation keeps original normalized form
* ambiguous abbreviation remains unresolved
* only same-document evidence is used
* canonical mention file validates against `MentionRecord`
* manifest generation

Test constraints:

* local only
* no network
* small fixtures only
* no dependence on bronze files from remote sources

## Worktree Notes

The coding agent will work in a Git worktree.

Allowed workflow:

* copy required `data/bronze/...` directories from `main` into the worktree
* generate new `data/silver/...` artifacts inside the worktree
* keep branch-local artifacts isolated to the worktree
* do not edit files in another worktree

## Acceptance Criteria

DC-1 is done when:

* silver document files exist for each processed split
* each silver document has `normalized_text`, `retrieval_text`, and `sentences`
* sentence span files exist and offsets are valid against `normalized_text`
* mention files exist and offsets are valid against `raw_text`
* outputs validate against schemas
* pytest coverage for the required DC-1 cases exists and passes

DC-2 is done when:

* canonical mention files exist for each processed split
* resolved abbreviations update `normalized_text`
* unresolved abbreviations preserve DC-1 normalized form
* outputs validate against `MentionRecord`
* pytest coverage for the required DC-2 cases exists and passes

## Implementation Notes

Keep the first version simple and robust.

Recommended execution order:

* implement pure text normalization helpers
* implement sentence segmentation with offset alignment
* implement mention extraction
* implement DC-1 CLI and writers
* implement abbreviation normalization
* implement DC-2 CLI and writers
* add pytest coverage last, then run end-to-end locally
