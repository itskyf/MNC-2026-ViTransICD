# ON-3: Extract Coding Rules and Notes from PDF Intro + Per-Code Notes

## Objective

Build a silver-stage rule extractor that converts official ICD-10 intro guidance and per-code note text into deterministic, auditable rule records for downstream weak supervision, rule-based baselines, and QA.

The task must combine:

* global intro rules from ON-1b
* per-code note rules from the official PDF bronze pages
* canonical code inventory from ON-2

## Dependencies

This task depends on:

* ON-1b output:
  * `data/silver/icd10_official_pdf/intro_guidance/document_records.jsonl`
* ON-2 output:
  * `data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl`
* ON-1 output:
  * `data/bronze/icd10_official_pdf/primary/document_records.jsonl`

Do not re-download the PDF.

## Scope

Implement only deterministic extraction of global coding rules and per-code note rules relevant to 3-character coding.

Persist one row per normalized rule instance.

Write outputs only under:

* `data/silver/icd10_official_pdf/coding_rules/`

## Out of Scope

Do not implement a full inference engine.

Do not score or apply rules to documents in this task.

Do not create chapter, block, or 4-character-specific rule artifacts unless a note is clearly attached to a 3-character code and remains useful at 3-character granularity.

Do not use LLM-based parsing.

Do not create any storage layer outside `data/{bronze,silver,gold}/...`.

## Required Data Structure

Create a new schema file:

* `src/mnc/schemas/rule.py`

Define one persisted artifact type:

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict

class RuleRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    scope: Literal["global", "code"]
    code_3char: str | None = None
    topic: Literal[
        "principal_diagnosis",
        "symptom_fallback",
        "mortality_coding",
        "official_3char_policy",
        "include_note",
        "exclude_note",
        "use_additional_code",
        "code_first",
        "general_note",
    ]
    action: Literal[
        "prefer",
        "fallback",
        "restrict",
        "allow",
        "require_additional_code",
        "code_first",
        "note",
    ]
    priority: int
    source_page_start: int | None = None
    source_page_end: int | None = None
    evidence_text: str
    normalized_text: str
    created_at: datetime
```

## Input Contract

Read:

* `DocumentRecord` rows from `data/silver/icd10_official_pdf/intro_guidance/document_records.jsonl`
* `OntologyCode` rows from `data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl`
* `DocumentRecord` rows from `data/bronze/icd10_official_pdf/primary/document_records.jsonl`

Fail loudly if any input is missing, invalid, or empty.

Fail loudly if no rule records are extracted.

## Output Contract

Write primary output to:

* `data/silver/icd10_official_pdf/coding_rules/rule_records.jsonl`

Write lightweight heuristic template export to:

* `data/silver/icd10_official_pdf/coding_rules/heuristic_templates.json`

Each JSONL row must validate against `RuleRecord`.

The template JSON is a convenience artifact for downstream consumers and may be generated from validated `RuleRecord` rows.

## Rule Granularity

Use one `RuleRecord` per normalized rule statement.

Support two scopes:

* `global` for intro guidance
* `code` for per-code notes

Attach `code_3char` only for `scope="code"`.

Keep `evidence_text` source-faithful.

Keep `normalized_text` minimally normalized for downstream matching.

## Rule Sources

Extract global rules only from ON-1b topics:

* `principal_diagnosis`
* `symptom_fallback`
* `mortality_coding`
* `official_3char_policy`

Extract per-code rules only from bronze PDF pages, anchored by ON-2 canonical codes.

Allowed per-code note types:

* inclusion note
* exclusion note
* use additional code
* code first
* general procedural note

## Extraction Rules

Use deterministic regex and line-window parsing only.

Anchor per-code note extraction by locating canonical 3-character code blocks in bronze pages.

Parse note blocks by explicit note prefixes, including:

* `bao gồm`
* `loại trừ`
* `sử dụng mã`
* `mã thêm`
* `includes`
* `excludes`
* `use additional code`
* `code first`

Map note prefixes to normalized rule fields:

* inclusion-like note → `topic="include_note"`, `action="allow"`
* exclusion-like note → `topic="exclude_note"`, `action="restrict"`
* additional-code note → `topic="use_additional_code"`, `action="require_additional_code"`
* code-first note → `topic="code_first"`, `action="code_first"`
* residual note text → `topic="general_note"`, `action="note"`

Map ON-1b global topics to normalized rule fields:

* `principal_diagnosis` → `action="prefer"`
* `symptom_fallback` → `action="fallback"`
* `mortality_coding` → `action="prefer"`
* `official_3char_policy` → `action="restrict"`

Use deterministic priority values:

* 100 for `official_3char_policy`
* 90 for `principal_diagnosis`
* 80 for `mortality_coding`
* 70 for `symptom_fallback`
* 60 for `code_first`
* 50 for `use_additional_code`
* 40 for `exclude_note`
* 30 for `include_note`
* 10 for `general_note`

## Normalization Rules

Build `normalized_text` by:

* normalizing line endings
* trimming whitespace
* collapsing repeated spaces
* collapsing repeated blank lines
* preserving Vietnamese diacritics
* preserving original meaning
* not paraphrasing

Do not attempt semantic rewriting.

## ID Rules

Set `rule_id` deterministically:

* global rule: `"global:{topic}"`
* code rule: `"{code_3char}:{topic}:{hash8}"`

Use a short stable hash of `normalized_text` for `hash8`.

This avoids collisions across multiple note lines of the same type under one code.

## Heuristic Template Export

Generate `heuristic_templates.json` from validated rule rows.

Include only fields needed by downstream rule consumers:

* `rule_id`
* `scope`
* `code_3char`
* `topic`
* `action`
* `priority`
* `normalized_text`

Do not add extra inferred logic.

## Storage Layout

Use only:

* `data/silver/icd10_official_pdf/coding_rules/rule_records.jsonl`
* `data/silver/icd10_official_pdf/coding_rules/heuristic_templates.json`

Do not create extra data layers.

## Suggested Module Structure

Implement in:

* `src/mnc/datasets/coding_rules.py`
* `src/mnc/schemas/rule.py`
* `tests/datasets/test_coding_rules.py`

Keep functions small and explicit.

Recommended functions:

* intro guidance reader
* ontology reader
* bronze reader
* code-block locator
* note-block parser
* global-rule builder
* code-rule builder
* heuristic-template exporter
* JSONL writer
* CLI entrypoint

## Suggested Public API

```python
def extract_icd10_coding_rules(
    intro_path: str = "data/silver/icd10_official_pdf/intro_guidance/document_records.jsonl",
    ontology_path: str = "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
    bronze_path: str = "data/bronze/icd10_official_pdf/primary/document_records.jsonl",
    output_dir: str = "data/silver/icd10_official_pdf/coding_rules",
) -> list"""Extract deterministic global and per-code coding rules from PDF-derived artifacts."""
```

## Validation

Validate every output row with `RuleRecord`.

Minimum checks:

* at least four global rules are produced
* all global ON-1b topics are present exactly once
* all `rule_id` values are unique
* all `scope="code"` rows have non-null `code_3char`
* all `code_3char` values in code-scoped rules exist in ON-2 ontology input
* `evidence_text` is non-empty
* `normalized_text` is non-empty
* `priority` is positive
* output ordering is deterministic, sorted by `scope`, then `code_3char`, then `topic`, then `rule_id`

## Logging

Log at least:

* intro input path
* ontology input path
* bronze input path
* number of intro guidance records read
* number of ontology codes read
* number of bronze pages read
* number of global rules written
* number of code-scoped rules written
* output paths

Do not persist custom log artifacts.

## Testing

Use `pytest`.

Keep tests simple.

Required unit tests:

* schema validation test for one `RuleRecord`
* global `rule_id` formatting test
* code-scoped `rule_id` stability test
* intro guidance to global rule mapping test
* inclusion note extraction smoke test
* exclusion note extraction smoke test
* `use additional code` extraction test
* `code first` extraction test
* missing-required-global-topic failure test
* JSONL writer test
* heuristic template export test
* deterministic ordering test on repeated runs

Do not require network access.

Use small mocked `DocumentRecord` and `OntologyCode` inputs.

## Acceptance Criteria

The task is complete when:

* the extractor reads only ON-1, ON-1b, and ON-2 artifacts
* output is written only under `data/silver/icd10_official_pdf/coding_rules/`
* JSONL rows validate against `RuleRecord`
* all four required global rule topics are emitted exactly once
* per-code notes are extracted only from official PDF-derived text
* note types are mapped deterministically to normalized rule topics and actions
* heuristic template export is generated from validated rule rows
* `pytest` unit tests exist and pass

***

## Implementation Notes for the Coding Agent

## File Placement

Create or modify only these code areas:

* `src/mnc/schemas/alias.py`
* `src/mnc/schemas/rule.py`
* `src/mnc/datasets/alias_dictionary.py`
* `src/mnc/datasets/coding_rules.py`
* `tests/datasets/test_alias_dictionary.py`
* `tests/datasets/test_coding_rules.py`

Do not refactor unrelated modules.

## Data Layout Constraint

Use only:

* `data/bronze/<dataset_name>/<scope>/...`
* `data/silver/<dataset_name>/<scope>/...`
* `data/gold/<dataset_name>/<scope>/...`

Do not create `data/raw/` or any extra layer.

## Worktree Constraint

The agent will work in a Git worktree.

If needed, copy the required `data/bronze/...` and `data/silver/...` inputs from `main` into the worktree before running local tests.

Do not change the canonical storage layout because of worktree usage.

## Style Constraint

Keep the implementation deterministic, auditable, and regex-first.

Prefer explicit parsing helpers over generic abstractions.

Fail loudly on missing required inputs or malformed records.
