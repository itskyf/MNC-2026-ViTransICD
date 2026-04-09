# ON-2b: Build Alias Dictionary from PDF Synonyms and Inclusion Text

## Objective

Build a silver-stage alias dictionary for ICD-10 3-character codes from the official PDF-derived artifacts.

Use only official PDF-derived text already available in the repository. Extract aliases from canonical titles, bilingual title variants, parenthetical descriptors, inclusion text, and NOS/KXĐK-like forms when explicitly present in the PDF-derived text.

The output must support downstream exact matching and lightweight fuzzy matching for mention normalization and candidate generation.

## Dependencies

This task depends on:

* ON-2 output:
  * `data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl`
* ON-1 output:
  * `data/bronze/icd10_official_pdf/primary/document_records.jsonl`

Do not use external lexicons, web sources, or manually curated synonym lists.

## Scope

Implement only alias extraction and normalization for 3-character ICD codes.

Persist one row per alias-code pair.

Keep provenance to the originating PDF page and source text span at line level when possible.

Write outputs only under:

* `data/silver/icd10_official_pdf/alias_dictionary/`

## Out of Scope

Do not modify ON-2 canonical ontology records in place.

Do not emit new ontology hierarchy artifacts.

Do not create any storage layer outside `data/{bronze,silver,gold}/...`.

Do not use LLM-based extraction.

Do not inject synonyms not directly supported by the PDF-derived text.

## Required Data Structure

Create a new schema file:

* `src/mnc/schemas/alias.py`

Define one persisted artifact type:

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict

class AliasRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alias_id: str
    code_3char: str
    alias: str
    alias_norm: str
    alias_type: Literal[
        "title_vi",
        "title_en",
        "parenthetical",
        "inclusion",
        "nos_form",
        "bilingual_variant",
    ]
    language: Literal["vi", "en", "mixed", "unknown"]
    match_level: Literal["exact", "normalized", "fuzzy_seed"] = "exact"
    source_page: int | None = None
    source_line: str | None = None
    created_at: datetime
```

## Input Contract

Read:

* `OntologyCode` rows from `data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl`
* `DocumentRecord` rows from `data/bronze/icd10_official_pdf/primary/document_records.jsonl`

Assume:

* `OntologyCode.code_3char` is canonical and unique
* `DocumentRecord.payload["page_no"]` exists for bronze PDF pages
* PDF page order is preserved

Fail loudly if either input is missing, invalid, or empty.

## Output Contract

Write primary output to:

* `data/silver/icd10_official_pdf/alias_dictionary/alias_records.jsonl`

Write convenience export to:

* `data/silver/icd10_official_pdf/alias_dictionary/alias_records.csv`

Each JSONL row must validate against `AliasRecord`.

CSV must contain the same fields except `created_at` may remain ISO-8601 string.

## Alias Sources

Allow only these alias sources:

* `OntologyCode.title_vi`
* `OntologyCode.title_en`
* Parenthetical descriptors attached to the same 3-character entry line or immediate title block in the bronze PDF text
* Inclusion text explicitly attached to the same 3-character entry in the bronze PDF text
* Explicit NOS/KXĐK forms present in Vietnamese or English source text
* Deterministic bilingual variants already present in the PDF-derived text

Do not use 4-character descendant labels as aliases unless they appear as inclusion-style descriptors directly under the same 3-character code and are clearly usable as lexical variants.

## Extraction Rules

Use deterministic parsing only.

Resolve candidate alias text by scanning bronze pages around each canonical code occurrence.

Use ON-2 canonical code inventory as the anchor set.

Use explicit line-prefix heuristics for note blocks, including:

* `bao gồm`
* `loại trừ`
* `sử dụng mã`
* `includes`
* `excludes`
* `use additional code`

Treat only `includes` or `bao gồm` style text as alias candidates for ON-2b.

Do not extract aliases from exclusion text.

Treat parenthetical descriptors in canonical title lines as aliases when the descriptor is lexical, not procedural.

Treat NOS/KXĐK forms as aliases only if the form appears in the same code block.

## Normalization Rules

Build `alias_norm` deterministically:

* lowercase
* trim whitespace
* collapse repeated spaces
* preserve Vietnamese diacritics
* remove trailing punctuation
* do not remove medically meaningful parentheses content before alias extraction
* do not include the ICD code itself
* do not keep empty strings

Deduplicate by `(code_3char, alias_norm)`.

If the same normalized alias appears for multiple source lines under the same code, keep one record and prefer the highest-priority `alias_type` in this order:

* `title_vi`
* `title_en`
* `parenthetical`
* `nos_form`
* `inclusion`
* `bilingual_variant`

## ID Rules

Set:

* `alias_id = "{code_3char}:{alias_norm}"`

This must be deterministic across repeated runs on the same input.

## Storage Layout

Use only:

* `data/silver/icd10_official_pdf/alias_dictionary/alias_records.jsonl`
* `data/silver/icd10_official_pdf/alias_dictionary/alias_records.csv`

Do not create extra data layers or sidecar artifacts outside this scope.

## Suggested Module Structure

Implement in:

* `src/mnc/datasets/alias_dictionary.py`
* `src/mnc/schemas/alias.py`
* `tests/datasets/test_alias_dictionary.py`

Keep helper functions small and task-focused.

Recommended functions:

* bronze reader
* ontology reader
* code-block locator
* parenthetical extractor
* inclusion-text extractor
* NOS/KXĐK detector
* alias normalizer
* JSONL writer
* CSV writer
* CLI entrypoint

## Suggested Public API

```python
def build_icd10_alias_dictionary(
    ontology_path: str = "data/silver/icd10_official_pdf/normalized_ontology/ontology_codes.jsonl",
    bronze_path: str = "data/bronze/icd10_official_pdf/primary/document_records.jsonl",
    output_dir: str = "data/silver/icd10_official_pdf/alias_dictionary",
) -> list"""Build a deterministic alias dictionary from official PDF-derived text."""
```

## Validation

Validate every output row with `AliasRecord`.

Minimum checks:

* at least one alias record is produced
* all `alias_id` values are unique
* all `code_3char` values exist in ON-2 ontology input
* `alias` is non-empty
* `alias_norm` is non-empty
* no record uses the code string itself as alias
* no duplicate `(code_3char, alias_norm)` pairs exist
* output ordering is deterministic, sorted by `code_3char`, then `alias_norm`

## Logging

Log at least:

* ontology input path
* bronze input path
* number of ontology codes read
* number of bronze pages read
* number of alias records written
* number of aliases by `alias_type`
* output paths

Do not persist custom log artifacts.

## Testing

Use `pytest`.

Keep tests simple.

Required unit tests:

* schema validation test for one `AliasRecord`
* `alias_id` formatting test
* alias normalization test
* duplicate alias deduplication test
* parenthetical extraction smoke test
* inclusion-text extraction smoke test
* NOS/KXĐK extraction test
* JSONL writer test
* CSV writer test
* deterministic ordering test on repeated runs

Do not require network access.

Use small mocked `OntologyCode` and `DocumentRecord` inputs.

## Acceptance Criteria

The task is complete when:

* the extractor reads only ON-1 and ON-2 artifacts
* output is written only under `data/silver/icd10_official_pdf/alias_dictionary/`
* JSONL rows validate against `AliasRecord`
* one row is emitted per unique `(code_3char, alias_norm)`
* aliases come only from official PDF-derived text
* inclusion text contributes aliases only when explicitly inclusion-like
* exclusion text never contributes aliases
* `pytest` unit tests exist and pass
