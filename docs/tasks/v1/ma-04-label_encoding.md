# MA-4: Label Description Encoding from ICD Ontology

## 1. Objective

Implement a label encoder that converts ICD ontology entries into dense label embeddings using the existing SEA-LION backbone from MA-1.

The component must:

* accept schema-based ontology input
* build a bilingual label description text per code
* encode each label description into token-level and pooled embeddings
* expose pooled label embeddings for MA-3 external label queries
* preserve code order for downstream joins and explanation

This task is model-component only. It is not a data-ingestion or persistence task.

## 2. Scope

In scope:

* consume `list[OntologyCode]` from `src/mnc/schemas/`
* reuse the existing backbone in `src/mnc/models/sea_lion_encoder.py`
* format bilingual label text from schema fields only
* generate runtime label embeddings
* add unit tests with `pytest`

Out of scope:

* ontology parsing from PDF
* web extraction
* ontology normalization
* training loop
* classifier logits
* loss computation
* persistence of embeddings to a new schema
* explainability logic beyond preserving audit metadata

## 3. Dependencies

Required upstream:

* MA-1: `SeaLionEncoder`

Logical downstream consumers:

* MA-3: external label embeddings for code-wise attention
* EX-4: label-side descriptor alignment and display

## 4. Schema Contract

Pipeline-facing input must use existing schemas only.

Accepted input artifact:

* `list[OntologyCode]`

Relevant schema fields:

* `code_3char`
* `title_vi`
* `title_en`
* `aliases`
* `search_text`

Pipeline-facing output:

* no persisted schema artifact
* runtime container only

Rationale:

* no label-embedding persistence schema exists in `src/mnc/schemas/`
* MA-4 is an internal model component, similar to MA-1

## 5. Runtime Interface

Implement one encoder module and one output container.

Recommended module path:

* `src/mnc/models/ontology_label_encoder.py`

Recommended public API:

* `OntologyLabelEncoder`
* `OntologyLabelEncoderOutput`

Required method:

```python
encode_labels(
    labels: list[OntologyCode],
    max_length: int | None = None,
) -> OntologyLabelEncoderOutput
```

Required output fields:

* `label_codes: list[str]`
* `label_texts: list[str]`
* `input_ids: torch.Tensor`
* `attention_mask: torch.Tensor`
* `token_embeddings: torch.Tensor`
* `label_embeddings: torch.Tensor`

Tensor shape contract:

* `input_ids`: `[num_labels, seq_len]`
* `attention_mask`: `[num_labels, seq_len]`
* `token_embeddings`: `[num_labels, seq_len, hidden_size]`
* `label_embeddings`: `[num_labels, hidden_size]`

Order contract:

* `label_codes[i]` must correspond exactly to `label_texts[i]` and `label_embeddings[i]`
* output order must match input order exactly

## 6. Backbone Reuse Requirements

Do not re-implement backbone loading.

Required behavior:

* reuse `SeaLionEncoder` from MA-1
* use its tokenizer and base encoder
* use its default pooling behavior for label embeddings

Recommended construction:

* allow injection of an existing `SeaLionEncoder`
* if not injected, instantiate `SeaLionEncoder()` with MA-1 defaults

Example:

```python
class OntologyLabelEncoder(nn.Module):
    def __init__(self, encoder: SeaLionEncoder | None = None) -> None:
        ...
```

## 7. Label Text Construction

Build one bilingual prompt per `OntologyCode`.

Use schema fields only:

* always include `code_3char`
* include `title_vi` if non-empty
* include `title_en` if non-empty
* include a compact descriptor derived from `search_text` if non-empty
* do not require `aliases` in the default prompt
* do not depend on fields that do not exist in schema

Prompt goal:

* produce stable, compact, semantically rich label text
* reflect the task note: prefer Vietnamese + English titles and short ontology descriptor text

Recommended default template:

```text
ICD-10 code: {code_3char}
Vietnamese title: {title_vi}
English title: {title_en}
Descriptor: {descriptor_text}
```

Prompt construction rules:

* normalize whitespace
* skip empty sections
* deduplicate repeated content between titles and `search_text`
* truncate descriptor text to a configurable limit
* keep prompt deterministic

Recommended default:

* `max_descriptor_chars = 256`

Reason:

* `search_text` is the only schema-safe place that may already contain normalized ontology descriptors from ON-2
* MA-4 must stay compatible with current schemas

## 8. Encoding Behavior

Required behavior:

* encode all labels in batch
* tokenize with padding and truncation
* allow caller-provided `max_length`
* if `max_length` is `None`, use tokenizer/model defaults
* return token-level label embeddings and pooled label embeddings

Implementation approach:

* build label texts in memory
* tokenize with the reused SEA-LION tokenizer
* run the reused encoder backbone
* map pooled output to `label_embeddings`

Compatibility requirement:

* `label_embeddings` must be directly usable as MA-3 external label embeddings
* shape must be `[num_labels, hidden_size]`

## 9. Data Handling Rules

Do not create a new data layer.

Allowed data layout in the repo:

* `data/bronze/<dataset_name>/<scope>/...`
* `data/silver/<dataset_name>/<scope>/...`
* `data/gold/<dataset_name>/<scope>/...`

MA-4 requirement:

* no file I/O is required for the core implementation
* operate entirely in memory on `list[OntologyCode]`
* do not introduce `data/raw/` or any new layer
* do not define a new embedding dataset format in this task

## 10. Error Handling

Required checks:

* raise `ValueError` if input list is empty
* raise `TypeError` if any item is not `OntologyCode`
* raise `ValueError` if duplicate `code_3char` values are provided
* raise `ValueError` if the constructed label text is empty or non-informative for any label
* keep exceptions explicit and short

Do not:

* silently drop labels
* reorder labels
* mutate input schemas
* auto-fill fake ontology text

## 11. Dummy Data Expectations

MA-4 must run without real ontology files.

Use small in-memory `OntologyCode` fixtures.

Minimum supported scenarios:

* one label with Vietnamese and English titles
* multiple labels in one batch
* label with empty `title_en` but valid `title_vi`
* label with useful `search_text` contributing descriptor text
* stable input-order preservation

## 12. Testing Requirements

Use `pytest`.

Keep tests simple and fast.

Recommended test path:

* `tests/models/test_ontology_label_encoder.py`

Minimum unit tests:

1. `encode_labels` returns all required fields
2. output batch size equals number of input labels
3. `label_codes` preserve input order
4. `label_embeddings.ndim == 2`
5. `token_embeddings.ndim == 3`
6. prompt formatting includes code and bilingual titles when present
7. descriptor text is included from `search_text` when present
8. empty input raises `ValueError`
9. invalid item type raises `TypeError`
10. duplicate codes raise `ValueError`

Test implementation guidance:

* use tiny fixtures
* avoid asserting exact numeric embedding values
* assert shapes, contract behavior, and prompt content only
* prefer mocking or monkeypatching the heavy encoder call if needed
* do not require real ontology files under `data/`

## 13. Implementation Notes

Recommended design:

* one small dataclass for runtime output
* one `nn.Module` wrapper around `SeaLionEncoder`
* one private helper to build label text
* one private helper to compact descriptor text

Recommended defaults:

* `max_descriptor_chars=256`
* `pooling` inherited from MA-1 encoder
* `return_dict=True` via MA-1 backbone path

Keep the code minimal:

* no registry
* no abstract factory
* no extra persistence layer
* no training logic
* no new schema definitions

## 14. Acceptance Criteria

The task is done when all conditions below are true:

1. A coding agent can instantiate `OntologyLabelEncoder` with the existing SEA-LION backbone.
2. The component accepts `list[OntologyCode]` only.
3. The component builds deterministic bilingual label text from schema fields only.
4. The component returns token-level and pooled label embeddings at runtime.
5. `label_embeddings` can be passed directly to `TransICDHead(..., label_embeddings=...)`.
6. Input order is preserved exactly in `label_codes` and embeddings.
7. No new data layer or new schema artifact is introduced.
8. Unit tests pass with `pytest`.

## 15. Deliverables

Expected deliverables:

* `src/mnc/models/ontology_label_encoder.py`
* `tests/models/test_ontology_label_encoder.py`
