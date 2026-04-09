# Technical Specification: EX-1 Token-to-Text Alignment

## Purpose

Implement a runtime alignment utility that maps model token positions back to `DocumentRecord.normalized_text`.

This task serves explainability only. It provides the alignment contract required by:

* EX-2 attention-based explanation
* EX-3 gradient saliency explanation
* any audit/debug view that needs token indices, token strings, and character spans in normalized text

## Dependencies

* DC-1 is required because EX-1 aligns against `normalized_text`
* MA-1 is required because EX-1 must stay consistent with the SEA-LION tokenizer and encoder sequence layout

## Scope

In scope:

* align token indices to character offsets in `normalized_text`
* preserve `doc_id` and input order
* support batch alignment for `list[DocumentRecord]`
* support validation against `SeaLionEncoderOutput.input_ids` and `attention_mask`
* handle special tokens and padded tokens explicitly
* handle truncation consistently with MA-1 tokenization
* add simple `pytest` unit tests

Out of scope:

* attention scoring
* gradient computation
* span ranking
* explanation text generation
* persistence of a new schema artifact
* any new data layer outside `data/{bronze,silver,gold}/...`

## Non-goals

* do not modify DC-1 outputs
* do not create JSONL/Parquet alignment datasets in P0
* do not add ontology logic
* do not add training logic
* do not introduce a new tokenizer different from MA-1

## Input Contract

Accepted pipeline input:

* `list[DocumentRecord]`
* optional `SeaLionEncoderOutput` from `src/mnc/models/sea_lion_encoder.py`

Required `DocumentRecord` fields:

* `doc_id`
* `normalized_text`

Text rule:

* use `normalized_text` only
* raise `ValueError` if `normalized_text` is empty for any document

Rationale:

* EX-1 explicitly aligns back to normalized text
* DC-1 already guarantees normalized text for valid silver documents
* using one fixed text field avoids offset drift

## Output Contract

EX-1 creates runtime-only data structures. Do not add a persisted Pydantic schema.

Implement two dataclasses.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class TokenAlignment:
    token_index: int
    token_id: int
    token_text: str
    char_start: int | None
    char_end: int | None
    text: str | None
    is_special: bool
    is_padding: bool

@dataclass(frozen=True)
class DocumentTokenAlignment:
    doc_id: str
    normalized_text: str
    input_ids: list[int]
    attention_mask: list[int]
    tokens: list[TokenAlignment]
```

Field semantics:

* `token_index` is the position in the encoded sequence after special tokens, truncation, and padding
* `token_id` is the tokenizer vocabulary id at `token_index`
* `token_text` is the decoded single-token string from the tokenizer
* `char_start` and `char_end` are offsets into `normalized_text`
* `char_end` is exclusive
* `text` is `normalized_text[char_start:char_end]` when offsets exist, else `None`
* `is_special=True` for tokenizer special tokens
* `is_padding=True` for padded positions where `attention_mask == 0`

## Alignment Rules

Use the same tokenizer family as MA-1:

* default model id: `aisingapore/SEA-LION-ModernBERT-300M`
* use `AutoTokenizer.from_pretrained(model_name)`

Tokenizer requirements:

* require a fast tokenizer
* use `return_offsets_mapping=True`
* use `padding=True`
* use `truncation=True`
* pass through caller-provided `max_length`

Offset rules:

* offsets must be relative to `normalized_text`
* special tokens must have `char_start=None` and `char_end=None`
* padded tokens must have `char_start=None` and `char_end=None`
* zero-length offsets from special handling must map to `None`, not `(0, 0)`
* non-null offsets must satisfy `0 <= char_start < char_end <= len(normalized_text)`

Sequence rules:

* batch order must remain identical to input document order
* token count in each `DocumentTokenAlignment.tokens` must equal encoded `seq_len`
* if truncation occurs, align only the retained token sequence
* do not attempt to reconstruct dropped text beyond `max_length`

Consistency rules with MA-1:

* if `encoder_output` is provided, `doc_ids` must match exactly
* if `encoder_output` is provided, aligned `input_ids` must match `encoder_output.input_ids` exactly
* if `encoder_output` is provided, aligned `attention_mask` must match `encoder_output.attention_mask` exactly
* mismatch must raise `ValueError`

## Public API

Implement one runtime aligner module.

Recommended file:

* `src/mnc/explain/alignment.py`

Required public API:

```python
class TokenTextAligner:
    def __init__(
        self,
        model_name: str = "aisingapore/SEA-LION-ModernBERT-300M",
    ) -> None: ...

    def align_documents(
        self,
        documents: list[DocumentRecord],
        max_length: int | None = None,
    ) -> list...

    def align_with_encoder_output(
        self,
        documents: list[DocumentRecord],
        encoder_output: SeaLionEncoderOutput,
        max_length: int | None = None,
    ) -> list...
```

Behavior:

* `align_documents` tokenizes `normalized_text` and returns runtime alignments
* `align_with_encoder_output` re-tokenizes with the same settings, validates equality against `encoder_output`, then returns runtime alignments
* both methods must preserve input order

## Internal Helpers

Implement small private helpers inside `src/mnc/explain/alignment.py`.

Required helpers:

```python
def _extract_alignment_text(doc: DocumentRecord) -> str: ...
def _build_token_alignment(
    text: str,
    token_index: int,
    token_id: int,
    token_text: str,
    offset: tuple[int, int],
    attention_value: int,
    special_token_ids: set[int],
) -> TokenAlignment: ...
def _validate_encoder_match(
    doc_ids: list[str],
    alignments: list[DocumentTokenAlignment],
    encoder_output: SeaLionEncoderOutput,
) -> None: ...
```

Implementation rules:

* keep helpers deterministic
* keep exceptions short and explicit
* do not mutate input documents
* do not mutate `SeaLionEncoderOutput`

## MA-1 Compatibility Notes

EX-1 must not change the MA-1 output contract.

A minimal MA-1-compatible update is allowed:

* store `self.model_name = model_name` in `SeaLionEncoder.__init__`

Do not require MA-1 to return `offset_mapping`.

Rationale:

* keeps MA-1 stable
* keeps EX-1 isolated
* avoids expanding encoder runtime outputs before EX-2/EX-3 need them

## Error Handling

Required checks:

* raise `ValueError` if input document list is empty
* raise `TypeError` if any item is not `DocumentRecord`
* raise `ValueError` if `normalized_text` is empty for any document
* raise `RuntimeError` if tokenizer is not a fast tokenizer and cannot provide offset mapping
* raise `ValueError` on any encoder-output mismatch in `doc_ids`, `input_ids`, or `attention_mask`
* raise `ValueError` if any non-null offset is out of range

Do not:

* silently fall back to `retrieval_text`
* silently drop tokens
* silently repair mismatched encoder outputs

## Storage and Data Layout

No new persisted artifact is required in EX-1.

Rules:

* do not create `data/raw/` or any non-approved layer
* do not write alignment JSONL by default
* if a temporary debug file is needed during local development, keep it outside committed deliverables
* if the agent needs sample DC-1 data in the worktree, it may copy `data/silver/<dataset_name>/documents/...` from `main` into the worktree

## Code Location

Create:

* `src/mnc/explain/alignment.py`

Optional update:

* `src/mnc/explain/__init__.py`

Minimal MA-1-compatible update if needed:

* `src/mnc/models/sea_lion_encoder.py`

Tests:

* `tests/explain/test_alignment.py`

Do not create a new top-level package.

## Testing Requirements

Use `pytest`.

Keep tests simple and local. No network calls.

Minimum unit tests:

1. `align_documents` returns one `DocumentTokenAlignment` per input document
2. returned `doc_id` order matches input order
3. each document alignment length matches encoded `seq_len`
4. non-special, non-padding tokens produce valid offsets within `normalized_text`
5. special tokens have null offsets
6. padded tokens have null offsets
7. `text` equals `normalized_text[char_start:char_end]` for aligned content tokens
8. empty input raises `ValueError`
9. invalid item type raises `TypeError`
10. empty `normalized_text` raises `ValueError`
11. `align_with_encoder_output` accepts matching `SeaLionEncoderOutput`
12. `align_with_encoder_output` raises `ValueError` on mismatched `input_ids`

Test implementation guidance:

* use tiny in-memory `DocumentRecord` fixtures
* prefer short Vietnamese examples
* avoid asserting exact token strings for every token if tokenizer normalization is brittle
* assert structural contract, offset validity, and encoder consistency
* if model download is expensive in CI, mock tokenizer behavior for most tests and keep one smoke test optional or marked

## Suggested Test Fixtures

Use small documents such as:

* `"Bệnh nhân đau ngực và khó thở."`
* `"Chẩn đoán viêm phổi. Điều trị bằng kháng sinh."`

Fixture rules:

* set `normalized_text` explicitly
* keep `retrieval_text` irrelevant for EX-1 tests
* keep timestamps fixed

## Acceptance Criteria

The task is done when all conditions below are true:

1. a coding agent can align SEA-LION token positions back to `DocumentRecord.normalized_text`
2. the aligner preserves `doc_id`, sequence order, and token indices
3. offsets are valid character spans into normalized text
4. special and padded tokens are represented explicitly with null offsets
5. alignment can be validated against `SeaLionEncoderOutput`
6. no new persisted schema or data layer is introduced
7. required `pytest` unit tests exist and pass

## Deliverables

Expected deliverables:

* `src/mnc/explain/alignment.py`
* `tests/explain/test_alignment.py`

Optional small update:

* `src/mnc/models/sea_lion_encoder.py`

## Implementation Notes

Recommended execution order:

1. implement runtime dataclasses
2. implement tokenizer-based alignment with offset mapping
3. implement encoder-output validation
4. add unit tests
5. run local `pytest`

Keep the first version minimal and strict. EX-1 is a runtime bridge between MA-1 tokenization and downstream explanation modules, not a new dataset stage.
