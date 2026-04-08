# Technical Specification: MA-1 SEA-LION Backbone (Dummy Data)

## 1. Objective

Implement a minimal, production-aligned encoder backbone for [`aisingapore/SEA-LION-ModernBERT-300M`](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M) using `transformers`. The integration must expose token-level embeddings and one pooled document representation for downstream heads. The model is an encoder-only [ModernBERT](https://huggingface.co/docs/transformers/model_doc/modernbert) variant, uses a custom Gemma 3 tokenizer, and supports up to 8k context, which is the main reason to use it over legacy BERT-style backbones for long clinical text.

This task is foundation-only. It must support P0/P1 downstream work:

* MA-3 TransICD head
* MA-2 linear head
* MA-4 label encoding
* TE-1 training loop
* TE-2 metrics integration
* EX-1 token-to-text alignment

## 2. Scope

In scope:

* Load tokenizer and backbone from Hugging Face using `transformers`
* Use fixed model id: `aisingapore/SEA-LION-ModernBERT-300M`
* Use `torch_dtype=torch.float16`
* Use `device_map="auto"`
* Accept schema-based document input from `src/mnc/schemas/`
* Produce runtime encoder outputs for downstream heads:
  * contextual token embeddings
  * pooled representation
  * attention mask
  * token ids
  * doc id mapping
* Support dummy-data execution with small in-memory fixtures
* Add unit tests with `pytest`

Out of scope:

* Classification logits
* Loss computation
* Label decoding
* Prediction persistence
* Training loop
* Explainability logic
* Any P2 work

## 3. Required References

The coding agent must:

* use `get-api-docs` skill to fetch transformers's documentation
* use `transformers` **v5**, reference the `modernbert` model card.

## 4. Schema Contract

Pipeline-facing input must be based on existing schemas only.

Accepted input artifact:

* `list[DocumentRecord]`

Relevant schema fields:

* `doc_id`
* `normalized_text`
* `retrieval_text`
* `raw_text`

Text selection rule:

* Use `normalized_text` if non-empty
* Else use `retrieval_text` if non-empty
* Else use `raw_text`

Pipeline-facing output:

* No persisted schema artifact is created in MA-1
* The backbone returns runtime tensors only
* `doc_id` must be preserved exactly for downstream joins with `SilverRecord`, `PredictionRecord`, `ExplanationRecord`, and alignment utilities

Rationale:

* No embedding schema exists in `src/mnc/schemas/`
* MA-1 is an internal model component, not a persistence boundary

## 5. Runtime Interface

Implement one encoder module and one output container.

Recommended module path:

* `src/mnc/models/sea_lion_encoder.py`

Recommended public API:

* `SeaLionEncoder`
* `SeaLionEncoderOutput`

Required behavior:

* `SeaLionEncoder` initializes tokenizer and model lazily or at construction time
* `encode_documents(documents: list[DocumentRecord], max_length: int | None = None) -> SeaLionEncoderOutput`
* `forward(tokenized_batch: dict[str, torch.Tensor]) -> SeaLionEncoderOutput`

Required output fields:

* `doc_ids: list[str]`
* `input_ids: torch.Tensor`
* `attention_mask: torch.Tensor`
* `token_embeddings: torch.Tensor`
* `pooled_embeddings: torch.Tensor`

Tensor shape contract:

* `input_ids`: `[batch_size, seq_len]`
* `attention_mask`: `[batch_size, seq_len]`
* `token_embeddings`: `[batch_size, seq_len, hidden_size]`
* `pooled_embeddings`: `[batch_size, hidden_size]`

## 6. Backbone Requirements

Load the backbone with:

* `AutoTokenizer.from_pretrained(model_id)`
* `AutoModel.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")`

Do not use task-specific heads.

Use the tokenizer shipped with the same checkpoint. The model card explicitly states this model is paired with a custom Gemma 3 tokenizer, and the base model is intended for downstream fine-tuning/classification use cases.

## 7. Pooling Requirements

Use masked mean pooling as the default pooled representation.

Reason:

* `ModernBertModel` returns `last_hidden_state` as the base output, not a ready-to-use `pooler_output`
* ModernBERT documentation exposes sequence hidden states from the base model, while classifier pooling is handled in task heads
* Mean pooling is explicitly recognized in the model card glossary and is more stable for long-text document encoding than relying only on CLS.

Pooling contract:

* Ignore padded positions using `attention_mask`
* Clamp denominator to avoid divide-by-zero
* Return one vector per document

Optional extension:

* Support `pooling="mean"` now
* Keep code structure easy to extend to `pooling="cls"` later
* Default must remain `mean`

## 8. Long-Context Handling

Requirements:

* Do not hardcode legacy BERT limits such as 512
* Allow caller-provided `max_length`
* If `max_length` is `None`, use tokenizer/model defaults
* Support truncation and padding in batch tokenization

Why:

* ModernBERT supports sequences up to 8192 tokens and introduces architectural changes such as rotary positional embeddings and alternating attention for long-context efficiency. [\[huggingface.co\]](https://huggingface.co/docs/transformers/model_doc/modernbert), [\[huggingface.co\]](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M)

Non-requirement:

* Do not implement chunking, sliding windows, or padding-free FlashAttention in MA-1
* Keep MA-1 simple and safe for dummy-data integration

## 9. Downstream Compatibility Requirements

The encoder component must be head-agnostic.

It must support both:

* linear head consumers needing `[batch, hidden]`
* TransICD-style code-wise attention consumers needing `[batch, seq_len, hidden]` plus `attention_mask`

Minimum compatibility guarantees:

* `pooled_embeddings` feeds MA-2
* `token_embeddings` and `attention_mask` feed MA-3
* `input_ids` and `doc_ids` support EX-1 alignment and auditability
* Batch order must remain stable from input documents to output tensors

## 10. Error Handling

Required checks:

* Raise `ValueError` if input list is empty
* Raise `ValueError` if all candidate text fields are empty for any document
* Raise `TypeError` if items are not `DocumentRecord`
* Keep exceptions explicit and short

Do not:

* silently drop documents
* mutate input schemas
* create fallback dummy text automatically

## 11. Dummy Data Expectations

MA-1 must be runnable without real training data.

Use test fixtures that build small valid `DocumentRecord` objects in memory. This task must not depend on bronze/silver/gold datasets.

Dummy-data scenarios to support:

* single short Vietnamese document
* multi-document batch
* document with `normalized_text=""` falling back to `raw_text`

## 12. Testing Requirements

Use `pytest`.

Keep tests simple. No integration test with distributed hardware is required.

Minimum unit tests:

1. model/tokenizer load smoke test
2. `encode_documents` returns all required fields
3. batch size equals number of input documents
4. `doc_ids` preserve input order
5. `token_embeddings.ndim == 3`
6. `pooled_embeddings.ndim == 2`
7. pooled embedding batch dimension matches input size
8. fallback text selection works as specified
9. empty input raises `ValueError`
10. invalid item type raises `TypeError`

Test implementation guidance:

* keep fixtures tiny
* avoid asserting exact numeric embedding values
* assert shapes, dtypes, non-empty outputs, and contract behavior only

Recommended test path:

* `tests/models/test_sea_lion_encoder.py`

## 13. Implementation Notes

Recommended design:

* one small dataclass for runtime output
* one `nn.Module` backbone for tokenizer + encoder
* one private helper to extract text from `DocumentRecord`
* one private helper for masked mean pooling

Recommended defaults:

* `model_name="aisingapore/SEA-LION-ModernBERT-300M"`
* `torch_dtype=torch.float16`
* `device_map="auto"`
* `pooling="mean"`
* `return_dict=True`

Keep the code minimal:

* no registry system
* no abstract factory
* no training logic
* no logging framework unless already standard in repo

## 14. Acceptance Criteria

The task is done when all conditions below are true:

1. A coding agent can instantiate `SeaLionEncoder` with the fixed SEA-LION checkpoint.
2. The encoder component accepts `list[DocumentRecord]` and tokenizes using the model’s own tokenizer.
3. The backbone returns token embeddings and pooled embeddings from the base encoder output.
4. The backbone preserves `doc_id` order for downstream joinability.
5. The backbone supports both MA-2 and MA-3 input needs without extra adaptation.
6. The backbone runs on dummy data only.
7. Unit tests pass with `pytest`.

## 15. Deliverable

Deliver one PyTorch encoder backbone class plus tests.

Expected deliverables:

* `src/mnc/models/sea_lion_encoder.py`
* `tests/models/test_sea_lion_encoder.py`
