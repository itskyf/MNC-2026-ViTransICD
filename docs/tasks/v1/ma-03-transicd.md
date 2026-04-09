# MA-3: TransICD Head (Dummy Data)

## 1. Objective

Implement a minimal, production-aligned TransICD-style classification head that consumes token-level outputs from MA-1 and produces:

* code-wise logits
* code-wise attention weights over tokens
* code-specific document representations

The design must stay close to the original TransICD idea: transformer encoder outputs feed a code-wise attention layer, then per-code classifiers produce multi-label scores. The original paper and repo also include LDAM loss and MIMIC-III preprocessing, but those are not part of MA-3. [\[arxiv.org\]](https://arxiv.org/abs/2104.10652), [\[github.com\]](https://github.com/AIMedLab/TransICD)

## 2. Scope

In scope:

* Add one TransICD head module on top of `SeaLionEncoderOutput`
* Support dummy labels and dummy label embeddings
* Support two label modes:
  * internal trainable label queries
  * externally supplied label embeddings
* Return logits and attention maps for explainability-ready downstream use
* Add unit tests with `pytest`
* Keep the module compatible with future MA-4 label encoding and TE-1 training loop

Out of scope:

* MIMIC-III preprocessing
* Porting original TransICD training scripts
* LDAM loss
* Dataset packaging
* Prediction persistence
* Metrics
* Attention-to-text span extraction
* Any real training data dependency

## 3. Required References

The coding agent must:

* use `get-api-docs` skill to fetch the latest API, SDK, or library documentation
* leverage `hf-mcp-server` tools when calling `get-api-docs`
* fetch the latest `transformers` documentation for ModernBERT output behavior before coding against MA-1
* fetch the latest `torch` API docs only if needed for masked softmax.

Reference intent:

* [ModernBERT](https://huggingface.co/docs/transformers/model_doc/modernbert) is an encoder model that exposes sequence hidden states and supports long context up to 8192 tokens.
* The SEA-LION checkpoint used in MA-1 is a ModernBERT-based encoder paired with a custom Gemma 3 tokenizer and 8k context. [\[huggingface.co\]](https://huggingface.co/aisingapore/SEA-LION-ModernBERT-300M), [\[github.com\]](https://github.com/aisingapore/sealion/blob/main/models/sea-embedding/sea-modernbert.md)
* TransICD uses encoder token representations, code-wise attention, and per-code prediction heads. [\[arxiv.org\]](https://arxiv.org/abs/2104.10652), [\[github.com\]](https://github.com/AIMedLab/TransICD)

## 4. Dependency Contract

This task depends on MA-1 only.

Available runtime input from MA-1:

* `doc_ids: list[str]`
* `input_ids: torch.Tensor`
* `attention_mask: torch.Tensor`
* `token_embeddings: torch.Tensor`
* `pooled_embeddings: torch.Tensor`

Relevant shape contract from MA-1:

* `token_embeddings`: `[batch_size, seq_len, hidden_size]`
* `attention_mask`: `[batch_size, seq_len]`

Schema dependency:

* No new persisted schema artifact is created in MA-3
* `doc_id` must remain unchanged for future joins with `PredictionRecord` and `ExplanationRecord`

## 5. Runtime Interface

Recommended module path:

* `src/mnc/models/transicd_head.py`

Recommended public API:

* `TransICDHead`
* `TransICDHeadOutput`

Recommended output container:

```python
@dataclass
class TransICDHeadOutput:
    doc_ids: list[str]
    label_codes: list[str]
    logits: torch.Tensor
    attention_weights: torch.Tensor
    code_representations: torch.Tensor
```

Required tensor shapes:

* `logits`: `[batch_size, num_labels]`
* `attention_weights`: `[batch_size, num_labels, seq_len]`
* `code_representations`: `[batch_size, num_labels, hidden_size]`

Required constructor:

```python
TransICDHead(
    label_codes: list[str],
    hidden_size: int,
    label_embedding_dim: int | None = None,
    dropout: float = 0.1,
)
```

Required forward:

```python
forward(
    encoder_output: SeaLionEncoderOutput,
    label_embeddings: torch.Tensor | None = None,
) -> TransICDHeadOutput
```

## 6. Architecture Requirements

Implement the head as:

* one code-query component
* one code-wise attention computation
* one per-code classifier projection

Required behavior:

* consume `encoder_output.token_embeddings` and `encoder_output.attention_mask`
* compute one attention distribution per label over valid tokens
* aggregate token embeddings into one code-specific representation per label
* compute one logit per label
* pass `doc_ids` through unchanged

Recommended parameterization:

* internal label query table: `nn.Embedding(num_labels, hidden_size)` when no external label embeddings are supplied
* optional label projection layer when `label_embedding_dim != hidden_size`
* per-label classifier weights: `nn.Parameter` of shape `[num_labels, hidden_size]`
* per-label classifier bias: `nn.Parameter` of shape `[num_labels]`

Recommended computation:

* attention scores via dot product between label query and token embeddings
* masked softmax over sequence dimension
* weighted sum of token embeddings to produce code-specific representation
* per-label linear scoring over each code-specific representation

## 7. Label Semantics Contract

MA-3 must not depend on MA-4, but it must be ready for MA-4.

Support both paths:

* internal label queries for dummy-data execution now
* external `label_embeddings` tensor for future semantic labels

Required external embedding contract:

* shape: `[num_labels, label_embedding_dim or hidden_size]`
* label order must match `label_codes`
* if supplied, external embeddings replace internal queries for attention computation

Non-requirement:

* do not build label text encoders in MA-3
* do not read `OntologyCode` directly in MA-3
* do not tokenize label descriptions in MA-3

## 8. Attention and Masking Requirements

Masking is mandatory.

Required behavior:

* padded tokens must not receive probability mass
* masked attention must be numerically stable
* attention must normalize over valid sequence positions only
* if a sequence has zero valid tokens, raise `ValueError`

Recommended implementation:

* set masked positions to a large negative value before softmax
* apply softmax across `seq_len`
* optionally zero masked positions again after softmax as a safety step

Explainability requirement:

* return full `attention_weights` for EX-2 compatibility
* keep attention aligned to MA-1 token order exactly

## 9. Dummy Data Expectations

MA-3 must run without real datasets.

Use in-memory fixtures only.

Dummy scenarios to support:

* one document with two labels
* multi-document batch with three or more labels
* internal label-query mode
* external label-embedding mode
* variable sequence lengths with padding in the attention mask

Dummy label examples:

* `["A00", "B20", "C34"]`

Do not require:

* MIMIC-III files
* ontology snapshots
* silver labels
* training splits

The original TransICD repo includes both model code and a large MIMIC-III preprocessing pipeline; MA-3 must read that repo for architectural reference only and must not port its data pipeline. [\[github.com\]](https://github.com/AIMedLab/TransICD)

## 10. Error Handling

Required checks:

* raise `ValueError` if `label_codes` is empty
* raise `ValueError` if `label_codes` contains duplicates
* raise `ValueError` if `hidden_size <= 0`
* raise `ValueError` if `encoder_output.token_embeddings.ndim != 3`
* raise `ValueError` if `encoder_output.attention_mask.ndim != 2`
* raise `ValueError` if batch or sequence dimensions do not match
* raise `ValueError` if external `label_embeddings` count does not match `label_codes`
* raise `ValueError` if external embedding dimension is incompatible and no projection path is configured
* keep exception messages short and explicit

Do not:

* silently reorder labels
* silently broadcast malformed tensors
* silently drop labels
* mutate `encoder_output`

## 11. Testing Requirements

Use `pytest`.

Recommended test path:

* `tests/models/test_transicd_head.py`

Minimum unit tests:

1. initialization smoke test with dummy labels
2. forward returns all required fields
3. output shapes match contract
4. `doc_ids` are preserved
5. `label_codes` are preserved
6. masked positions receive near-zero attention probability
7. attention sums to 1 across valid tokens
8. external label embeddings path works
9. duplicate label codes raise `ValueError`
10. empty label list raises `ValueError`
11. mismatched tensor shapes raise `ValueError`
12. no real model download is required for MA-3 tests

Test guidance:

* create synthetic `SeaLionEncoderOutput` tensors directly
* avoid loading the Hugging Face backbone in MA-3 unit tests
* assert shapes and behavior, not exact numeric logits

## 12. Implementation Notes

Recommended defaults:

* use the encoder hidden size from MA-1 runtime tensors
* keep dropout default at `0.1`
* use `torch.float32` for head parameters unless the caller explicitly casts the module
* do not add a loss function in this task

Recommended file updates:

* add `src/mnc/models/transicd_head.py`
* export new symbols in `src/mnc/models/__init__.py`

Keep the code minimal:

* no registry
* no trainer hooks
* no dataset readers
* no ontology loaders
* no config framework unless already standard in repo

## 13. Required Subagent Plan for Reading Upstream TransICD

The coding agent should clone the upstream TransICD repo into a temporary folder for reference. The repo contains both model code and MIMIC-III preprocessing/training scripts, so reading must be split across focused subagents. The official repo layout explicitly includes `code/` for model/training code and `mimicdata/` for preprocessing inputs. [\[github.com\]](https://github.com/AIMedLab/TransICD)

Spawn exactly four subagents:

* Subagent 1: repo mapper
* Subagent 2: model-path reader
* Subagent 3: data-boundary reader
* Subagent 4: parity summarizer

Subagent 1: repo mapper

* identify the exact files that define the TransICD model, attention block, classifier block, and training entrypoint
* ignore generated result files
* produce a short file map with one-line purpose per file

Subagent 2: model-path reader

* read only the model implementation path
* extract the actual attention formula
* extract tensor shapes at each forward stage
* extract how per-label classifiers are parameterized
* extract whether label queries and classifier weights are shared or separate

Subagent 3: data-boundary reader

* inspect preprocessing only enough to understand:
  * label ordering assumptions
  * padding conventions
  * sequence length assumptions
  * any label-vocabulary indexing logic
* do not port any MIMIC-III logic
* do not bring dataset-specific constants into MA-3 unless strictly needed for shape parity

Subagent 4: parity summarizer

* reconcile paper architecture with repo implementation
* identify which parts are core to preserve in MA-3
* identify which parts are dataset- or training-specific and must be excluded
* write one concise adaptation note for integrating the head with MA-1 SEA-LION encoder outputs

Required deliverable from the subagent phase:

* one short implementation memo in PR notes or agent scratchpad containing:
  * file map
  * core forward formula
  * shape summary
  * excluded repo parts
  * final MA-3 adaptation decisions

## 14. Adaptation Rules from Original TransICD to This Repo

Preserve:

* code-wise attention over token embeddings
* one code-specific representation per label
* one score per label
* exported attention maps for explainability

Adapt:

* replace the original encoder with MA-1 `SeaLionEncoderOutput`
* replace original dataset label ids with `label_codes: list[str]`
* allow optional external label embeddings for MA-4 readiness

Exclude:

* MIMIC-III preprocessing
* original training script wiring
* LDAM loss
* original max document length assumptions such as 2500 from the reference repo hyperparameter setup; MA-3 must trust MA-1 sequence length instead. [\[github.com\]](https://github.com/AIMedLab/TransICD), [\[huggingface.co\]](https://huggingface.co/docs/transformers/model_doc/modernbert)

## 15. Acceptance Criteria

The task is done when all conditions below are true:

1. A coding agent can instantiate `TransICDHead` with dummy ICD-style label codes.
2. The head accepts `SeaLionEncoderOutput` from MA-1 directly.
3. The head returns logits, code representations, and attention maps.
4. Attention respects the encoder attention mask.
5. `doc_ids` and label order remain stable.
6. The head works with both internal label queries and external label embeddings.
7. The implementation runs on dummy data only.
8. Unit tests pass with `pytest`.
9. No MIMIC-III preprocessing or upstream training code is copied into the repo.

## 16. Deliverables

Expected deliverables:

* `src/mnc/models/transicd_head.py`
* `tests/models/test_transicd_head.py`
* `src/mnc/models/__init__.py` update
