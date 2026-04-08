# Technical Specification: TE-2 Eval Metrics Setup

## Objective

Implement a small evaluation module for training and baseline workflows.

The module must:

* consume existing schemas only from `src/mnc/schemas/`
* compute metrics for two evaluation modes
* use `torchmetrics`
* include basic unit tests with `pytest`

No new persisted schema is required in TE-2.

## Scope

### In scope

* 3-character multi-label evaluation
* chapter classification evaluation
* schema-based validation and tensor conversion
* metric computation utilities
* simple `pytest` unit tests

### Out of scope

* training loop integration
* logging/report rendering
* CLI/service/API layer
* explainability
* calibration, threshold search, bootstrap CI, distributed sync
* new database/file output formats

## Required Inputs

Use only these existing schemas as contracts:

* `PredictionRecord`
* `SilverRecord`
* `OntologyCode`

## Required Outputs

Return in-memory metric dictionaries only.

Recommended return shape:

```python
dict[str, float]
```

Example keys:

```python
{
    "micro_f1": 0.0,
    "macro_f1": 0.0,
    "macro_auc": 0.0,
    "precision_at_1": 0.0,
    "recall_at_1": 0.0,
}
```

TE-2 must not introduce a new Pydantic result schema.

## Supported Modes

### Mode 1: 3-character multi-label

Input contract:

* `PredictionRecord.label_granularity == "code_3char"`
* `SilverRecord.label_granularity == "code_3char"`

Metrics:

* Micro F1
* Macro F1
* Macro-AUC
* P\@k
* R\@k

### Mode 2: chapter classification

Input contract:

* preferred: `PredictionRecord.label_granularity == "chapter"` and `SilverRecord.label_granularity == "chapter"`
* also support projection from `code_3char` to chapter using `OntologyCode.chapter_id`

Metrics:

* Accuracy
* Macro F1
* Macro-AUC

## Functional Requirements

### 1. Public API

Expose one main entry point and two internal mode-specific evaluators.

Recommended API:

```python
evaluate_predictions(
    predictions: list[PredictionRecord],
    targets: list[SilverRecord],
    mode: str,
    ontology: list[OntologyCode] | None = None,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict[str, float]
```

Rules:

* `mode` must be one of: `"code_3char_multilabel"`, `"chapter_classification"`
* `ontology` is required only when chapter evaluation needs projection from 3-character codes
* `ks` applies only to 3-character multi-label mode

### 2. Record alignment

Align records strictly by `doc_id`.

Requirements:

* one target per `doc_id`
* one prediction per `doc_id`
* prediction/target doc sets must match exactly
* raise `ValueError` on duplicates or mismatched doc sets

### 3. Validation rules

Validate before metric computation.

Rules:

* `PredictionRecord.predicted_codes` must be a subset of `PredictionRecord.scores.keys()`
* all labels in `SilverRecord.silver_labels` must be non-empty strings
* `scores` values must be finite floats
* for chapter projection, every used `code_3char` must resolve to a non-null `chapter_id`
* chapter classification mode must resolve to exactly one target chapter per document
* chapter classification mode must resolve to exactly one predicted chapter per document when using discrete predictions
* raise `ValueError` for invalid inputs; do not silently drop labels

### 4. Label vocabulary construction

Build a deterministic label index.

Rules:

* 3-character mode vocabulary = sorted union of:
  * all target `silver_labels`
  * all prediction `scores.keys()`
  * all prediction `predicted_codes`
* chapter mode vocabulary = sorted union of resolved chapter labels
* if `ontology` is supplied, use it only for code-to-chapter projection, not for expanding unseen labels

### 5. Tensor conversion

Convert aligned records into tensors.

For 3-character multi-label:

* `y_true`: shape `[N, C]`, binary multi-hot from `SilverRecord.silver_labels`
* `y_pred`: shape `[N, C]`, binary multi-hot from `PredictionRecord.predicted_codes`
* `y_score`: shape `[N, C]`, float scores from `PredictionRecord.scores`, default missing labels to `0.0`

For chapter classification:

* `y_true_class`: shape `[N]`, integer class ids
* `y_pred_class`: shape `[N]`, integer class ids
* `y_score_class`: shape `[N, K]`, class scores

### 6. Chapter projection logic

Support evaluation from 3-character outputs when chapter labels are not directly predicted.

Projection rules:

* map each `code_3char` to `chapter_id` via `OntologyCode`
* target chapter set = projected chapters from `SilverRecord.silver_labels`
* prediction chapter scores = aggregate code scores per chapter using `max`
* predicted chapter label = `argmax` of chapter scores
* if a target document maps to more than one distinct chapter, raise `ValueError`
* if a prediction cannot be projected because of missing ontology mapping, raise `ValueError`

This keeps chapter mode single-label and compatible with Accuracy, Macro F1, and Macro-AUC.

## Metric Requirements

## Torchmetrics usage

Use `torchmetrics` for all final metric computations.

Implementation note:

* use `get-api-docs` skill to fetch torchmetrics's documentation

### 1. 3-character multi-label metrics

Use:

* `MultilabelF1Score` for Micro F1 and Macro F1
* `MultilabelAUROC` for Macro-AUC

For P\@k and R\@k:

* derive a top-k binary prediction mask from `y_score`
* compute per-sample precision and recall at each `k`
* aggregate with a torchmetrics reducer such as `MeanMetric`

Definitions:

* `P@k = mean_i( hits_i / k )`
* `R@k = mean_i( hits_i / max(1, positives_i) )`

Tie handling:

* rely on `torch.topk`
* deterministic behavior is sufficient; no custom tie-break policy required

### 2. Chapter classification metrics

Use:

* `MulticlassAccuracy`
* `MulticlassF1Score` with `average="macro"`
* `MulticlassAUROC` with `average="macro"`

Requirements:

* class count must match resolved chapter vocabulary size
* `y_score_class` must be dense over all classes

## File Layout

Recommended files:

* `src/mnc/metrics/__init__.py`
* `src/mnc/metrics/evaluator.py`
* `src/mnc/metrics/adapters.py`
* `tests/metrics/test_evaluator.py`

Responsibilities:

* `adapters.py`: validation, alignment, vocabulary build, tensor conversion, chapter projection
* `evaluator.py`: mode dispatch and torchmetrics computation

## Implementation Constraints

* Do not modify schema definitions under `src/mnc/schemas/`
* Do not introduce new persisted artifact schemas
* Keep functions pure and stateless
* Keep error messages explicit and actionable
* Prefer small helpers over one large evaluator function

## Testing Requirements

Use `pytest`.

Minimum required tests:

* perfect 3-character multi-label case returns `1.0` for Micro F1, Macro F1, Macro-AUC, and valid P\@k / R\@k
* partial 3-character multi-label case verifies P\@k and R\@k on a tiny hand-checked example
* direct chapter classification case verifies Accuracy, Macro F1, Macro-AUC
* code-to-chapter projection case verifies ontology-based aggregation works
* invalid input case raises `ValueError` for one of:
  * mismatched `doc_id`
  * duplicate `doc_id`
  * predicted label missing from `scores`
  * multi-chapter target in chapter mode

Test complexity should stay low:

* use tiny synthetic fixtures
* avoid heavy parametrization
* no integration or performance tests

## Acceptance Criteria

TE-2 is complete when:

* evaluator supports both required modes
* evaluator accepts only `PredictionRecord`, `SilverRecord`, and optional `OntologyCode`
* all listed metrics are implemented
* chapter projection from 3-character codes is supported
* invalid schema usage fails fast with clear errors
* `pytest` unit tests pass

## Non-Goals for This Task

Do not add:

* threshold tuning
* sample weighting
* per-label reports
* confusion matrices
* serialization/export
* dashboard/report generation
* distributed metric accumulation across processes

## Suggested Definition of Done

* code merged under `feature/te-2-metrics`
* metrics module is importable by TE-1 and baseline tasks
* tests pass locally with `pytest`
* function docstrings clearly state expected schema inputs and failure conditions
