# TE-2: Eval Metrics Setup

## Goal

Set up a small, reusable evaluation module for multilabel classification.

The module must support both target granularities used in this project:

* ICD chapter
* ICD-10 3-character code

The module must be compatible with later training and benchmarking work, but it must not depend on weak-supervision completion or model architecture details.

## Dependencies

* DE-1 is already complete.
* Frozen schemas in `src/mnc/schemas/` must not be changed.
* TE-2 depends only on tensorized labels and prediction scores produced downstream from:
  * `SilverRecord`
  * `PredictionRecord`

## Scope

In scope:

* torchmetrics-based metric setup for multilabel evaluation
* one shared evaluator API for chapter and `code_3char`
* batch-wise accumulation across an epoch
* deterministic metric naming
* pytest unit tests

Out of scope:

* training loop integration
* dataset split creation
* ontology mapping
* explainability
* report generation
* plotting

## Required Metrics

Headline metrics:

* `micro_f1`
* `macro_f1`
* `macro_auroc`

Support metrics:

* `micro_precision`
* `micro_recall`

These metrics are enough for the reduced-scope feasibility study and keep the implementation close to the original proposal without overexpanding scope.

## Input Contract

The evaluator works on tensors, not raw schema objects.

Required inputs:

* `scores`: `torch.FloatTensor` with shape `[batch_size, num_labels]`
* `targets`: `torch.IntTensor` or `torch.BoolTensor` with shape `[batch_size, num_labels]`

Rules:

* `scores` must be probabilities in `[0, 1]`
* `targets` must be binary values `{0, 1}`
* `num_labels` must be passed explicitly
* `label_granularity` must be either `"chapter"` or `"code_3char"`

This task should not implement record-to-tensor conversion. That belongs to later training/eval wiring.

## Output Contract

`compute()` must return a flat JSON-serializable dictionary:

* keys are stable metric names
* values are Python `float`
* metric names must include granularity prefix

Example key format:

* `chapter/micro_f1`
* `chapter/macro_f1`
* `chapter/macro_auroc`
* `code_3char/micro_precision`

## Suggested File Layout

* `src/mnc/eval/__init__.py`
* `src/mnc/eval/metrics.py`
* `tests/eval/test_metrics.py`

## Public API

Implement one small config object and one evaluator class.

Config:

* `EvalMetricConfig`
* fields:
  * `label_granularity: Literal["chapter", "code_3char"]`
  * `num_labels: int`
  * `threshold: float`

Evaluator:

* `MultilabelEvaluator`
* methods:
  * `update(scores: torch.Tensor, targets: torch.Tensor) -> None`
  * `compute() -> dict[str, float]`
  * `reset() -> None`

Helper:

* `build_metrics(config: EvalMetricConfig) -> torchmetrics.MetricCollection`

## Metric Definitions

Use torchmetrics classification metrics.

Required mapping:

* `micro_f1` -> multilabel F1 with `average="micro"`
* `macro_f1` -> multilabel F1 with `average="macro"`
* `micro_precision` -> multilabel precision with `average="micro"`
* `micro_recall` -> multilabel recall with `average="micro"`
* `macro_auroc` -> multilabel AUROC with `average="macro"`

Use the same `threshold` for thresholded metrics.

Do not add ranking metrics, retrieval metrics, calibration metrics, or custom metrics in this task.

## Validation Rules

Fail fast with typed errors.

Required checks:

* `scores.ndim == 2`
* `targets.ndim == 2`
* `scores.shape == targets.shape`
* `scores.shape[1] == config.num_labels`
* `scores` is floating point
* `targets` is boolean or integer
* `targets` contains only `0/1`
* `threshold` is strictly between `0.0` and `1.0`

Raise `ValueError` for invalid shapes, invalid ranges, or invalid label values.

## AUROC Edge Case Handling

`macro_auroc` may be undefined if some labels have only one class in the accumulated targets.

Required behavior:

* evaluator must not crash the whole evaluation step
* if AUROC cannot be computed, return `float("nan")` for `macro_auroc`
* keep other metrics available

Catch only concrete exception types raised during AUROC compute.

## Non-Goals

This module must not:

* read or write files
* depend on a trainer framework
* mutate schema classes
* infer chapters from codes
* perform threshold search
* compute per-label reports

## Implementation Notes

* Keep the evaluator stateful at epoch scope.
* Use torchmetrics metric objects directly.
* Prefix every output metric name with `label_granularity`.
* Convert scalar tensors to Python floats before returning results.
* Keep the implementation CPU/GPU agnostic.
* Keep docstrings minimal and Google-style.
* Use modern Python typing only.

## Pytest Requirements

Write unit tests with `pytest`.

Minimum required tests:

* perfect predictions return `1.0` for `micro_f1`, `macro_f1`, `micro_precision`, and `micro_recall`
* perfect predictions return `1.0` for `macro_auroc` on a valid non-degenerate target set
* multi-batch accumulation matches single-pass evaluation
* `reset()` clears state correctly
* invalid score shape raises `ValueError`
* invalid target shape raises `ValueError`
* mismatched `num_labels` raises `ValueError`
* non-binary targets raise `ValueError`
* scores outside `[0, 1]` raise `ValueError`
* output keys are prefixed by `label_granularity`
* `compute()` returns JSON-serializable Python floats
* AUROC failure path returns `NaN` instead of crashing

Test constraints:

* no network access
* no external datasets
* use tiny synthetic tensors only
* deterministic expected values

## Definition of Done

TE-2 is done when:

* `src/mnc/eval/metrics.py` exists and exposes the evaluator API
* all required metrics are implemented with torchmetrics
* validation is strict and typed
* pytest unit tests cover normal and failure paths
* `uv run pytest tests/eval/test_metrics.py -v` passes
* no schema files under `src/mnc/schemas/` are changed

## Handoff Notes

This task prepares evaluation only.

Later tasks may build on it to evaluate:

* silver-label training runs
* chapter prediction baselines
* `code_3char` prediction runs
* TransICD-style models with the SEA-LION ModernBERT backbone
