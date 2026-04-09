"""Unit tests for multilabel evaluation metrics."""

from __future__ import annotations

import json
import math

import pytest
import torch

from mnc.eval.metrics import EvalMetricConfig, MultilabelEvaluator

_GRANULARITY = "chapter"


def _config(
    *,
    label_granularity: str = _GRANULARITY,
    num_labels: int = 4,
    threshold: float = 0.5,
) -> EvalMetricConfig:
    """Build an EvalMetricConfig with sensible test defaults."""
    return EvalMetricConfig(
        label_granularity=label_granularity,
        num_labels=num_labels,
        threshold=threshold,
    )


# -- Perfect predictions -------------------------------------------------------


class TestPerfectPredictions:
    def test_thresholded_metrics_at_one(self) -> None:
        config = _config()
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor(
            [
                [0.9, 0.9, 0.1, 0.1],
                [0.1, 0.9, 0.9, 0.1],
                [0.9, 0.1, 0.9, 0.1],
                [0.1, 0.1, 0.1, 0.9],
            ],
        )
        targets = torch.tensor(
            [[1, 1, 0, 0], [0, 1, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]],
        )
        evaluator.update(scores, targets)
        result = evaluator.compute()

        assert result["chapter/micro_f1"] == 1.0
        assert result["chapter/macro_f1"] == 1.0
        assert result["chapter/micro_precision"] == 1.0
        assert result["chapter/micro_recall"] == 1.0

    def test_macro_auroc_perfect(self) -> None:
        config = _config()
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor(
            [
                [0.9, 0.1, 0.9, 0.1],
                [0.1, 0.9, 0.9, 0.1],
                [0.9, 0.1, 0.1, 0.9],
                [0.1, 0.9, 0.1, 0.9],
            ],
        )
        targets = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 1, 0], [1, 0, 0, 1], [0, 1, 0, 1]],
        )
        evaluator.update(scores, targets)
        result = evaluator.compute()

        assert result["chapter/macro_auroc"] == 1.0


# -- Multi-batch accumulation --------------------------------------------------


class TestMultiBatch:
    def test_accumulation_matches_single_pass(self) -> None:
        config = _config(num_labels=3)
        all_scores = torch.tensor(
            [[0.9, 0.2, 0.8], [0.1, 0.8, 0.3], [0.7, 0.3, 0.9], [0.2, 0.7, 0.4]],
        )
        all_targets = torch.tensor(
            [[1, 0, 1], [0, 1, 0], [1, 0, 1], [0, 1, 0]],
        )

        # Multi-batch
        evaluator = MultilabelEvaluator(config)
        evaluator.update(all_scores[:2], all_targets[:2])
        evaluator.update(all_scores[2:], all_targets[2:])
        multi_result = evaluator.compute()

        # Single-pass
        single_eval = MultilabelEvaluator(config)
        single_eval.update(all_scores, all_targets)
        single_result = single_eval.compute()

        for key in multi_result:
            assert abs(multi_result[key] - single_result[key]) < 1e-6, f"{key} mismatch"


# -- Reset ---------------------------------------------------------------------


class TestReset:
    def test_reset_clears_state(self) -> None:
        config = _config()
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor([[0.9, 0.1, 0.9, 0.1]])
        targets = torch.tensor([[1, 0, 1, 0]])

        evaluator.update(scores, targets)
        evaluator.reset()
        evaluator.update(scores, targets)
        result = evaluator.compute()

        assert result["chapter/micro_f1"] == 1.0


# -- Validation ----------------------------------------------------------------


class TestValidation:
    def test_invalid_score_shape(self) -> None:
        config = _config()
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
        targets = torch.tensor([[1, 0, 1, 0]])
        with pytest.raises(ValueError, match="2-D"):
            evaluator.update(scores, targets)

    def test_invalid_target_shape(self) -> None:
        config = _config()
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        targets = torch.tensor([1, 0, 1, 0])
        with pytest.raises(ValueError, match="2-D"):
            evaluator.update(scores, targets)

    def test_mismatched_num_labels(self) -> None:
        config = _config(num_labels=4)
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor([[0.5, 0.5, 0.5]])
        targets = torch.tensor([[1, 0, 1]])
        with pytest.raises(ValueError, match="4 labels"):
            evaluator.update(scores, targets)

    def test_non_binary_targets(self) -> None:
        config = _config()
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor([[0.5, 0.5, 0.5, 0.5]])
        targets = torch.tensor([[1, 0, 2, 0]])
        with pytest.raises(ValueError, match="0 and 1"):
            evaluator.update(scores, targets)

    def test_scores_above_one(self) -> None:
        config = _config()
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor([[1.5, 0.5, 0.5, 0.5]])
        targets = torch.tensor([[1, 0, 1, 0]])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            evaluator.update(scores, targets)

    def test_scores_below_zero(self) -> None:
        config = _config()
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor([[-0.1, 0.5, 0.5, 0.5]])
        targets = torch.tensor([[1, 0, 1, 0]])
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            evaluator.update(scores, targets)

    def test_invalid_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            _config(threshold=1.5)

    def test_threshold_zero(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            _config(threshold=0.0)


# -- Output format -------------------------------------------------------------


class TestOutputFormat:
    def test_keys_prefixed_by_granularity(self) -> None:
        config = _config(label_granularity="code_3char")
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor([[0.9, 0.1, 0.9, 0.1]])
        targets = torch.tensor([[1, 0, 1, 0]])
        evaluator.update(scores, targets)
        result = evaluator.compute()

        for key in result:
            assert key.startswith("code_3char/"), f"key {key} not prefixed"

    def test_returns_python_floats(self) -> None:
        config = _config()
        evaluator = MultilabelEvaluator(config)
        scores = torch.tensor([[0.9, 0.1, 0.9, 0.1]])
        targets = torch.tensor([[1, 0, 1, 0]])
        evaluator.update(scores, targets)
        result = evaluator.compute()

        for key, value in result.items():
            assert isinstance(value, float), f"{key} is {type(value)}, not float"

        serialized = json.dumps(result)
        assert isinstance(serialized, str)


# -- AUROC edge case -----------------------------------------------------------


class TestAUROCEdgeCase:
    def test_auroc_nan_on_degenerate_targets(self) -> None:
        config = _config(num_labels=2)
        evaluator = MultilabelEvaluator(config)
        # Both labels are degenerate: label 0 has all 1s, label 1 has all 0s
        scores = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
        targets = torch.tensor([[1, 0], [1, 0]])
        evaluator.update(scores, targets)
        result = evaluator.compute()

        assert math.isnan(result["chapter/macro_auroc"])
        # Other metrics must remain available
        assert isinstance(result["chapter/micro_f1"], float)
        assert not math.isnan(result["chapter/micro_f1"])
