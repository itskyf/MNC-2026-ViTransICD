"""Multilabel classification evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)


@dataclass(frozen=True)
class EvalMetricConfig:
    """Configuration for multilabel evaluation metrics.

    Args:
        label_granularity: Target label granularity.
        num_labels: Number of labels per sample.
        threshold: Decision threshold for binarizing scores.
    """

    label_granularity: Literal["chapter", "code_3char"]
    num_labels: int
    threshold: float

    def __post_init__(self) -> None:
        """Validate metric configuration."""
        if self.num_labels < 1:
            msg = f"num_labels must be >= 1, got {self.num_labels}"
            raise ValueError(msg)
        if not (0.0 < self.threshold < 1.0):
            msg = (
                f"threshold must be strictly between 0.0 and 1.0, got {self.threshold}"
            )
            raise ValueError(msg)


def build_metrics(config: EvalMetricConfig) -> MetricCollection:
    """Build a MetricCollection for multilabel evaluation."""
    return MetricCollection(
        {
            "micro_f1": MultilabelF1Score(
                num_labels=config.num_labels,
                threshold=config.threshold,
                average="micro",
            ),
            "macro_f1": MultilabelF1Score(
                num_labels=config.num_labels,
                threshold=config.threshold,
                average="macro",
            ),
            "micro_precision": MultilabelPrecision(
                num_labels=config.num_labels,
                threshold=config.threshold,
                average="micro",
            ),
            "micro_recall": MultilabelRecall(
                num_labels=config.num_labels,
                threshold=config.threshold,
                average="micro",
            ),
            "macro_auroc": MultilabelAUROC(
                num_labels=config.num_labels,
                average="macro",
            ),
        },
    )


class MultilabelEvaluator:
    """Stateful multilabel evaluator for epoch-level metric accumulation."""

    def __init__(self, config: EvalMetricConfig) -> None:
        """Initialize the evaluator with a configuration."""
        self._config = config
        self._metrics = build_metrics(config)
        self._has_zero = torch.zeros(config.num_labels, dtype=torch.bool)
        self._has_one = torch.zeros(config.num_labels, dtype=torch.bool)

    def _validate(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        expected_ndim = 2
        if scores.ndim != expected_ndim:
            msg = f"scores must be 2-D, got {scores.ndim}-D"
            raise ValueError(msg)
        if targets.ndim != expected_ndim:
            msg = f"targets must be 2-D, got {targets.ndim}-D"
            raise ValueError(msg)
        if scores.shape != targets.shape:
            msg = f"shape mismatch: scores {scores.shape} vs targets {targets.shape}"
            raise ValueError(msg)
        if scores.shape[1] != self._config.num_labels:
            msg = f"expected {self._config.num_labels} labels, got {scores.shape[1]}"
            raise ValueError(msg)
        if not scores.is_floating_point():
            msg = f"scores must be floating point, got {scores.dtype}"
            raise ValueError(msg)
        if targets.is_floating_point():
            msg = f"targets must be integer or boolean, got {targets.dtype}"
            raise ValueError(msg)
        if scores.min().item() < 0.0 or scores.max().item() > 1.0:
            msg = "scores must be in [0, 1]"
            raise ValueError(msg)
        if not torch.all((targets == 0) | (targets == 1)):
            msg = "targets must contain only 0 and 1"
            raise ValueError(msg)

    def update(self, scores: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch of predictions and targets."""
        self._validate(scores, targets)
        self._metrics.update(scores, targets)
        self._has_zero |= (targets == 0).any(dim=0)
        self._has_one |= (targets == 1).any(dim=0)

    def compute(self) -> dict[str, float]:
        """Compute all metrics and return a prefixed dict."""
        prefix = self._config.label_granularity
        result: dict[str, float] = {}
        for name, metric in self._metrics.items():
            if name == "macro_auroc":
                auroc_valid = (self._has_zero & self._has_one).all()
                try:
                    if not auroc_valid:
                        result[f"{prefix}/{name}"] = float("nan")
                    else:
                        value = metric.compute()
                        result[f"{prefix}/{name}"] = float(value)
                except (ValueError, RuntimeError):
                    result[f"{prefix}/{name}"] = float("nan")
            else:
                value = metric.compute()
                result[f"{prefix}/{name}"] = float(value)
        return result

    def reset(self) -> None:
        """Reset all accumulated metric state."""
        self._metrics.reset()
        self._has_zero = torch.zeros(self._config.num_labels, dtype=torch.bool)
        self._has_one = torch.zeros(self._config.num_labels, dtype=torch.bool)
