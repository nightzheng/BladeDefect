"""Normalized segmentation metric representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SegmentationMetrics:
    precision: float
    recall: float
    map50: float
    map50_95: float
    f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _mean(value: Any) -> float:
    if hasattr(value, "mean"):
        value = value.mean()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def metrics_from_ultralytics(result: Any) -> SegmentationMetrics:
    """Extract mask metrics from an Ultralytics validation result."""
    metrics = getattr(result, "seg", None)
    if metrics is None:
        raise ValueError("验证结果不包含 segmentation 指标，请确认使用 seg 模型和数据")
    precision = _mean(metrics.p)
    recall = _mean(metrics.r)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SegmentationMetrics(
        precision=precision,
        recall=recall,
        map50=_mean(metrics.map50),
        map50_95=_mean(metrics.map),
        f1=f1,
    )
