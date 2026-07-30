"""Normalized segmentation metric representation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

FPS_METHOD = "ultralytics_val_speed_inference_ms"


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


def _metric_block(metrics: Any) -> dict[str, float]:
    precision = _mean(metrics.p)
    recall = _mean(metrics.r)
    return {
        "precision": precision,
        "recall": recall,
        "map50": _mean(metrics.map50),
        "map50_95": _mean(metrics.map),
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def extended_metrics_from_ultralytics(result: Any) -> dict[str, Any]:
    """提取 Mask 与 Box 分离的指标；Mask 字段缺失时抛错，Box 缺失时留空。"""
    seg = getattr(result, "seg", None)
    if seg is None:
        raise ValueError("验证结果不包含 segmentation 指标，请确认使用 seg 模型和数据")
    mask = _metric_block(seg)
    box_metrics = getattr(result, "box", None)
    box = _metric_block(box_metrics) if box_metrics is not None else None
    payload: dict[str, Any] = {
        "mask_precision": mask["precision"],
        "mask_recall": mask["recall"],
        "mask_mAP50": mask["map50"],
        "mask_mAP50-95": mask["map50_95"],
        "mask_f1": mask["f1"],
        "box_precision": box["precision"] if box else None,
        "box_recall": box["recall"] if box else None,
        "box_mAP50": box["map50"] if box else None,
        "box_mAP50-95": box["map50_95"] if box else None,
        "box_f1": box["f1"] if box else None,
    }
    return payload


def _per_class_block(metrics: Any, names: Any, metric_branch: str) -> list[dict[str, Any]]:
    class_index = getattr(metrics, "ap_class_index", None)
    if class_index is None:
        return []
    try:
        indices = [int(value) for value in class_index]
    except TypeError:
        return []

    def _column(values: Any) -> list[float]:
        try:
            return [float(value) for value in values]
        except TypeError:
            return []

    precisions = _column(getattr(metrics, "p", []))
    recalls = _column(getattr(metrics, "r", []))
    ap50s = _column(getattr(metrics, "ap50", []))
    aps = _column(getattr(metrics, "ap", []))
    rows: list[dict[str, Any]] = []
    for position, class_id in enumerate(indices):
        if isinstance(names, dict):
            class_name = str(names.get(class_id, f"unknown_{class_id}"))
        elif isinstance(names, (list, tuple)) and class_id < len(names):
            class_name = str(names[class_id])
        else:
            class_name = f"unknown_{class_id}"
        rows.append({
            "metric_branch": metric_branch,
            "class_id": class_id,
            "class_name": class_name,
            "precision": precisions[position] if position < len(precisions) else None,
            "recall": recalls[position] if position < len(recalls) else None,
            "ap50": ap50s[position] if position < len(ap50s) else None,
            "ap50_95": aps[position] if position < len(aps) else None,
        })
    return rows


def per_class_metrics_from_ultralytics(result: Any) -> list[dict[str, Any]]:
    """提取逐类 Precision/Recall/AP50/AP50-95，Mask 与 Box 分支分行记录。"""
    names = getattr(result, "names", {})
    rows: list[dict[str, Any]] = []
    seg = getattr(result, "seg", None)
    if seg is not None:
        rows.extend(_per_class_block(seg, names, "mask"))
    box = getattr(result, "box", None)
    if box is not None:
        rows.extend(_per_class_block(box, names, "box"))
    return rows
