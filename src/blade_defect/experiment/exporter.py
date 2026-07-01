"""将实验指标导出为紧凑的对比表。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
from blade_defect.utils.paths import resolve_path

SUMMARY_FIELDS = ["experiment name", "model", "mAP50", "mAP50-95", "precision", "recall", "fps"]


def _value(metrics: dict[str, Any], *names: str) -> Any:
    return next((metrics[name] for name in names if name in metrics), "")


def export_summary(runs_dir: str | Path = "runs",
                   output: str | Path = "results/summary.csv") -> Path:
    """汇总成功实验的 metrics.json，生成字段稳定的对比 CSV。"""
    root, output_path = resolve_path(runs_dir), resolve_path(output)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        if metrics.get("status", "ok") != "ok":
            continue
        rows.append({"experiment name": _value(metrics, "name", "experiment name"),
                     "model": _value(metrics, "model"), "mAP50": _value(metrics, "mAP50", "map50"),
                     "mAP50-95": _value(metrics, "mAP50-95", "map50_95"),
                     "precision": _value(metrics, "precision"), "recall": _value(metrics, "recall"),
                     "fps": _value(metrics, "fps")})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output_path
