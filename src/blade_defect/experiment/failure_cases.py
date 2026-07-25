"""汇总各实验的逐样本预测，生成失败案例 CSV。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from blade_defect.utils.paths import resolve_path

CASE_FIELDS = [
    "image_path",
    "split",
    "true_class",
    "predicted_class",
    "confidence",
    "iou",
    "error_type",
    "experiment_id",
]


def _collect_cases(predictions_path: Path) -> list[dict[str, Any]]:
    with predictions_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    experiment_id = data.get("experiment_id", predictions_path.parent.name)
    cases: list[dict[str, Any]] = []
    for sample in data.get("samples", []):
        true_ids = set(sample.get("true_classes", []))
        pred_ids = set(sample.get("predicted_classes", []))
        matched = true_ids & pred_ids
        fn_ids = true_ids - pred_ids
        fp_ids = pred_ids - true_ids

        for cls_id in fn_ids:
            cases.append({
                "image_path": sample["image_path"],
                "split": sample.get("split", "val"),
                "true_class": cls_id,
                "predicted_class": "",
                "confidence": "",
                "iou": "",
                "error_type": "FN",
                "experiment_id": experiment_id,
            })
        for cls_id in fp_ids:
            conf_values = [
                p["confidence"]
                for p in sample.get("predictions", [])
                if p["class_id"] == cls_id
            ]
            cases.append({
                "image_path": sample["image_path"],
                "split": sample.get("split", "val"),
                "true_class": "",
                "predicted_class": cls_id,
                "confidence": round(max(conf_values), 4) if conf_values else "",
                "iou": "",
                "error_type": "FP",
                "experiment_id": experiment_id,
            })
        for cls_id in matched:
            conf_values = [
                p["confidence"]
                for p in sample.get("predictions", [])
                if p["class_id"] == cls_id
            ]
            cases.append({
                "image_path": sample["image_path"],
                "split": sample.get("split", "val"),
                "true_class": cls_id,
                "predicted_class": cls_id,
                "confidence": round(max(conf_values), 4) if conf_values else "",
                "iou": "",
                "error_type": "matched",
                "experiment_id": experiment_id,
            })
    return cases


def export_failure_cases(
    runs_dir: str | Path = "runs",
    output: str | Path = "results/failure_cases/cases.csv",
    error_only: bool = False,
) -> Path:
    root = resolve_path(runs_dir)
    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_cases: list[dict[str, Any]] = []
    for predictions_path in sorted(root.glob("*/validation_predictions.json")):
        all_cases.extend(_collect_cases(predictions_path))
    if error_only:
        all_cases = [c for c in all_cases if c["error_type"] != "matched"]
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CASE_FIELDS)
        writer.writeheader()
        writer.writerows(all_cases)
    return output_path
