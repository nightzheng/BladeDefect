"""汇总各实验的逐样本预测，生成失败案例 CSV。"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from blade_defect.data.defect_classes import DEFECT_GROUPS
from blade_defect.utils.paths import resolve_path

CASE_FIELDS = [
    "image_path", "split", "true_class", "predicted_class",
    "coarse_true_class", "coarse_predicted_class",
    "confidence", "iou",
    "candidate_type", "error_type", "review_status", "review_note",
    "experiment_id",
]

VALID_ERROR_TYPES: set[str] = {
    "small_object_miss", "low_confidence_miss", "class_confusion",
    "background_false_positive", "blade_edge_false_positive",
    "overexposure", "shadow", "blur",
    "mask_boundary_error", "multiple_defects", "possible_label_error",
}

_EXPERIMENT_ID_TO_NAME: dict[str, str] = {}


def _resolve_coarse(class_id_str: str) -> str:
    """将 15 类 ID 映射为 6 大类名称。"""
    try:
        class_id = int(class_id_str)
    except (ValueError, TypeError):
        return ""
    for group_name, ids in DEFECT_GROUPS.items():
        if class_id in ids:
            return group_name
    return ""


def _validate_case(case: dict[str, Any]) -> None:
    error_type = case.get("error_type", "")
    if error_type and error_type not in VALID_ERROR_TYPES:
        raise ValueError(
            f"unknown error_type: {error_type!r}"
        )
    confidence = case.get("confidence")
    if confidence is not None and confidence != "":
        try:
            val = float(confidence)
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"confidence {val} outside [0, 1]"
                )
        except (TypeError, ValueError) as exc:
            if "outside" in str(exc):
                raise
            raise ValueError(f"invalid confidence value: {confidence!r}") from exc


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
                "true_class": str(cls_id),
                "predicted_class": "",
                "coarse_true_class": _resolve_coarse(str(cls_id)),
                "coarse_predicted_class": "",
                "confidence": "",
                "iou": "",
                "candidate_type": "auto_fn",
                "error_type": "",
                "review_status": "pending",
                "review_note": "",
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
                "predicted_class": str(cls_id),
                "coarse_true_class": "",
                "coarse_predicted_class": _resolve_coarse(str(cls_id)),
                "confidence": round(max(conf_values), 4) if conf_values else "",
                "iou": "",
                "candidate_type": "auto_fp",
                "error_type": "",
                "review_status": "pending",
                "review_note": "",
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
                "true_class": str(cls_id),
                "predicted_class": str(cls_id),
                "coarse_true_class": _resolve_coarse(str(cls_id)),
                "coarse_predicted_class": _resolve_coarse(str(cls_id)),
                "confidence": round(max(conf_values), 4) if conf_values else "",
                "iou": "",
                "candidate_type": "auto_matched",
                "error_type": "",
                "review_status": "pending",
                "review_note": "",
                "experiment_id": experiment_id,
            })
    return cases


def export_failure_cases(
    cases_or_runs_dir: list[dict[str, Any]] | str | Path,
    output: str | Path,
    error_only: bool = False,
) -> Path:
    """
    导出失败案例 CSV。

    支持两种调用方式：
    1. export_failure_cases(list_of_dicts, output_path)
    2. export_failure_cases(runs_dir, output_path, error_only=...)
    """
    if isinstance(cases_or_runs_dir, (str, Path)):
        root = resolve_path(cases_or_runs_dir)
        output_path = resolve_path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_cases: list[dict[str, Any]] = []
        for predictions_path in sorted(root.glob("*/validation_predictions.json")):
            all_cases.extend(_collect_cases(predictions_path))
        if error_only:
            all_cases = [c for c in all_cases if c["error_type"] not in ("", "matched")]
    else:
        all_cases = list(cases_or_runs_dir)
        output_path = resolve_path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for case in all_cases:
            _validate_case(case)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CASE_FIELDS)
        writer.writeheader()
        for case in all_cases:
            row = {field: case.get(field, "") for field in CASE_FIELDS}
            writer.writerow(row)
    return output_path


def export_failure_cases_csv(
    source: str | Path,
    output: str | Path = "results/failure_cases/cases.csv",
) -> Path:
    """读取并校验准备好的失败候选 CSV，校验后输出标准化索引。"""
    source_path = resolve_path(source)
    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Failure candidate CSV is empty: {source_path}")
    for case in rows:
        _validate_case(case)
    with output_path.open("w", newline="", encoding="utf-8-sig") as file:
        if rows:
            fields = CASE_FIELDS if all(f in rows[0] for f in CASE_FIELDS) else list(rows[0].keys())
        else:
            fields = CASE_FIELDS
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for case in rows:
            row = {field: case.get(field, "") for field in fields}
            writer.writerow(row)
    return output_path


__all__ = ["export_failure_cases", "export_failure_cases_csv"]