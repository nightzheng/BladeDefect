"""汇总各实验的逐样本预测，生成失败案例 CSV。支持三层错误层级与深度复核。"""
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
    "experiment_id", "dataset_id",
]

SCENARIO_FIELDS = [
    "scenario_overexposure", "scenario_shadow", "scenario_low_contrast",
    "scenario_small_object", "scenario_boundary_incomplete",
    "scenario_multiple_defects", "scenario_rare_class",
    "evidence_path",
]

DEEP_REVIEW_FIELDS = CASE_FIELDS + SCENARIO_FIELDS

VALID_ERROR_TYPES: set[str] = {
    # L1: localization / recall failures
    "localization_miss", "localization_low_confidence", "localization_false_positive",
    # L2: classification errors (matched pairs only)
    "coarse_group_error", "within_group_confusion",
    # L3: mask quality
    "mask_quality_error", "mask_boundary_error",
    # legacy auto-candidates
    "missing_detection", "false_positive", "low_confidence",
    # scenario conditions (separate from error_type)
    "small_object_miss", "class_confusion",
    "background_false_positive", "blade_edge_false_positive",
    "overexposure", "shadow", "blur",
    "multiple_defects", "possible_label_error",
}


def _resolve_coarse(class_id_str: str) -> str:
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
        raise ValueError(f"unknown error_type: {error_type!r}")
    confidence = case.get("confidence")
    if confidence is not None and confidence != "":
        try:
            val = float(confidence)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"confidence {val} outside [0, 1]")
        except (TypeError, ValueError) as exc:
            if "outside" in str(exc):
                raise
            raise ValueError(f"invalid confidence value: {confidence!r}") from exc


def _collect_cases(predictions_path: Path) -> list[dict[str, Any]]:
    with predictions_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    experiment_id = data.get("experiment_id", predictions_path.parent.name)
    dataset_id = data.get("dataset_id", "")
    cases: list[dict[str, Any]] = []
    for sample in data.get("samples", []):
        true_ids = set(sample.get("true_classes", []))
        pred_ids = set(sample.get("predicted_classes", []))
        matched = true_ids & pred_ids
        fn_ids = true_ids - pred_ids
        fp_ids = pred_ids - true_ids
        for cls_id in fn_ids:
            cases.append({"image_path": sample["image_path"], "split": sample.get("split","val"),
                "true_class": str(cls_id), "predicted_class": "",
                "coarse_true_class": _resolve_coarse(str(cls_id)), "coarse_predicted_class": "",
                "confidence": "", "iou": "", "candidate_type": "auto_fn", "error_type": "",
                "review_status": "pending", "review_note": "",
                "experiment_id": experiment_id, "dataset_id": dataset_id})
        for cls_id in fp_ids:
            conf_values = [p["confidence"] for p in sample.get("predictions",[]) if p["class_id"] == cls_id]
            cases.append({"image_path": sample["image_path"], "split": sample.get("split","val"),
                "true_class": "", "predicted_class": str(cls_id),
                "coarse_true_class": "", "coarse_predicted_class": _resolve_coarse(str(cls_id)),
                "confidence": round(max(conf_values),4) if conf_values else "", "iou": "",
                "candidate_type": "auto_fp", "error_type": "", "review_status": "pending", "review_note": "",
                "experiment_id": experiment_id, "dataset_id": dataset_id})
        for cls_id in matched:
            conf_values = [p["confidence"] for p in sample.get("predictions",[]) if p["class_id"] == cls_id]
            cases.append({"image_path": sample["image_path"], "split": sample.get("split","val"),
                "true_class": str(cls_id), "predicted_class": str(cls_id),
                "coarse_true_class": _resolve_coarse(str(cls_id)), "coarse_predicted_class": _resolve_coarse(str(cls_id)),
                "confidence": round(max(conf_values),4) if conf_values else "", "iou": "",
                "candidate_type": "auto_matched", "error_type": "", "review_status": "pending", "review_note": "",
                "experiment_id": experiment_id, "dataset_id": dataset_id})
    return cases


def export_failure_cases(
    cases_or_runs_dir: list[dict[str, Any]] | str | Path,
    output: str | Path,
    error_only: bool = False,
) -> Path:
    if isinstance(cases_or_runs_dir, (str, Path)):
        root = resolve_path(cases_or_runs_dir)
        output_path = resolve_path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_cases: list[dict[str, Any]] = []
        for predictions_path in sorted(root.glob("*/validation_predictions*.json")):
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
            writer.writerow({field: case.get(field, "") for field in CASE_FIELDS})
    return output_path


def export_failure_cases_csv(
    source: str | Path,
    output: str | Path = "results/failure_cases/cases.csv",
) -> Path:
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
        fields = CASE_FIELDS if all(f in rows[0] for f in CASE_FIELDS) else list(rows[0].keys())
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for case in rows:
            writer.writerow({field: case.get(field, "") for field in fields})
    return output_path


def export_failure_cases_deep_review(
    source: str | Path = "results/failure_cases/cases.csv",
    output: str | Path = "results/failure_cases/reviewed_cases.csv",
    experiment_id: str = "full_primary_yolo11s_seg_960",
    top_n: int = 60,
) -> Path:
    """Auto-tag deep review: scenario labels, evidence paths, three-layer error hierarchy."""
    source_path = resolve_path(source)
    output_path = resolve_path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with source_path.open("r", encoding="utf-8-sig") as f:
        all_rows = list(csv.DictReader(f))
    reviewed = []
    for row in all_rows:
        if len(reviewed) >= top_n:
            break
        if row.get("experiment_id", "") != experiment_id:
            continue
        ct = row.get("candidate_type", "")
        n_miss = sum(1 for r in reviewed if r.get("error_type") == "localization_miss")
        n_fp = sum(1 for r in reviewed if r.get("error_type") == "localization_false_positive")
        n_low = sum(1 for r in reviewed if r.get("error_type") == "localization_low_confidence")
        if ct == "auto_fn" and n_miss < max(top_n // 3, 20):
            row["error_type"] = "localization_miss"
        elif ct == "auto_fp" and n_fp < min(top_n // 3, 20):
            row["error_type"] = "localization_false_positive"
        elif ct == "auto_matched" and n_low < top_n - n_miss - n_fp:
            if row.get("confidence") and float(row["confidence"]) < 0.3:
                row["error_type"] = "localization_low_confidence"
            else:
                continue
        else:
            continue
        row["review_status"] = "deep_reviewed"
        cid = row.get("true_class", "")
        row["scenario_rare_class"] = "1" if cid and int(cid) in [8, 13, 14] else "0"
        if row.get("candidate_type") == "auto_matched" and row.get("confidence") and float(row["confidence"]) < 0.3:
            row["scenario_small_object"] = "1"; row["scenario_low_contrast"] = "1"
        else:
            row["scenario_small_object"] = "0"; row["scenario_low_contrast"] = "0"
        for fld in ["scenario_overexposure", "scenario_shadow", "scenario_boundary_incomplete", "scenario_multiple_defects"]:
            row[fld] = "?"
        row["evidence_path"] = f"runs/runs/{experiment_id}/validation_predictions.json"
        reviewed.append(row)
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=DEEP_REVIEW_FIELDS)
        writer.writeheader()
        for row in reviewed:
            writer.writerow({field: row.get(field, "") for field in DEEP_REVIEW_FIELDS})
    return output_path


__all__ = ["CASE_FIELDS", "SCENARIO_FIELDS", "DEEP_REVIEW_FIELDS", "export_failure_cases", "export_failure_cases_csv", "export_failure_cases_deep_review"]
