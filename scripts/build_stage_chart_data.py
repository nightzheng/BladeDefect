"""构建阶段正式图表的数据源 CSV（冻结 v3 正式实验，只读复算）。

为 `scripts/generate_v3_charts.py` 准备四张正式图表的数值源，全部来自冻结的
v3 正式实验产物，不使用旧泄漏 baseline 或历史抽样数据：

- ``training_curves.csv``：三项正式实验逐 epoch 训练/验证指标
  （来源 runs/*/results.csv）；
- ``pr_curve.csv`` / ``f1_curve.csv``：15 类 seg e100 与 15 类 OBB e50 在
  共有 Box 口径（GT/预测统一取轴对齐外接框、类别感知匹配、IoU≥0.5）下的
  置信度扫描 Precision/Recall/F1（来源 validation_predictions.json 与
  obb_val_predictions.json，导出阈值 conf≥0.25，曲线只覆盖 conf≥0.25）；
- ``confidence_reliability.csv``：同一匹配规则下的置信度可靠性分箱
  （10 等宽箱，含 count/mean_confidence/accuracy 与 ECE）。

本脚本不读取正式 test、不启动训练，只读取已冻结的预测导出与训练日志。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from blade_defect.experiment.metadata import load_json, sha256_file
from blade_defect.utils.paths import resolve_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_RUNS: list[dict[str, Any]] = [
    {"experiment_id": "v3_yolo11s_seg_960_e100", "task": "segment"},
    {"experiment_id": "v3_hier_coarse_yolo11s_seg_960_e50", "task": "segment"},
    {"experiment_id": "v3_yolo11s_obb_960_e50", "task": "obb"},
]

TRAINING_METRICS = [
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "train/seg_loss",
    "train/angle_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "metrics/precision(M)",
    "metrics/recall(M)",
    "metrics/mAP50(M)",
    "metrics/mAP50-95(M)",
    "val/box_loss",
    "val/cls_loss",
]

CURVE_EXPERIMENTS: list[dict[str, Any]] = [
    {
        "experiment_id": "v3_yolo11s_seg_960_e100",
        "predictions": PROJECT_ROOT / "runs/v3_yolo11s_seg_960_e100/validation_predictions.json",
        "gt_kind": "segment_polygon",
    },
    {
        "experiment_id": "v3_yolo11s_obb_960_e50",
        "predictions": PROJECT_ROOT / "results/obb_v3_val_predictions/obb_val_predictions.json",
        "gt_kind": "obb_polygon",
    },
]

IOU_THRESHOLD = 0.5
RELIABILITY_BINS = 10


def polygon_envelope(polygon: list[float], width: float, height: float) -> tuple[float, float, float, float]:
    """归一化多边形（可含 OBB 四点）转像素轴对齐外接框。"""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return (min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """轴对齐框 IoU。"""
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_common_box_samples(
    predictions_path: Path,
) -> tuple[list[dict[str, Any]], int]:
    """把预测导出 JSON 统一成共有 Box 口径的 (GT框, 预测框) 列表。

    返回 (samples, export_conf_count)：每个样本含 gt=[(class_id, xyxy)] 与
    preds=[(class_id, conf, xyxy)]；坐标均为像素轴对齐外接框。
    """
    payload = load_json(predictions_path)
    samples: list[dict[str, Any]] = []
    for sample in payload.get("samples", []):
        width = float(sample.get("image_width") or 0)
        height = float(sample.get("image_height") or 0)
        gt = [
            (int(item["class_id"]), polygon_envelope([float(v) for v in item["polygon"]], width, height))
            for item in sample.get("ground_truth", [])
            if width > 0 and height > 0
        ]
        preds = [
            (int(item["class_id"]), float(item["confidence"]), tuple(float(v) for v in item["bbox"]))
            for item in sample.get("predictions", [])
        ]
        samples.append({"gt": gt, "preds": preds})
    return samples, len(payload.get("samples", []))


def match_predictions(
    samples: list[dict[str, Any]],
    conf_threshold: float,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[int, int, int]:
    """类别感知贪心匹配：按置信度降序，与同图同类未匹配 GT 取 IoU≥阈值者配对。

    返回 (TP, FP, FN)。
    """
    tp = fp = fn = 0
    for sample in samples:
        gt = sample["gt"]
        preds = [p for p in sample["preds"] if p[1] >= conf_threshold]
        preds.sort(key=lambda item: item[1], reverse=True)
        used = [False] * len(gt)
        for cls_id, _conf, box in preds:
            best_index = -1
            best_iou = iou_threshold
            for index, (gt_cls, gt_box) in enumerate(gt):
                if used[index] or gt_cls != cls_id:
                    continue
                value = box_iou(box, gt_box)
                if value >= best_iou:
                    best_iou = value
                    best_index = index
            if best_index >= 0:
                used[best_index] = True
                tp += 1
            else:
                fp += 1
        fn += sum(1 for flag in used if not flag)
    return tp, fp, fn


def sweep_pr_f1(
    samples: list[dict[str, Any]],
    gt_total: int,
) -> list[dict[str, Any]]:
    """按导出预测的全部置信度取值扫描 P/R/F1（conf≥导出阈值的部分）。"""
    thresholds = sorted({p[1] for sample in samples for p in sample["preds"]}, reverse=True)
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        tp, fp, fn = match_predictions(samples, threshold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / gt_total if gt_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "confidence_threshold": round(threshold, 4),
                "precision": round(precision, 6),
                "recall": round(recall, 6),
                "f1": round(f1, 6),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )
    return rows


def reliability_bins(
    samples: list[dict[str, Any]],
    conf_min: float = 0.25,
    bins: int = RELIABILITY_BINS,
) -> tuple[list[dict[str, Any]], float]:
    """置信度可靠性分箱：全量导出预测按 conf 等宽分箱，accuracy 为 IoU≥0.5 匹配率。

    返回 (rows, ECE)。
    """
    width = (1.0 - conf_min) / bins
    bucket_preds: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for sample in samples:
        gt = sample["gt"]
        preds = sorted(sample["preds"], key=lambda item: item[1], reverse=True)
        used = [False] * len(gt)
        for cls_id, conf, box in preds:
            best_index = -1
            best_iou = IOU_THRESHOLD
            for index, (gt_cls, gt_box) in enumerate(gt):
                if used[index] or gt_cls != cls_id:
                    continue
                value = box_iou(box, gt_box)
                if value >= best_iou:
                    best_iou = value
                    best_index = index
            matched = best_index >= 0
            if matched:
                used[best_index] = True
            bucket = min(bins - 1, int((conf - conf_min) / width))
            bucket_preds[bucket].append((conf, matched))
    rows: list[dict[str, Any]] = []
    total = sum(len(bucket) for bucket in bucket_preds)
    ece = 0.0
    for index, bucket in enumerate(bucket_preds):
        count = len(bucket)
        mean_conf = sum(conf for conf, _ in bucket) / count if count else 0.0
        accuracy = sum(1 for _, matched in bucket if matched) / count if count else 0.0
        if total:
            ece += (count / total) * abs(accuracy - mean_conf)
        rows.append(
            {
                "bin_index": index,
                "bin_low": round(conf_min + index * width, 4),
                "bin_high": round(conf_min + (index + 1) * width, 4),
                "count": count,
                "mean_confidence": round(mean_conf, 6),
                "accuracy": round(accuracy, 6),
            }
        )
    return rows, round(ece, 6)


def build_training_curves(output_dir: Path) -> tuple[Path, int]:
    rows: list[dict[str, Any]] = []
    for run in TRAINING_RUNS:
        results_csv = PROJECT_ROOT / "runs" / run["experiment_id"] / "results.csv"
        with results_csv.open(encoding="utf-8-sig", newline="") as file:
            for record in csv.DictReader(file):
                epoch = int(float(record["epoch"]))
                for metric in TRAINING_METRICS:
                    value = record.get(metric)
                    if value is None or str(value).strip() == "":
                        continue
                    rows.append(
                        {
                            "experiment_id": run["experiment_id"],
                            "task": run["task"],
                            "epoch": epoch,
                            "metric": metric,
                            "value": round(float(value), 8),
                            "source": results_csv.relative_to(PROJECT_ROOT).as_posix(),
                        }
                    )
    path = output_dir / "training_curves.csv"
    _write_csv(path, rows, ["experiment_id", "task", "epoch", "metric", "value", "source"])
    return path, len(rows)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "results/stage_20260830/figures/data"),
        help="图表数据源 CSV 输出目录",
    )
    args = parser.parse_args()
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"files": {}, "experiments": {}}
    training_path, training_rows = build_training_curves(output_dir)
    summary["files"]["training_curves.csv"] = {
        "rows": training_rows,
        "sha256": sha256_file(training_path),
    }

    pr_rows: list[dict[str, Any]] = []
    f1_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    for experiment in CURVE_EXPERIMENTS:
        experiment_id = experiment["experiment_id"]
        samples, num_images = load_common_box_samples(experiment["predictions"])
        gt_total = sum(len(sample["gt"]) for sample in samples)
        pred_total = sum(len(sample["preds"]) for sample in samples)
        sweep = sweep_pr_f1(samples, gt_total)
        bins, ece = reliability_bins(samples)
        source_rel = experiment["predictions"].relative_to(PROJECT_ROOT).as_posix()
        for row in sweep:
            pr_rows.append(
                {
                    "experiment_id": experiment_id,
                    "confidence_threshold": row["confidence_threshold"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "tp": row["tp"],
                    "fp": row["fp"],
                    "fn": row["fn"],
                    "source": source_rel,
                }
            )
            f1_rows.append(
                {
                    "experiment_id": experiment_id,
                    "confidence_threshold": row["confidence_threshold"],
                    "f1": row["f1"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "source": source_rel,
                }
            )
        for row in bins:
            reliability_rows.append(
                {
                    "experiment_id": experiment_id,
                    **row,
                    "ece": ece,
                    "source": source_rel,
                }
            )
        summary["experiments"][experiment_id] = {
            "val_images": num_images,
            "gt_instances": gt_total,
            "exported_predictions": pred_total,
            "ece": ece,
            "source": source_rel,
            "source_sha256": sha256_file(experiment["predictions"]),
        }

    pr_path = output_dir / "pr_curve.csv"
    _write_csv(
        pr_path,
        pr_rows,
        ["experiment_id", "confidence_threshold", "precision", "recall", "tp", "fp", "fn", "source"],
    )
    f1_path = output_dir / "f1_curve.csv"
    _write_csv(
        f1_path,
        f1_rows,
        ["experiment_id", "confidence_threshold", "f1", "precision", "recall", "source"],
    )
    reliability_path = output_dir / "confidence_reliability.csv"
    _write_csv(
        reliability_path,
        reliability_rows,
        [
            "experiment_id",
            "bin_index",
            "bin_low",
            "bin_high",
            "count",
            "mean_confidence",
            "accuracy",
            "ece",
            "source",
        ],
    )
    for name, path in (
        ("pr_curve.csv", pr_path),
        ("f1_curve.csv", f1_path),
        ("confidence_reliability.csv", reliability_path),
    ):
        summary["files"][name] = {"rows": sum(1 for _ in path.open(encoding="utf-8-sig")) - 1, "sha256": sha256_file(path)}

    summary["matching_rule"] = {
        "scope": "common_box",
        "iou_threshold": IOU_THRESHOLD,
        "matching": "class_aware_greedy_by_confidence",
        "coverage": "conf>=0.25（与冻结预测导出阈值一致）",
        "note": "GT/预测统一取轴对齐外接框；不比较Mask口径；仅正式val；test未读取",
    }
    summary_path = output_dir / "chart_data_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
