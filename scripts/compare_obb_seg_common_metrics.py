"""比较 OBB 与 15 类 seg 正式实验的共有 Box 指标（不比较 Mask mAP）。

口径约束：
- 只比较共有 Box 指标（Precision/Recall/mAP50/mAP50-95）、逐类 AP 与 FPS；
- 明确不比较 Mask mAP（OBB 无 Mask 分支）；
- 自动 OBB 来自 polygon 最小外接矩形，不等同于人工旋转框真值；
- OBB 侧结果未完成时只输出状态说明，不用 smoke/部分结果冒充正式结果。

用法：
    python scripts/compare_obb_seg_common_metrics.py
    python scripts/compare_obb_seg_common_metrics.py --obb-run runs/v3_yolo11s_obb_960_e50 --seg-run runs/v3_yolo11s_seg_960_e100
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMMON_BOX_METRICS = ("box_precision", "box_recall", "box_mAP50", "box_mAP50-95")
METRIC_LABELS = {
    "box_precision": "Box Precision",
    "box_recall": "Box Recall",
    "box_mAP50": "Box mAP50",
    "box_mAP50-95": "Box mAP50-95",
    "fps": "FPS",
}
OUTPUT_CSV = "obb_seg_common_metrics.csv"
OUTPUT_MD = "obb_seg_common_metrics.md"
PER_CLASS_CSV = "obb_seg_per_class_ap.csv"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_per_class_box(run_dir: Path) -> dict[int, dict[str, Any]]:
    path = run_dir / "per_class_metrics.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig") as handle:
        rows = [row for row in csv.DictReader(handle) if row["metric_branch"] == "box"]
    return {int(row["class_id"]): row for row in rows}


def load_run_metrics(run_dir: Path) -> tuple[dict[str, Any] | None, str]:
    """读取 run 的 metrics.json；不可用时返回原因，绝不编造数值。"""
    path = run_dir / "metrics.json"
    if not path.is_file():
        return None, f"metrics.json 不存在：{path}"
    metrics = _load_json(path)
    if metrics.get("status") != "ok":
        return None, f"metrics.status={metrics.get('status')}（实验未完成或失败）"
    return metrics, ""


def compare_runs(obb_run: Path, seg_run: Path, output_dir: Path) -> dict[str, Any]:
    """生成共有 Box 指标对比与逐类 AP 对比，返回结构化结果。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    obb_metrics, obb_reason = load_run_metrics(obb_run)
    seg_metrics, seg_reason = load_run_metrics(seg_run)
    payload: dict[str, Any] = {
        "obb_run": str(obb_run),
        "seg_run": str(seg_run),
        "scope": (
            "仅共有 Box 指标、逐类 AP 与速度；不比较 Mask mAP；"
            "自动 OBB 来自 polygon 最小外接矩形，不等同于人工旋转框真值"
        ),
        "obb_available": obb_metrics is not None,
        "seg_available": seg_metrics is not None,
    }
    if obb_metrics is None or seg_metrics is None:
        payload["status"] = "unavailable"
        payload["reason"] = "; ".join(
            reason for reason in (obb_reason, seg_reason) if reason
        )
        _write_outputs(output_dir, payload, [], [])
        return payload

    metric_rows: list[dict[str, Any]] = []
    for key in COMMON_BOX_METRICS + ("fps",):
        obb_value = float(obb_metrics.get(key) or 0.0)
        seg_value = float(seg_metrics.get(key) or 0.0)
        metric_rows.append(
            {
                "metric": key,
                "metric_label": METRIC_LABELS[key],
                "obb": obb_value,
                "seg": seg_value,
                "delta_obb_minus_seg": obb_value - seg_value,
            }
        )

    obb_per_class = _load_per_class_box(obb_run)
    seg_per_class = _load_per_class_box(seg_run)
    per_class_rows: list[dict[str, Any]] = []
    for class_id in sorted(set(obb_per_class) | set(seg_per_class)):
        obb_row = obb_per_class.get(class_id, {})
        seg_row = seg_per_class.get(class_id, {})

        def _float(row: dict[str, Any], key: str) -> float | None:
            value = row.get(key)
            return float(value) if value not in (None, "") else None

        obb_ap50 = _float(obb_row, "ap50")
        seg_ap50 = _float(seg_row, "ap50")
        obb_ap = _float(obb_row, "ap50_95")
        seg_ap = _float(seg_row, "ap50_95")
        per_class_rows.append(
            {
                "class_id": class_id,
                "class_name": obb_row.get("class_name") or seg_row.get("class_name"),
                "obb_ap50": obb_ap50,
                "seg_ap50": seg_ap50,
                "delta_ap50": (
                    obb_ap50 - seg_ap50 if obb_ap50 is not None and seg_ap50 is not None else None
                ),
                "obb_ap50_95": obb_ap,
                "seg_ap50_95": seg_ap,
                "delta_ap50_95": obb_ap - seg_ap if obb_ap is not None and seg_ap is not None else None,
            }
        )

    payload.update(
        {
            "status": "ok",
            "obb_experiment": {
                "experiment_id": obb_metrics.get("name"),
                "dataset_id": obb_metrics.get("dataset_id"),
                "epochs": obb_metrics.get("epochs"),
                "task": obb_metrics.get("task", "obb"),
            },
            "seg_experiment": {
                "experiment_id": seg_metrics.get("name"),
                "dataset_id": seg_metrics.get("dataset_id"),
                "epochs": seg_metrics.get("epochs"),
                "task": "seg",
            },
            "metrics": metric_rows,
            "per_class": per_class_rows,
        }
    )
    _write_outputs(output_dir, payload, metric_rows, per_class_rows)
    return payload


def _write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    metric_rows: list[dict[str, Any]],
    per_class_rows: list[dict[str, Any]],
) -> None:
    (output_dir / "obb_seg_comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / OUTPUT_CSV).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("metric", "metric_label", "obb", "seg", "delta_obb_minus_seg"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(metric_rows)
    with (output_dir / PER_CLASS_CSV).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "class_id", "class_name", "obb_ap50", "seg_ap50", "delta_ap50",
                "obb_ap50_95", "seg_ap50_95", "delta_ap50_95",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(per_class_rows)

    lines = [
        "# OBB vs seg 共有 Box 指标对比",
        "",
        f"- 口径：{payload['scope']}",
        f"- OBB run：`{payload['obb_run']}`",
        f"- seg run：`{payload['seg_run']}`",
    ]
    if payload["status"] != "ok":
        lines += ["", f"**对比不可用：{payload['reason']}**"]
    else:
        lines += [
            "",
            "## 共有 Box 指标（val 口径）",
            "",
            "| 指标 | OBB | seg | 差值（OBB-seg） |",
            "|---|---|---|---|",
        ]
        for row in metric_rows:
            lines.append(
                f"| {row['metric_label']} | {row['obb']:.4f} | {row['seg']:.4f} "
                f"| {row['delta_obb_minus_seg']:+.4f} |"
            )
        if per_class_rows:
            lines += [
                "",
                "## 逐类 AP（box 分支）",
                "",
                "| 类别 | 名称 | OBB AP50 | seg AP50 | 差值 | OBB AP50-95 | seg AP50-95 | 差值 |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for row in per_class_rows:
                def _fmt(value: float | None) -> str:
                    return f"{value:.4f}" if value is not None else "—"

                def _fmt_delta(value: float | None) -> str:
                    return f"{value:+.4f}" if value is not None else "—"

                lines.append(
                    f"| {row['class_id']} | {row['class_name']} "
                    f"| {_fmt(row['obb_ap50'])} | {_fmt(row['seg_ap50'])} | {_fmt_delta(row['delta_ap50'])} "
                    f"| {_fmt(row['obb_ap50_95'])} | {_fmt(row['seg_ap50_95'])} | {_fmt_delta(row['delta_ap50_95'])} |"
                )
    (output_dir / OUTPUT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--obb-run", type=Path, default=PROJECT_ROOT / "runs" / "v3_yolo11s_obb_960_e50"
    )
    parser.add_argument(
        "--seg-run", type=Path, default=PROJECT_ROOT / "runs" / "v3_yolo11s_seg_960_e100"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "obb_v3"
    )
    args = parser.parse_args()
    payload = compare_runs(args.obb_run, args.seg_run, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
