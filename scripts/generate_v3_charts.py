"""生成阶段四张正式图表（PR / F1 / 训练曲线 / 置信度可靠性）。

数据来源仅限 `results/stage_20260830/figures/data/` 下由
`scripts/build_stage_chart_data.py` 从冻结 v3 正式实验复算的 CSV；
不使用旧泄漏 baseline 或历史抽样数据。每张图的数据源 SHA-256、experiment_id、
坐标轴、图例与指标口径记录在同目录 `chart_manifest.json`，供复核 PNG 与 CSV
数值一致性。

用法：
    python scripts/generate_v3_charts.py
    python scripts/generate_v3_charts.py --data results/stage_20260830/figures/data --output results/stage_20260830/figures
"""
from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from blade_defect.experiment.metadata import collect_git_metadata, sha256_file
from blade_defect.utils.paths import resolve_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT_ROOT / "results/stage_20260830/figures/data"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/stage_20260830/figures"

EXPERIMENT_STYLES: dict[str, dict[str, str]] = {
    "v3_yolo11s_seg_960_e100": {"label": "seg e100 (15-class)", "color": "#1f77b4"},
    "v3_yolo11s_obb_960_e50": {"label": "OBB e50 (15-class)", "color": "#d62728"},
    "v3_hier_coarse_yolo11s_seg_960_e50": {"label": "seg e50 (6-class)", "color": "#2ca02c"},
}

COMMON_BOX_NOTE = (
    "Common-box scope: axis-aligned envelopes, class-aware greedy match, IoU>=0.5, "
    "conf>=0.25 (frozen export threshold); val only; test locked."
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _series(rows: list[dict[str, str]], experiment_id: str, x_key: str, y_key: str) -> tuple[list[float], list[float]]:
    selected = [row for row in rows if row["experiment_id"] == experiment_id]
    return ([float(row[x_key]) for row in selected], [float(row[y_key]) for row in selected])


def render_pr_curve(rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    points = {}
    for experiment_id in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50"):
        style = EXPERIMENT_STYLES[experiment_id]
        recall, precision = _series(rows, experiment_id, "recall", "precision")
        order = sorted(range(len(recall)), key=lambda i: recall[i])
        ax.plot(
            [recall[i] for i in order],
            [precision[i] for i in order],
            color=style["color"],
            linewidth=1.6,
            label=style["label"],
        )
        points[experiment_id] = len(recall)
    ax.set_xlabel("Recall (common-box, IoU>=0.5)")
    ax.set_ylabel("Precision (common-box, IoU>=0.5)")
    ax.set_title("Stage 20260830 PR curve - frozen v3 formal val")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.text(0.01, 0.01, COMMON_BOX_NOTE, fontsize=6.5, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output)
    plt.close(fig)
    return {
        "file": output.name,
        "x_axis": "Recall (common-box, IoU>=0.5)",
        "y_axis": "Precision (common-box, IoU>=0.5)",
        "legend": [EXPERIMENT_STYLES[key]["label"] for key in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50")],
        "plotted_points": points,
    }


def render_f1_curve(rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    points = {}
    best: dict[str, Any] = {}
    for experiment_id in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50"):
        style = EXPERIMENT_STYLES[experiment_id]
        conf, f1 = _series(rows, experiment_id, "confidence_threshold", "f1")
        pairs = sorted(zip(conf, f1))
        ax.plot(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            color=style["color"],
            linewidth=1.6,
            label=style["label"],
        )
        peak = max(pairs, key=lambda item: item[1])
        ax.scatter([peak[0]], [peak[1]], color=style["color"], s=22, zorder=5)
        ax.annotate(
            f"peak F1={peak[1]:.3f}@{peak[0]:.2f}",
            xy=(peak[0], peak[1]),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=7,
            color=style["color"],
        )
        points[experiment_id] = len(pairs)
        best[experiment_id] = {"f1": round(peak[1], 6), "confidence_threshold": peak[0]}
    ax.set_xlabel("Confidence threshold (export conf>=0.25)")
    ax.set_ylabel("F1 (common-box, IoU>=0.5)")
    ax.set_title("Stage 20260830 F1-confidence curve - frozen v3 formal val")
    ax.set_xlim(0.25, 1.0)
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.text(0.01, 0.01, COMMON_BOX_NOTE, fontsize=6.5, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output)
    plt.close(fig)
    return {
        "file": output.name,
        "x_axis": "Confidence threshold (export conf>=0.25)",
        "y_axis": "F1 (common-box, IoU>=0.5)",
        "legend": [EXPERIMENT_STYLES[key]["label"] for key in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50")],
        "plotted_points": points,
        "peak": best,
    }


def render_training_curves(rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    fig, (ax_map, ax_loss) = plt.subplots(1, 2, figsize=(13, 5.2), dpi=200)
    points: dict[str, Any] = {}
    for experiment_id, style in EXPERIMENT_STYLES.items():
        map_rows = [row for row in rows if row["experiment_id"] == experiment_id and row["metric"] == "metrics/mAP50(B)"]
        loss_rows = [row for row in rows if row["experiment_id"] == experiment_id and row["metric"] == "train/box_loss"]
        epochs_map = [int(row["epoch"]) for row in map_rows]
        epochs_loss = [int(row["epoch"]) for row in loss_rows]
        ax_map.plot(epochs_map, [float(row["value"]) for row in map_rows], color=style["color"], linewidth=1.4, label=style["label"])
        ax_loss.plot(epochs_loss, [float(row["value"]) for row in loss_rows], color=style["color"], linewidth=1.2, label=style["label"])
        points[experiment_id] = {"metrics/mAP50(B)": len(epochs_map), "train/box_loss": len(epochs_loss)}
    ax_map.set_xlabel("Epoch")
    ax_map.set_ylabel("val mAP50 (Box)")
    ax_map.set_title("Validation Box mAP50 per epoch")
    ax_map.grid(alpha=0.3)
    ax_map.legend(loc="lower right")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("train box_loss")
    ax_loss.set_title("Training box loss per epoch")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend(loc="upper right")
    fig.suptitle("Stage 20260830 training curves - frozen v3 formal runs (50/100-epoch budgets differ)")
    fig.text(0.01, 0.01, "Source: runs/*/results.csv of frozen v3 formal runs; no new training this week.", fontsize=6.5, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(output)
    plt.close(fig)
    return {
        "file": output.name,
        "x_axis": "Epoch",
        "y_axis": "left: val mAP50 (Box); right: train box_loss",
        "legend": [style["label"] for style in EXPERIMENT_STYLES.values()],
        "plotted_points": points,
    }


def render_confidence_reliability(rows: list[dict[str, str]], output: Path) -> dict[str, Any]:
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=200)
    ax.plot([0.25, 1.0], [0.25, 1.0], linestyle="--", color="#888888", linewidth=1.0, label="perfect calibration")
    points = {}
    eces = {}
    for experiment_id in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50"):
        style = EXPERIMENT_STYLES[experiment_id]
        selected = [row for row in rows if row["experiment_id"] == experiment_id and int(row["count"]) > 0]
        xs = [float(row["mean_confidence"]) for row in selected]
        ys = [float(row["accuracy"]) for row in selected]
        counts = [int(row["count"]) for row in selected]
        ece = float(next((row["ece"] for row in rows if row["experiment_id"] == experiment_id), 0.0))
        ax.plot(xs, ys, marker="o", markersize=4, color=style["color"], linewidth=1.6, label=f"{style['label']} (ECE={ece:.4f})")
        for x, y, count in zip(xs, ys, counts):
            if count:
                ax.annotate(str(count), xy=(x, y), xytext=(0, 5), textcoords="offset points", fontsize=6, color=style["color"], ha="center")
        points[experiment_id] = len(xs)
        eces[experiment_id] = ece
    ax.set_xlabel("Mean predicted confidence per bin (10 bins over [0.25, 1.0])")
    ax.set_ylabel("Empirical accuracy (common-box IoU>=0.5 match rate)")
    ax.set_title("Stage 20260830 confidence reliability - frozen v3 formal val")
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, COMMON_BOX_NOTE + " Bin annotations = prediction counts.", fontsize=6.5, color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output)
    plt.close(fig)
    return {
        "file": output.name,
        "x_axis": "Mean predicted confidence per bin (10 bins over [0.25, 1.0])",
        "y_axis": "Empirical accuracy (common-box IoU>=0.5 match rate)",
        "legend": ["perfect calibration"] + [
            f"{EXPERIMENT_STYLES[key]['label']} (ECE)" for key in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50")
        ],
        "plotted_points": points,
        "ece": eces,
    }


FIGURE_SOURCES: dict[str, str] = {
    "PR_curve.png": "pr_curve.csv",
    "F1_curve.csv": "f1_curve.csv",
    "training_curves.png": "training_curves.csv",
    "confidence_reliability.png": "confidence_reliability.csv",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="图表数据源 CSV 目录")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="PNG 输出目录")
    args = parser.parse_args()
    data_dir = resolve_path(args.data)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    figures: dict[str, Any] = {}
    figure_specs = [
        ("PR_curve.png", "pr_curve.csv", render_pr_curve),
        ("F1_curve.png", "f1_curve.csv", render_f1_curve),
        ("training_curves.png", "training_curves.csv", render_training_curves),
        ("confidence_reliability.png", "confidence_reliability.csv", render_confidence_reliability),
    ]
    for figure_name, csv_name, renderer in figure_specs:
        csv_path = data_dir / csv_name
        rows = _load_csv(csv_path)
        meta = renderer(rows, output_dir / figure_name)
        meta["source_csv"] = f"figures/data/{csv_name}"
        meta["source_csv_sha256"] = sha256_file(csv_path)
        meta["source_csv_rows"] = len(rows)
        figures[figure_name] = meta

    manifest = {
        "generated_by": "scripts/generate_v3_charts.py（张劲焱供图表规格，杨静怡在完整环境执行）",
        "figure_data_built_by": "scripts/build_stage_chart_data.py（仅冻结 v3 正式实验，只读复算）",
        "figures": figures,
        "metric_scope": {
            "common_box": "GT/预测统一取轴对齐外接框，类别感知贪心匹配，IoU>=0.5；conf>=0.25 与冻结导出阈值一致",
            "excluded": "不比较 Mask mAP；OBB 与 seg e100 训练预算不同（50 vs 100 轮）不做排名；未使用旧泄漏 baseline 或历史抽样数据",
            "splits": "仅正式 val；正式 test 全程锁定未读取",
        },
        "environment": {
            "python_version": platform.python_version(),
            "matplotlib_version": importlib.metadata.version("matplotlib"),
            "platform": platform.platform(),
        },
        "git": collect_git_metadata(PROJECT_ROOT),
    }
    manifest_path = output_dir / "chart_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": sorted(figures), "manifest": str(manifest_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
