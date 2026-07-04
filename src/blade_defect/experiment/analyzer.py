"""为 baseline 实验结果生成论文级图表。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from blade_defect.utils.paths import resolve_path

_FONT_FALLBACKS = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei",
    "Arial Unicode MS", "DejaVu Sans",
]
_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5", "#5B9BD5", "#FFC000"]


def _configure_matplotlib() -> None:
    """配置无界面渲染和跨平台中文字体回退。"""
    plt.rcParams.update({
        "font.sans-serif": _FONT_FALLBACKS,
        "axes.unicode_minus": False,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _read_summary(path: Path) -> list[dict[str, Any]]:
    """读取对比表，并在输入不可用时尽早报错。"""
    if not path.is_file():
        raise FileNotFoundError(f"Experiment summary not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Experiment summary is empty: {path}")
    return rows


def _models(rows: Iterable[dict[str, Any]]) -> list[str]:
    # 实验名包含输入尺寸，优先使用它可区分同一模型的 640/960 等多组结果。
    return [str(row.get("experiment name") or row.get("model") or "unknown") for row in rows]


def _series_colors(count: int) -> list[str]:
    """按需循环调色板，保证颜色数量与实验数量严格一致。"""
    return [_COLORS[index % len(_COLORS)] for index in range(count)]


def _bar(ax: Any, labels: list[str], values: list[float], title: str, ylabel: str) -> None:
    bars = ax.bar(labels, values, color=_series_colors(len(labels)))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        if value == value:
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)


def _save_summary_plot(rows: list[dict[str, Any]], output: Path) -> None:
    """生成可作为论文主图的四面板总览图。"""
    labels = _models(rows)
    map50 = [_float(row.get("mAP50")) for row in rows]
    map95 = [_float(row.get("mAP50-95")) for row in rows]
    fps = [_float(row.get("fps")) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    _bar(axes[0, 0], labels, map50, "mAP50 vs Model", "mAP50")
    _bar(axes[0, 1], labels, map95, "mAP50-95 vs Model", "mAP50-95")
    _bar(axes[1, 0], labels, fps, "FPS vs Model", "FPS")
    axes[1, 1].scatter(
        fps,
        map95,
        s=90,
        c=_series_colors(len(labels)),
        edgecolors="white",
    )
    for x, y, label in zip(fps, map95, labels):
        axes[1, 1].annotate(label, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axes[1, 1].set(title="FPS–Accuracy Trade-off", xlabel="FPS", ylabel="mAP50-95")
    axes[1, 1].grid(alpha=0.25)
    fig.savefig(output)
    plt.close(fig)


def _save_pr_plot(rows: list[dict[str, Any]], output: Path) -> None:
    labels = _models(rows)
    precision = [_float(row.get("precision")) for row in rows]
    recall = [_float(row.get("recall")) for row in rows]
    positions = list(range(len(labels)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.bar([x - width / 2 for x in positions], precision, width, label="Precision", color=_COLORS[0])
    ax.bar([x + width / 2 for x in positions], recall, width, label="Recall", color=_COLORS[1])
    ax.set(title="Precision / Recall Comparison", ylabel="Score", xticks=positions,
           xticklabels=labels)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, max(1.0, *(value for value in precision + recall if value == value)))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output)
    plt.close(fig)


def _numeric_series(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [_float(item) for item in value]


def _curves_from_metrics(path: Path) -> dict[str, list[float]]:
    """从 JSON 顶层或 ``curves`` 字段读取可选曲线数组。"""
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    curves = payload.get("curves", payload)
    aliases = {
        "train_loss": ("train_loss", "loss/train", "train/loss"),
        "val_loss": ("val_loss", "loss/val", "val/loss"),
        "map50": ("map50_curve", "mAP50_curve", "map50"),
        "map50_95": ("map50_95_curve", "mAP50-95_curve", "map50_95"),
    }
    result: dict[str, list[float]] = {}
    if isinstance(curves, dict):
        for target, names in aliases.items():
            for name in names:
                values = _numeric_series(curves.get(name))
                if values:
                    result[target] = values
                    break
    return result


def _curves_from_results_csv(path: Path) -> dict[str, list[float]]:
    """将 Ultralytics results.csv 列名归一化为分析器曲线名。"""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [{key.strip(): value for key, value in row.items()} for row in csv.DictReader(file)]
    if not rows:
        return {}

    # 不同 Ultralytics 版本的列名会变化，但 train/val 前缀和 loss 后缀较稳定；
    # 将各任务 loss 求和，得到实验间可比较的聚合曲线。
    def columns(containing: tuple[str, ...]) -> list[str]:
        return [key for key in rows[0] if all(token in key.lower() for token in containing)]

    def summed(keys: list[str]) -> list[float]:
        return [sum(_float(row.get(key)) for key in keys) for row in rows] if keys else []

    map50_keys = [key for key in rows[0] if "map50" in key.lower() and "map50-95" not in key.lower()]
    map95_keys = [key for key in rows[0] if "map50-95" in key.lower()]
    return {key: values for key, values in {
        "train_loss": summed(columns(("train/", "loss"))),
        "val_loss": summed(columns(("val/", "loss"))),
        "map50": [_float(row.get(map50_keys[0])) for row in rows] if map50_keys else [],
        "map50_95": [_float(row.get(map95_keys[0])) for row in rows] if map95_keys else [],
    }.items() if values}


def _load_run_data(runs_dir: Path) -> tuple[dict[str, dict[str, list[float]]], dict[str, float]]:
    """从全部实验目录收集可选曲线和类别计数。"""
    curves: dict[str, dict[str, list[float]]] = {}
    distribution: dict[str, float] = {}
    if not runs_dir.exists():
        return curves, distribution
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "metrics.json"
        run_curves: dict[str, list[float]] = {}
        if metrics_path.is_file():
            run_curves = _curves_from_metrics(metrics_path)
            with metrics_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
            raw_distribution = payload.get("class_distribution", {})
            if isinstance(raw_distribution, dict):
                for name, count in raw_distribution.items():
                    distribution[str(name)] = distribution.get(str(name), 0.0) + _float(count)
        results_csv = run_dir / "results.csv"
        if results_csv.is_file():
            # 显式 JSON 曲线优先于自动识别的 CSV 曲线。
            run_curves = {**_curves_from_results_csv(results_csv), **run_curves}
        if run_curves:
            curves[run_dir.name] = run_curves
    return curves, distribution


def _save_curve_plot(curves: dict[str, dict[str, list[float]]], output: Path) -> None:
    """绘制 loss 和 mAP 历史；数据缺失时保留说明占位图。"""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for index, (name, values) in enumerate(curves.items()):
        color = _COLORS[index % len(_COLORS)]
        for key, linestyle in (("train_loss", "-"), ("val_loss", "--")):
            if values.get(key):
                axes[0].plot(range(1, len(values[key]) + 1), values[key], linestyle,
                             color=color, label=f"{name} {key}")
        for key, linestyle in (("map50", "-"), ("map50_95", "--")):
            if values.get(key):
                axes[1].plot(range(1, len(values[key]) + 1), values[key], linestyle,
                             color=color, label=f"{name} {key}")
    for ax, title, ylabel in ((axes[0], "Train / Validation Loss", "Loss"),
                              (axes[1], "mAP Curves", "mAP")):
        ax.set(title=title, xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.25)
        if ax.lines:
            ax.legend(fontsize=7)
        else:
            # 兼容只有最终标量指标的旧实验，同时保持输出文件约定。
            ax.text(0.5, 0.5, "No curve data available", ha="center", va="center",
                    transform=ax.transAxes)
    fig.savefig(output)
    plt.close(fig)


def _save_class_distribution(distribution: dict[str, float], output: Path) -> None:
    labels, values = list(distribution), list(distribution.values())
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    _bar(ax, labels, values, "Class Distribution", "Samples")
    fig.savefig(output)
    plt.close(fig)


def analyze_experiments(
    summary: str | Path = "results/summary.csv", runs_dir: str | Path = "runs",
    output_dir: str | Path = "results/analysis",
) -> list[Path]:
    """生成模型对比、PR、训练曲线和可选类别分布图。"""
    _configure_matplotlib()
    rows = _read_summary(resolve_path(summary))
    runs_path, output = resolve_path(runs_dir), resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_plot = output / "summary_plot.png"
    pr_plot = output / "pr_curve.png"
    loss_plot = output / "loss_curve_comparison.png"
    _save_summary_plot(rows, summary_plot)
    _save_pr_plot(rows, pr_plot)
    curves, distribution = _load_run_data(runs_path)
    _save_curve_plot(curves, loss_plot)
    generated = [summary_plot, pr_plot, loss_plot]
    if distribution:
        class_plot = output / "class_distribution.png"
        _save_class_distribution(distribution, class_plot)
        generated.append(class_plot)
    return generated


__all__ = ["analyze_experiments"]
