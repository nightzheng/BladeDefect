"""为 baseline 实验结果生成论文级图表、对比 CSV 和层级分析资产。"""

from __future__ import annotations

import csv
import json
import shutil
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
_COLORS = ["#4472C4", "#ED7D31", "#70AD47", "#A5A5A5", "#5B9BD5", "#FFC000", "#264478", "#9B2D20"]


def _configure_matplotlib() -> None:
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
    if not path.is_file():
        raise FileNotFoundError(f"Experiment summary not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Experiment summary is empty: {path}")
    return rows


def _extract_imgsz(name: str) -> str:
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return str(float(parts[1]))
        except ValueError:
            pass
    return ""


def _models(rows: Iterable[dict[str, Any]]) -> list[str]:
    return [str(row.get("experiment name") or row.get("model") or "unknown") for row in rows]


def _imgsz_list(rows: Iterable[dict[str, Any]]) -> list[str]:
    result: list[str] = []
    for row in rows:
        imgsz = str(row.get("imgsz", "")).strip()
        if imgsz:
            result.append(str(float(imgsz)))
        else:
            name = str(row.get("experiment name", ""))
            result.append(_extract_imgsz(name))
    return result


def _series_colors(count: int) -> list[str]:
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


def _scatter(ax: Any, x: list[float], y: list[float], labels: list[str],
             title: str, xlabel: str, ylabel: str) -> None:
    colors = _series_colors(len(labels))
    ax.scatter(x, y, s=90, c=colors, edgecolors="white", zorder=3)
    for xi, yi, label in zip(x, y, labels):
        ax.annotate(label, (xi, yi), xytext=(5, 5),
                    textcoords="offset points", fontsize=7)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.25)


def _save_summary_plot(rows: list[dict[str, Any]], output: Path) -> None:
    labels = _models(rows)
    map50 = [_float(row.get("mAP50")) for row in rows]
    map95 = [_float(row.get("mAP50-95")) for row in rows]
    fps = [_float(row.get("fps")) for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    _bar(axes[0, 0], labels, map50, "mAP50 vs Model", "mAP50")
    _bar(axes[0, 1], labels, map95, "mAP50-95 vs Model", "mAP50-95")
    _bar(axes[1, 0], labels, fps, "FPS vs Model", "FPS")
    axes[1, 1].scatter(fps, map95, s=90, c=_series_colors(len(labels)),
                       edgecolors="white")
    for x, y, label in zip(fps, map95, labels):
        axes[1, 1].annotate(label, (x, y), xytext=(5, 5),
                            textcoords="offset points", fontsize=8)
    axes[1, 1].set(title="FPS-Accuracy Trade-off", xlabel="FPS", ylabel="mAP50-95")
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
    ax.bar([x - width / 2 for x in positions], precision, width,
           label="Precision", color=_COLORS[0])
    ax.bar([x + width / 2 for x in positions], recall, width,
           label="Recall", color=_COLORS[1])
    ax.set(title="Precision / Recall Comparison", ylabel="Score",
           xticks=positions, xticklabels=labels)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylim(0, max(1.0, *(v for v in precision + recall if v == v)))
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(output)
    plt.close(fig)


def _numeric_series(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []
    return [_float(item) for item in value]


def _curves_from_metrics(path: Path) -> dict[str, list[float]]:
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
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [{key.strip(): value for key, value in row.items()}
                for row in csv.DictReader(file)]
    if not rows:
        return {}

    def columns(containing: tuple[str, ...]) -> list[str]:
        return [key for key in rows[0]
                if all(token in key.lower() for token in containing)]

    def summed(keys: list[str]) -> list[float]:
        return [sum(_float(row.get(key)) for key in keys) for row in rows] if keys else []

    map50_keys = [key for key in rows[0]
                  if "map50" in key.lower() and "map50-95" not in key.lower()]
    map95_keys = [key for key in rows[0] if "map50-95" in key.lower()]
    return {k: v for k, v in {
        "train_loss": summed(columns(("train/", "loss"))),
        "val_loss": summed(columns(("val/", "loss"))),
        "map50": [_float(row.get(map50_keys[0])) for row in rows] if map50_keys else [],
        "map50_95": [_float(row.get(map95_keys[0])) for row in rows] if map95_keys else [],
    }.items() if v}


def _load_run_data(
    runs_dir: Path,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, float]]:
    curves: dict[str, dict[str, list[float]]] = {}
    distribution: dict[str, float] = {}
    if not runs_dir.exists():
        return curves, distribution
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
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
            run_curves = {**_curves_from_results_csv(results_csv), **run_curves}
        if run_curves:
            curves[run_dir.name] = run_curves
    return curves, distribution


def _save_curve_plot(curves: dict[str, dict[str, list[float]]], output: Path) -> None:
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


def _save_model_comparison_csv(
    rows: list[dict[str, Any]], output: Path,
) -> None:
    imgsz_vals = _imgsz_list(rows)
    fields = ["experiment name", "model", "imgsz", "f1",
              "mAP50", "mAP50-95", "precision", "recall", "fps"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, row in enumerate(rows):
            p = _float(row.get("precision"))
            r = _float(row.get("recall"))
            f1 = 2 * p * r / (p + r) if p + r else 0.0
            writer.writerow({
                "experiment name": row.get("experiment name", ""),
                "model": row.get("model", ""),
                "imgsz": imgsz_vals[index] if index < len(imgsz_vals) else "",
                "f1": f"{f1:.4f}",
                "mAP50": row.get("mAP50", ""),
                "mAP50-95": row.get("mAP50-95", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "fps": row.get("fps", ""),
            })


def _save_per_class_metrics_csv(
    runs_dir: Path, output: Path,
) -> None:
    collected: list[dict[str, Any]] = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        with metrics_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        experiment_id = payload.get("name", metrics_path.parent.name)
        per_class = (payload.get("per_class_metrics") or payload.get("per_class") or [])
        for entry in per_class:
            entry["experiment_id"] = experiment_id
            collected.append(entry)
    output.parent.mkdir(parents=True, exist_ok=True)
    if collected:
        fields = list(collected[0].keys())
        with output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(collected)
    else:
        with output.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(
                file, fieldnames=["class", "precision", "recall", "AP50", "AP50-95"],
            )
            writer.writeheader()


def _save_model_comparison_bar(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [str(row.get("experiment name") or row.get("model") or "?")
              for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    _bar(axes[0], labels, [_float(row.get("mAP50")) for row in rows], "mAP50 vs Model", "mAP50")
    _bar(axes[1], labels, [_float(row.get("mAP50-95")) for row in rows],
         "mAP50-95 vs Model", "mAP50-95")
    fig.savefig(output)
    plt.close(fig)


def _save_input_size_map(rows: list[dict[str, Any]], output: Path) -> None:
    imgsz_vals = _imgsz_list(rows)
    labels = _models(rows)
    xs = [_float(v) for v in imgsz_vals]
    ys = [_float(row.get("mAP50")) for row in rows]
    valid = [(x, y, label) for x, y, label in zip(xs, ys, labels) if x == x and y == y]
    if not valid:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(output)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _scatter(ax, [v[0] for v in valid], [v[1] for v in valid], [v[2] for v in valid],
             "Input Size vs mAP50", "Input Size", "mAP50")
    fig.savefig(output)
    plt.close(fig)


def _save_input_size_fps(rows: list[dict[str, Any]], output: Path) -> None:
    imgsz_vals = _imgsz_list(rows)
    labels = _models(rows)
    xs = [_float(v) for v in imgsz_vals]
    ys = [_float(row.get("fps")) for row in rows]
    valid = [(x, y, label) for x, y, label in zip(xs, ys, labels) if x == x and y == y]
    if not valid:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, "No data available", ha="center", va="center",
                transform=ax.transAxes)
        fig.savefig(output)
        plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    _scatter(ax, [v[0] for v in valid], [v[1] for v in valid], [v[2] for v in valid],
             "Input Size vs FPS", "Input Size", "FPS")
    fig.savefig(output)
    plt.close(fig)


def _save_f1_curve(rows: list[dict[str, Any]], output: Path) -> None:
    labels = _models(rows)
    f1_vals = []
    for row in rows:
        p = _float(row.get("precision"))
        r = _float(row.get("recall"))
        f1_vals.append(2 * p * r / (p + r) if p + r else 0.0)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    bars = ax.bar(labels, f1_vals, color=_series_colors(len(labels)))
    ax.set(title="F1 Score vs Model", ylabel="F1 Score")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, f1_vals):
        if value == value:
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}",
                    ha="center", va="bottom", fontsize=8)
    fig.savefig(output)
    plt.close(fig)


def _copy_confusion_matrix(runs_dir: Path, output: Path) -> Path:
    for cm_path in sorted(runs_dir.glob("*/confusion_matrix.png")):
        shutil.copy2(cm_path, output)
        return output
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5, "No confusion matrix available\\n(not yet generated)",
            ha="center", va="center", transform=ax.transAxes)
    ax.set(title="Confusion Matrix (placeholder)")
    fig.savefig(output)
    plt.close(fig)
    return output


def analyze_experiments(
    summary: str | Path = "results/summary.csv",
    runs_dir: str | Path = "runs",
    output_dir: str | Path = "results/analysis",
) -> list[Path]:
    _configure_matplotlib()
    rows = _read_summary(resolve_path(summary))
    runs_path = resolve_path(runs_dir)
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    generated: list[Path] = []
    model_csv = output / "model_comparison.csv"
    _save_model_comparison_csv(rows, model_csv)
    generated.append(model_csv)
    per_class_csv = output / "per_class_metrics.csv"
    _save_per_class_metrics_csv(runs_path, per_class_csv)
    generated.append(per_class_csv)
    model_bar = output / "model_comparison.png"
    _save_model_comparison_bar(rows, model_bar)
    generated.append(model_bar)
    imgsz_map = output / "input_size_map.png"
    _save_input_size_map(rows, imgsz_map)
    generated.append(imgsz_map)
    imgsz_fps = output / "input_size_fps.png"
    _save_input_size_fps(rows, imgsz_fps)
    generated.append(imgsz_fps)
    pr_chart = output / "pr_curve.png"
    _save_pr_plot(rows, pr_chart)
    generated.append(pr_chart)
    f1_chart = output / "f1_curve.png"
    _save_f1_curve(rows, f1_chart)
    generated.append(f1_chart)
    cm_path = output / "confusion_matrix.png"
    _copy_confusion_matrix(runs_path, cm_path)
    generated.append(cm_path)
    curves, distribution = _load_run_data(runs_path)
    if curves:
        loss_plot = output / "loss_curve_comparison.png"
        _save_curve_plot(curves, loss_plot)
        generated.append(loss_plot)
    if distribution:
        class_dist = output / "class_distribution.png"
        _save_class_distribution(distribution, class_dist)
        generated.append(class_dist)
    return generated


def publish_analysis_assets(
    analysis_dir: str | Path = "results/analysis",
    docs_assets: str | Path = "docs/assets/analysis",
) -> list[Path]:
    analysis_path = resolve_path(analysis_dir)
    required = ["input_size_map.png", "input_size_fps.png", "model_comparison.png"]
    missing = [name for name in required if not (analysis_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Run analyze_experiments first; missing: {', '.join(missing)}"
        )
    docs_path = resolve_path(docs_assets)
    docs_path.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in required:
        dest = docs_path / name
        shutil.copy2(analysis_path / name, dest)
        copied.append(dest)
    return copied


__all__ = ["analyze_experiments", "publish_analysis_assets"]