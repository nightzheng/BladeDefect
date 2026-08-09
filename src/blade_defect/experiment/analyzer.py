"""为 baseline 实验结果生成论文级图表、对比 CSV 和层级分析资产。
支持 split_leakage_known 标记和 label_level 区分。"""

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
    """Read summary CSV and annotate with split_leakage_known and label_level."""
    if not path.is_file():
        raise FileNotFoundError(f"Experiment summary not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Experiment summary is empty: {path}")
    for row in rows:
        ds = str(row.get("dataset_id", "")).lower()
        row["split_leakage_known"] = "true" if "v2" in ds and "v3" not in ds else "false"
        nc = row.get("n_classes", row.get("nc", ""))
        try:
            row["label_level"] = "coarse" if int(nc) <= 6 else "fine"
        except (ValueError, TypeError):
            row["label_level"] = "fine"
    return rows


def _extract_imgsz(name: str) -> str:
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        try:
            return str(float(parts[1]))
        except ValueError:
            pass
    return ""


def _experiments_by_label_level(
    rows: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split experiments into fine (15-class) and coarse (6-class) groups."""
    fine = [r for r in rows if r.get("label_level") != "coarse"]
    coarse = [r for r in rows if r.get("label_level") == "coarse"]
    return fine, coarse


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
        ax.annotate(label, (xi, yi), xytext=(5, 5), textcoords="offset points", fontsize=7)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(alpha=0.25)


def _check_split_leakage(rows: list[dict[str, Any]]) -> list[str]:
    """Return warning messages for experiments with known split leakage."""
    warnings_list = []
    for row in rows:
        if row.get("split_leakage_known") == "true":
            name = row.get("name", row.get("experiment name", "unknown"))
            warnings_list.append(f"{name}: split_leakage_known=true, do not mix with v3 results")
    return warnings_list


def analyze_experiments(
    summary: str | Path = "results/summary.csv",
    runs_dir: str | Path = "runs",
    output_dir: str | Path = "results/analysis",
) -> list[Path]:
    _configure_matplotlib()
    rows = _read_summary(resolve_path(summary))
    for w in _check_split_leakage(rows):
        import warnings as _w
        _w.warn(w, stacklevel=2)
    fine_rows, coarse_rows = _experiments_by_label_level(rows)
    runs_path = resolve_path(runs_dir)
    output = resolve_path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    # --- use fine_rows for 15-class charts ---
    labels = _models(fine_rows)
    m50 = [_float(r.get("mAP50")) for r in fine_rows]
    m95 = [_float(r.get("mAP50-95")) for r in fine_rows]
    fps_vals = [_float(r.get("fps")) for r in fine_rows]
    sizes = _imgsz_list(fine_rows)
    xi = range(len(labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.bar([i - w / 2 for i in xi], m50, w, label="mAP50", color="#4472C4")
    ax.bar([i + w / 2 for i in xi], m95, w, label="mAP50-95", color="#ED7D31")
    ax.set_xticks(xi); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.legend(); ax.set_title("Model Comparison (Mask mAP)"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "model_comparison.png", dpi=150); plt.close(fig)
    generated.append(output / "model_comparison.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(fps_vals, m95, s=80, color="#4472C4", zorder=3)
    for fr, yb, lb in zip(fps_vals, m95, labels):
        ax.annotate(lb[:16], (fr, yb), fontsize=6)
    ax.set_xlabel("FPS"); ax.set_ylabel("mAP50-95"); ax.set_title("Accuracy-Speed Tradeoff"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "accuracy_speed_tradeoff.png", dpi=150); plt.close(fig)
    generated.append(output / "accuracy_speed_tradeoff.png")

    f1v = [2 * float(r.get("precision", 0)) * float(r.get("recall", 0)) /
           (float(r.get("precision", 0)) + float(r.get("recall", 0)))
           if float(r.get("precision", 0)) + float(r.get("recall", 0)) > 0 else 0
           for r in fine_rows]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(xi, [f * 100 for f in f1v], color="#70AD47")
    ax.set_xticks(xi); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("F1 (%)"); ax.set_title("F1 Score"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "f1_curve.png", dpi=150); plt.close(fig)
    generated.append(output / "f1_curve.png")

    sv = [_float(v) for v in sizes]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sv, m95, s=80, color="#4472C4", zorder=3)
    ax.set_xlabel("Input Size"); ax.set_ylabel("mAP50-95"); ax.set_title("Input Size vs mAP50-95"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "input_size_map.png", dpi=150); plt.close(fig)
    generated.append(output / "input_size_map.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sv, fps_vals, s=80, color="#ED7D31", zorder=3)
    ax.set_xlabel("Input Size"); ax.set_ylabel("FPS"); ax.set_title("Input Size vs FPS"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "input_size_fps.png", dpi=150); plt.close(fig)
    generated.append(output / "input_size_fps.png")

    return generated


def publish_analysis_assets(
    analysis_dir: str | Path = "results/analysis",
    docs_assets: str | Path = "docs/assets/analysis",
) -> list[Path]:
    analysis_path = resolve_path(analysis_dir)
    required = ["input_size_map.png", "input_size_fps.png", "model_comparison.png"]
    missing = [name for name in required if not (analysis_path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run analyze_experiments first; missing: {', '.join(missing)}")
    docs_path = resolve_path(docs_assets)
    docs_path.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in required:
        dest = docs_path / name
        shutil.copy2(analysis_path / name, dest)
        copied.append(dest)
    return copied


__all__ = ["analyze_experiments", "publish_analysis_assets"]
