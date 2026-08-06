"""统一实验资产盘点：生成唯一实验索引、重复ID检查、指标核对和资产完整性报告。"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any, Iterable

SAMPLED_RUNS_DEFAULT = "run/run/runs"
PRIMARY_RUNS_DEFAULT = "runs"
OUTPUT_DEFAULT = "results/experiment_registry"

REQUIRED_ASSETS = ["metrics.json", "results.csv", "weights/best.pt"]
METRIC_BRANCHES = ["mask", "box"]


def _read_json(path: pathlib.Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _resolve_metrics_file(run_dir: pathlib.Path) -> pathlib.Path | None:
    for name in ("metrics.json", "metrics(1).json"):
        candidate = run_dir / name
        if candidate.is_file():
            return candidate
    return None


def _find_run_dirs(root: pathlib.Path) -> list[pathlib.Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _asset_status(run_dir: pathlib.Path) -> dict[str, bool]:
    return {
        "metrics.json": any((run_dir / n).is_file() for n in ("metrics.json", "metrics(1).json")),
        "per_class_metrics.csv": any((run_dir / n).is_file() for n in ("per_class_metrics.csv", "per_class_metrics(1).csv")),
        "validation_predictions.json": any(
            (run_dir / n).is_file() for n in ("validation_predictions.json", "validation_predictions(1).json")
        ),
        "results.csv": (run_dir / "results.csv").is_file(),
        "weights/best.pt": (run_dir / "weights" / "best.pt").is_file(),
    }


def _registry_row(experiment_id: str, source: str, run_dir: pathlib.Path) -> dict[str, Any]:
    metrics_path = _resolve_metrics_file(run_dir)
    metrics = _read_json(metrics_path) if metrics_path else {}
    assets = _asset_status(run_dir)
    hw = metrics.get("hardware", {})
    return {
        "experiment_id": metrics.get("name", experiment_id),
        "source_dir": source,
        "dataset_id": metrics.get("dataset_id", "blade-v2-sampled-4987" if "sampled" in source else "blade-v2"),
        "model": metrics.get("model", ""),
        "imgsz": metrics.get("imgsz", ""),
        "epochs": metrics.get("epochs", ""),
        "seed": metrics.get("seed", ""),
        "batch": metrics.get("batch", ""),
        "code_commit": metrics.get("code_commit", ""),
        "device": hw.get("gpu_name", ""),
        "fps_method": metrics.get("fps_method", "ultralytics_val_speed_inference_ms"),
        "mask_mAP50": round(float(metrics.get("mask_mAP50", metrics.get("mAP50", 0))), 4),
        "mask_mAP50-95": round(float(metrics.get("mask_mAP50-95", metrics.get("mAP50-95", 0))), 4),
        "box_mAP50": round(float(metrics.get("box_mAP50", 0)), 4),
        "box_mAP50-95": round(float(metrics.get("box_mAP50-95", 0)), 4),
        "precision": round(float(metrics.get("precision", 0)), 4),
        "recall": round(float(metrics.get("recall", 0)), 4),
        "fps": round(float(metrics.get("fps", 0)), 2),
        "assets_metrics_json": assets["metrics.json"],
        "assets_per_class": assets["per_class_metrics.csv"],
        "assets_predictions": assets["validation_predictions.json"],
        "assets_results_csv": assets["results.csv"],
        "assets_best_pt": assets["weights/best.pt"],
        "status": metrics.get("status", "unknown"),
    }


def build_experiment_registry(
    primary_runs: str | pathlib.Path = PRIMARY_RUNS_DEFAULT,
    sampled_runs: str | pathlib.Path = SAMPLED_RUNS_DEFAULT,
    output_dir: str | pathlib.Path = OUTPUT_DEFAULT,
) -> list[pathlib.Path]:
    primary_root = pathlib.Path(primary_runs)
    sampled_root = pathlib.Path(sampled_runs)
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for run_dir in _find_run_dirs(sampled_root):
        rows.append(_registry_row(run_dir.name, "sampled", run_dir))
    for run_dir in _find_run_dirs(primary_root):
        rows.append(_registry_row(run_dir.name, "primary", run_dir))

    # experiment_registry.csv
    fields = list(rows[0].keys()) if rows else []
    registry_path = out / "experiment_registry.csv"
    with registry_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # duplicate_experiment_ids.csv
    dup_path = out / "duplicate_experiment_ids.csv"
    by_id: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_id.setdefault(row["experiment_id"], []).append(row)
    dup_rows = [
        {"experiment_id": eid, "occurrences": len(group), "source_dirs": " | ".join(r["source_dir"] for r in group)}
        for eid, group in sorted(by_id.items())
        if len(group) > 1
    ]
    with dup_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment_id", "occurrences", "source_dirs"])
        writer.writeheader()
        writer.writerows(dup_rows)

    # metric_reconciliation.csv
    rec_path = out / "metric_reconciliation.csv"
    by_model_size: dict[tuple[str, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row["model"], row["imgsz"])
        by_model_size.setdefault(key, []).append(row)
    rec_rows = []
    for (model, imgsz), group in sorted(by_model_size.items()):
        if len(group) < 2:
            continue
        base = group[0]
        for other in group[1:]:
            rec_rows.append({
                "model": model,
                "imgsz": imgsz,
                "experiment_id_a": base["experiment_id"],
                "source_a": base["source_dir"],
                "mask_mAP50_a": base["mask_mAP50"],
                "mask_mAP50-95_a": base["mask_mAP50-95"],
                "fps_a": base["fps"],
                "experiment_id_b": other["experiment_id"],
                "source_b": other["source_dir"],
                "mask_mAP50_b": other["mask_mAP50"],
                "mask_mAP50-95_b": other["mask_mAP50-95"],
                "fps_b": other["fps"],
                "mAP50_diff": round(other["mask_mAP50"] - base["mask_mAP50"], 4),
                "mAP50-95_diff": round(other["mask_mAP50-95"] - base["mask_mAP50-95"], 4),
                "fps_diff": round(other["fps"] - base["fps"], 2),
            })
    with rec_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["model", "imgsz", "experiment_id_a", "source_a", "mask_mAP50_a",
                  "mask_mAP50-95_a", "fps_a", "experiment_id_b", "source_b",
                  "mask_mAP50_b", "mask_mAP50-95_b", "fps_b",
                  "mAP50_diff", "mAP50-95_diff", "fps_diff"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rec_rows)

    # asset_completeness.csv
    asset_path = out / "asset_completeness.csv"
    asset_rows = [
        {
            "experiment_id": r["experiment_id"],
            "source_dir": r["source_dir"],
            "metrics.json": r["assets_metrics_json"],
            "per_class_metrics.csv": r["assets_per_class"],
            "validation_predictions.json": r["assets_predictions"],
            "results.csv": r["assets_results_csv"],
            "weights/best.pt": r["assets_best_pt"],
            "complete": all([r["assets_metrics_json"], r["assets_per_class"],
                             r["assets_predictions"], r["assets_results_csv"], r["assets_best_pt"]]),
        }
        for r in rows
    ]
    with asset_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(asset_rows[0].keys()))
        writer.writeheader()
        writer.writerows(asset_rows)

    return [registry_path, dup_path, rec_path, asset_path]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-runs", default=PRIMARY_RUNS_DEFAULT, help="全量/正式实验根目录")
    parser.add_argument("--sampled-runs", default=SAMPLED_RUNS_DEFAULT, help="抽样实验根目录")
    parser.add_argument("--output-dir", default=OUTPUT_DEFAULT, help="输出目录")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in build_experiment_registry(args.primary_runs, args.sampled_runs, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
