"""运行独立的单轮 OBB 冒烟实验并生成 Markdown 报告。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from blade_defect.data import (
    DEFECT_CLASSES,
    check_obb_dataset,
    check_obb_indexed_samples,
    load_indexed_splits,
)
from blade_defect.experiment.metadata import (
    atomic_write_json,
    collect_dataset_metadata,
    collect_environment_metadata,
    collect_git_metadata,
    create_run_manifest,
    sha256_file,
    utc_timestamp,
)
from blade_defect.utils import load_project_config, resolve_model_reference
from blade_defect.utils.files import resolved_data_yaml


OFFICIAL_WEIGHT = "yolo11s-obb.pt"
OFFICIAL_WEIGHT_SOURCE = (
    "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-obb.pt"
)


def _mean(value: Any) -> float:
    array = np.asarray(value if value is not None else [], dtype=float)
    return float(array.mean()) if array.size else 0.0


def _metrics(result: Any) -> dict[str, float]:
    box = getattr(result, "box", None)
    speed = getattr(result, "speed", {}) or {}
    inference_ms = float(speed.get("inference", 0.0) or 0.0)
    return {
        "box_precision": _mean(getattr(box, "p", None)),
        "box_recall": _mean(getattr(box, "r", None)),
        "box_map50": _mean(getattr(box, "map50", None)),
        "box_map50_95": _mean(getattr(box, "map", None)),
        "fps": 1000.0 / inference_ms if inference_ms > 0 else 0.0,
    }


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    validation = payload.get("validation", {})
    metrics = payload.get("metrics", {})
    lines = [
        "# OBB smoke report",
        "",
        f"- Status: **{payload['status']}**",
        f"- Timestamp: {payload['timestamp']}",
        f"- Config: `{payload['config']}`",
        f"- Model: `{payload['model']}`",
        f"- Dataset: `{payload['data']}`",
        f"- Training output: `{payload.get('save_dir', '')}`",
        f"- Best weights: `{payload.get('best_weights', '')}`",
        f"- Prediction output: `{payload.get('prediction_output', '')}`",
        "",
        "## Dataset validation",
        "",
        f"- Valid: {validation.get('valid', False)}",
        f"- Train instances: {validation.get('train_instances', 0)}",
        f"- Val instances: {validation.get('val_instances', 0)}",
        "",
        "## Validation metrics",
        "",
        f"- Box Precision: {metrics.get('box_precision', 0):.6f}",
        f"- Box Recall: {metrics.get('box_recall', 0):.6f}",
        f"- Box mAP50: {metrics.get('box_map50', 0):.6f}",
        f"- Box mAP50-95: {metrics.get('box_map50_95', 0):.6f}",
        f"- FPS: {metrics.get('fps', 0):.3f}",
    ]
    if payload.get("error"):
        lines.extend(["", "## Error", "", str(payload["error"])])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_smoke_dataset(data_path: Path) -> tuple[dict[str, Any], list[Path]]:
    """Validate train/val only; a declared test split is intentionally never loaded."""
    config = yaml.safe_load(data_path.read_text(encoding="utf-8-sig"))
    indexed_mode = all(
        str(config.get(split, "")).casefold().endswith(".txt") for split in ("train", "val")
    )
    if indexed_mode:
        indexed = load_indexed_splits(data_path, ("train", "val"))
        reports = {
            split: check_obb_indexed_samples(
                indexed[split], num_classes=len(DEFECT_CLASSES)
            )
            for split in ("train", "val")
        }
        val_images = [sample.image_path for sample in indexed["val"]]
    else:
        dataset_root = data_path.parent
        reports = {
            split: check_obb_dataset(
                dataset_root / "images" / split,
                dataset_root / "labels" / split,
                num_classes=len(DEFECT_CLASSES),
            )
            for split in ("train", "val")
        }
        val_images = sorted((dataset_root / "images" / "val").glob("*"))
    if not all(report.valid for report in reports.values()):
        raise RuntimeError(
            "OBB dataset validation failed: "
            + "; ".join(
                f"{split}={report.error_type_counts}" for split, report in reports.items()
                if not report.valid
            )
        )
    return {
        "valid": True,
        "mode": "indexed" if indexed_mode else "directory",
        "validated_splits": ["train", "val"],
        "test_used": False,
        "train_samples": reports["train"].images,
        "val_samples": reports["val"].images,
        "train_instances": reports["train"].valid_instances,
        "val_instances": reports["val"].valid_instances,
    }, val_images


def acquire_official_weight(
    model_reference: str | Path, project_root: Path, provenance_path: Path
) -> Path:
    """Acquire only the official Ultralytics YOLO11s-obb checkpoint and record provenance."""
    requested = Path(str(model_reference)).name
    if requested != OFFICIAL_WEIGHT:
        raise ValueError(f"smoke requires official {OFFICIAL_WEIGHT}, got {model_reference}")
    config_dir = project_root / "results" / "ultralytics_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(config_dir))
    local = Path(model_reference) if isinstance(model_reference, Path) else project_root / requested
    acquired_at = datetime.now().astimezone().isoformat(timespec="seconds")
    payload: dict[str, Any] = {
        "filename": OFFICIAL_WEIGHT,
        "source": OFFICIAL_WEIGHT_SOURCE,
        "local_path": str(local),
        "size_bytes": 0,
        "sha256": None,
        "acquired_at": acquired_at,
        "verified": False,
    }
    try:
        downloaded_now = False
        if not local.is_file():
            from ultralytics.utils.downloads import attempt_download_asset

            downloaded = Path(attempt_download_asset(local, release="v8.4.0"))
            local = downloaded.resolve()
            downloaded_now = True
        if not local.is_file() or local.stat().st_size < 100_000:
            raise RuntimeError(f"official checkpoint is missing or too small: {local}")
        current_sha = _sha256(local)
        if not downloaded_now:
            cached_provenance = (
                json.loads(provenance_path.read_text(encoding="utf-8"))
                if provenance_path.is_file()
                else {}
            )
            if not (
                cached_provenance.get("verified") is True
                and cached_provenance.get("source") == OFFICIAL_WEIGHT_SOURCE
                and cached_provenance.get("sha256") == current_sha
            ):
                raise RuntimeError(
                    "existing checkpoint has no matching verified official provenance"
                )
        from ultralytics import YOLO

        inspected = YOLO(str(local))
        task = getattr(inspected, "task", None)
        if task != "obb":
            raise RuntimeError(f"checkpoint task is {task!r}, expected 'obb'")
        payload.update(
            {
                "local_path": str(local),
                "size_bytes": local.stat().st_size,
                "sha256": current_sha,
                "verified": True,
                "verification": (
                    "downloaded from pinned official Ultralytics release and loaded with task=obb"
                    if downloaded_now
                    else "SHA matched prior verified official provenance and loaded with task=obb"
                ),
            }
        )
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        provenance_path.parent.mkdir(parents=True, exist_ok=True)
        provenance_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        raise
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return local


def run_smoke(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config, project_root = load_project_config(
        config_file,
        path_fields=("data", "project", "report", "prediction_output", "weight_provenance"),
    )
    model_reference = resolve_model_reference(config.pop("model"), project_root)
    data_path = Path(config.pop("data"))
    report_path = Path(config.pop("report"))
    prediction_output = Path(config.pop("prediction_output"))
    weight_provenance = Path(
        config.pop("weight_provenance", report_path.parent / "weight_provenance.json")
    )
    project = Path(config["project"])
    device = config.get("device", 0)
    imgsz = int(config.get("imgsz", 960))
    payload: dict[str, Any] = {
        "status": "failed",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": str(config_file),
        "model": str(model_reference),
        "data": str(data_path),
        "prediction_output": str(prediction_output),
    }
    try:
        payload["validation"], val_images = validate_smoke_dataset(data_path)
        official_weight = acquire_official_weight(
            model_reference, project_root, weight_provenance
        )

        from ultralytics import YOLO

        model = YOLO(str(official_weight))
        # 派生 data.yaml 的 path 为 "."；Ultralytics 按当前工作目录解析相对 path，
        # 必须先绝对化（与 seg 正式实验同一 resolved_data_yaml 链路）。
        with resolved_data_yaml(data_path) as normalized_data:
            train_started = time.perf_counter()
            train_result = model.train(data=normalized_data, **config)
            train_seconds = time.perf_counter() - train_started
            save_dir = Path(getattr(train_result, "save_dir", None) or project / str(config["name"]))
            best_weights = save_dir / "weights" / "best.pt"
            last_weights = save_dir / "weights" / "last.pt"
            if not best_weights.is_file():
                raise RuntimeError(f"smoke training did not produce best weights: {best_weights}")
            if not last_weights.is_file():
                raise RuntimeError(f"smoke training did not produce last weights: {last_weights}")

            trained = YOLO(str(best_weights))
            validation_result = trained.val(
                data=normalized_data, task="obb", imgsz=imgsz, device=device, workers=config.get("workers", 2)
            )
        if not val_images:
            raise RuntimeError("validation split contains no images for prediction smoke")
        trained.predict(
            source=str(val_images[0]), task="obb", imgsz=imgsz, device=device,
            save=True, project=str(prediction_output.parent), name=prediction_output.name,
            exist_ok=True,
        )
        metrics = _metrics(validation_result)
        environment = collect_environment_metadata()
        git_metadata = collect_git_metadata(project_root)
        dataset_metadata = collect_dataset_metadata(data_path)
        dataset_manifest = dataset_metadata.get("manifest") or {}
        metrics_payload = {
            **metrics,
            "train_seconds": train_seconds,
            "seconds_per_epoch": train_seconds,
            "epochs": int(config.get("epochs", 1)),
            "batch": int(config.get("batch", 4)),
        }
        if not (save_dir / "results.csv").is_file():
            raise RuntimeError(f"smoke training did not produce results.csv: {save_dir / 'results.csv'}")
        # smoke 同样需要逐类指标（box 分支），与正式实验同一导出函数。
        from blade_defect.evaluation import per_class_metrics_from_ultralytics
        from blade_defect.experiment.runner import _write_per_class_metrics

        _write_per_class_metrics(
            per_class_metrics_from_ultralytics(validation_result),
            save_dir / "per_class_metrics.csv",
        )
        atomic_write_json(save_dir / "metrics.json", metrics_payload)
        atomic_write_json(save_dir / "environment.json", environment)
        manifest = create_run_manifest(
            experiment={
                "name": config["name"], "imgsz": imgsz,
                "epochs": int(config.get("epochs", 1)), "batch": int(config.get("batch", 4)),
                "seed": int(config.get("seed", 42)),
            },
            effective_config={**config, "data": str(data_path), "task": "obb"},
            config_path=config_file,
            model=official_weight,
            requested_device=device,
            git=git_metadata,
            dataset=dataset_metadata,
            environment=environment,
        )
        manifest.update(
            {
                "status": "completed",
                "finished_at": utc_timestamp(),
                "parent_dataset_id": dataset_manifest.get("parent_dataset_id"),
                "test_used": False,
                "validated_splits": ["train", "val"],
                "weight_provenance": json.loads(weight_provenance.read_text(encoding="utf-8")),
                "metrics": metrics_payload,
                "resume_supported": True,
                "resume_checkpoint": str(last_weights),
                "artifacts": {
                    "run_directory": str(save_dir),
                    "last_weights": str(last_weights),
                    "last_weights_sha256": sha256_file(last_weights),
                    "best_weights": str(best_weights),
                    "best_weights_sha256": sha256_file(best_weights),
                    "results_csv": str(save_dir / "results.csv"),
                    "metrics": str(save_dir / "metrics.json"),
                    "environment": str(save_dir / "environment.json"),
                    "prediction_output": str(prediction_output),
                },
            }
        )
        atomic_write_json(save_dir / "run_manifest.json", manifest)
        payload.update(
            {
                "status": "passed",
                "save_dir": str(save_dir),
                "best_weights": str(best_weights),
                "last_weights": str(last_weights),
                "metrics": metrics_payload,
                "environment": str(save_dir / "environment.json"),
                "run_manifest": str(save_dir / "run_manifest.json"),
            }
        )
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
        _write_report(report_path, payload)
        raise
    _write_report(report_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/experiments/obb_smoke_yolo11s_960.yaml")
    )
    args = parser.parse_args()
    print(json.dumps(run_smoke(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
