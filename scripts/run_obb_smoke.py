"""Run the isolated one-epoch OBB smoke experiment and write a Markdown report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from blade_defect.data import DEFECT_CLASSES, check_obb_dataset
from blade_defect.utils import load_project_config, resolve_model_reference


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


def run_smoke(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    config, project_root = load_project_config(
        config_file, path_fields=("data", "project", "report", "prediction_output")
    )
    model_reference = resolve_model_reference(config.pop("model"), project_root)
    data_path = Path(config.pop("data"))
    report_path = Path(config.pop("report"))
    prediction_output = Path(config.pop("prediction_output"))
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
        dataset_root = data_path.parent
        reports = {
            split: check_obb_dataset(
                dataset_root / "images" / split,
                dataset_root / "labels" / split,
                num_classes=len(DEFECT_CLASSES),
            )
            for split in ("train", "val")
        }
        if not all(report.valid for report in reports.values()):
            raise RuntimeError(
                "OBB dataset validation failed: "
                + "; ".join(
                    f"{split}={report.error_type_counts}" for split, report in reports.items()
                    if not report.valid
                )
            )
        payload["validation"] = {
            "valid": True,
            "train_instances": reports["train"].valid_instances,
            "val_instances": reports["val"].valid_instances,
        }

        from ultralytics import YOLO

        model = YOLO(str(model_reference))
        train_result = model.train(data=str(data_path), **config)
        save_dir = Path(getattr(train_result, "save_dir", None) or project / str(config["name"]))
        best_weights = save_dir / "weights" / "best.pt"
        if not best_weights.is_file():
            raise RuntimeError(f"smoke training did not produce best weights: {best_weights}")

        trained = YOLO(str(best_weights))
        validation_result = trained.val(
            data=str(data_path), task="obb", imgsz=imgsz, device=device, workers=config.get("workers", 2)
        )
        val_images = sorted((dataset_root / "images" / "val").glob("*"))
        if not val_images:
            raise RuntimeError("validation split contains no images for prediction smoke")
        trained.predict(
            source=str(val_images[0]), task="obb", imgsz=imgsz, device=device,
            save=True, project=str(prediction_output.parent), name=prediction_output.name,
        )
        payload.update(
            {
                "status": "passed",
                "save_dir": str(save_dir),
                "best_weights": str(best_weights),
                "metrics": _metrics(validation_result),
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
