"""将验证集逐样本预测结果导出为结构化 JSON。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from blade_defect.data.defect_classes import DEFECT_CLASSES
from blade_defect.models.predictor import SegmentationPredictor
from blade_defect.utils.files import load_dataset_config, save_json


def _parse_yolo_label(label_path: Path) -> list[dict[str, Any]]:
    instances: list[dict[str, Any]] = []
    if not label_path.is_file():
        return instances
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        try:
            class_id = int(parts[0])
        except ValueError:
            continue
        instances.append({
            "class_id": class_id,
            "class_name": DEFECT_CLASSES.get(class_id, f"unknown_{class_id}"),
        })
    return instances


def export_validation_predictions(
    model_path: str | Path,
    data_yaml: str | Path,
    output_path: str | Path,
    experiment_id: str,
    imgsz: int = 640,
    device: str = "0",
    conf: float = 0.25,
    iou: float = 0.7,
) -> Path:
    data_config = load_dataset_config(data_yaml)
    val_images_dir = data_config.get("val")
    if val_images_dir is None or not Path(val_images_dir).is_dir():
        raise FileNotFoundError(f"val images directory not found: {val_images_dir}")
    val_labels_dir = data_config["path"] / "labels" / "val"
    if not val_labels_dir.is_dir():
        raise FileNotFoundError(f"val labels directory not found: {val_labels_dir}")

    predictor = SegmentationPredictor(model_path)
    predict_kwargs: dict[str, Any] = dict(
        source=str(val_images_dir),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=False,
        save_txt=False,
        verbose=False,
        stream=True,
    )
    samples: list[dict[str, Any]] = []
    for result in predictor.model.predict(**predict_kwargs):
        image_path = Path(result.path)
        relative_path = image_path.relative_to(val_images_dir)
        label_path = val_labels_dir / relative_path.with_suffix(".txt")
        ground_truth = _parse_yolo_label(label_path)

        predictions: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                conf_val = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
                bbox = box.xyxy.tolist()[0] if hasattr(box.xyxy, "tolist") else list(box.xyxy)
                predictions.append({
                    "class_id": cls_id,
                    "class_name": DEFECT_CLASSES.get(cls_id, f"unknown_{cls_id}"),
                    "confidence": round(conf_val, 4),
                    "bbox": [round(v, 2) for v in bbox],
                })

        true_ids = sorted({gt["class_id"] for gt in ground_truth})
        pred_ids = sorted({p["class_id"] for p in predictions})
        samples.append({
            "image_path": str(image_path),
            "split": "val",
            "true_classes": true_ids,
            "true_class_names": [DEFECT_CLASSES.get(cid, f"unknown_{cid}") for cid in true_ids],
            "predicted_classes": pred_ids,
            "predicted_class_names": [DEFECT_CLASSES.get(cid, f"unknown_{cid}") for cid in pred_ids],
            "ground_truth": ground_truth,
            "predictions": predictions,
        })

    payload = {
        "experiment_id": experiment_id,
        "model": str(model_path),
        "imgsz": imgsz,
        "num_samples": len(samples),
        "samples": samples,
    }
    return save_json(payload, output_path)
