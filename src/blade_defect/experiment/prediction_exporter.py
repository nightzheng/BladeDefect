"""将验证集逐样本预测结果导出为结构化 JSON。

导出的是原始逐样本/逐实例字段（GT polygon、预测 bbox/mask、置信度、
图像尺寸），不在导出侧判定最终错误类型；FN/FP/匹配关系由分析侧结合
实例匹配与人工复核确认。类别名从 data.yaml 的 ``names`` 读取，
兼容 15 类细粒度与 6 类粗粒度数据集。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from blade_defect.data.validation import (
    _iter_images,
    _label_path_for,
    _resolve_index_entry,
    load_dataset_config_portable,
)
from blade_defect.models.predictor import SegmentationPredictor
from blade_defect.utils.files import save_json


def _names_lookup(data_config: dict[str, Any]) -> dict[int, str]:
    names = data_config.get("names", {})
    if isinstance(names, dict):
        return {int(key): str(value) for key, value in names.items()}
    if isinstance(names, list):
        return {index: str(value) for index, value in enumerate(names)}
    return {}


def _parse_yolo_label(label_path: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    """解析 GT 标签，保留类别与原始归一化 polygon 坐标。"""
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
            polygon = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        instances.append({
            "class_id": class_id,
            "class_name": names.get(class_id, f"unknown_{class_id}"),
            "polygon": polygon,
        })
    return instances


def _val_image_entries(data_config: dict[str, Any]) -> list[tuple[Path, Path]]:
    """返回 (image_path, label_path) 列表，支持目录型与 txt 清单型 val。"""
    dataset_root = Path(data_config["path"])
    labels_dir = dataset_root / "labels" / "val"
    val_entry = data_config.get("val")
    if val_entry is None:
        raise FileNotFoundError("data.yaml 缺少 val 字段")
    val_path = Path(val_entry)
    if val_path.is_file() and val_path.suffix.lower() == ".txt":
        # txt 清单型：图片/标签可位于 dataset_root 之外（如 v3 grouped 复用
        # blade-v2 实体文件），标签逐张按 images→labels 约定推导。
        images = [
            _resolve_index_entry(dataset_root, line.strip())
            for line in val_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif val_path.is_dir():
        images = _iter_images(val_path)
        if not labels_dir.is_dir():
            raise FileNotFoundError(f"val labels directory not found: {labels_dir}")
    else:
        raise FileNotFoundError(f"val images entry not found: {val_entry}")
    return [(image, _label_path_for(dataset_root, image, "val")) for image in images]


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
    data_config = load_dataset_config_portable(data_yaml)
    names = _names_lookup(data_config)
    entries = _val_image_entries(data_config)
    label_by_image = {str(image): label for image, label in entries}

    predictor = SegmentationPredictor(model_path)
    # 逐张推理：一次性传入整张清单会触发 ultralytics autocast_list
    # 将全部图片同时载入内存（48k 张数据集会直接 OOM）。
    samples: list[dict[str, Any]] = []
    for image_path, _ in entries:
        results = predictor.model.predict(
            source=str(image_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            save=False,
            save_txt=False,
            verbose=False,
        )
        result = next(iter(results))
        label_path = label_by_image.get(str(image_path))
        if label_path is None:
            label_path = _label_path_for(Path(data_config["path"]), image_path, "val")
        ground_truth = _parse_yolo_label(label_path, names)

        orig_shape = getattr(result, "orig_shape", None) or (None, None)
        predictions: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        mask_polygons = getattr(masks, "xy", None) if masks is not None else None
        if boxes is not None:
            for index, box in enumerate(boxes):
                cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls)
                conf_val = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf)
                bbox = box.xyxy.tolist()[0] if hasattr(box.xyxy, "tolist") else list(box.xyxy)
                prediction: dict[str, Any] = {
                    "class_id": cls_id,
                    "class_name": names.get(cls_id, f"unknown_{cls_id}"),
                    "confidence": round(conf_val, 4),
                    "bbox": [round(v, 2) for v in bbox],
                }
                if mask_polygons is not None and index < len(mask_polygons):
                    polygon = mask_polygons[index]
                    if hasattr(polygon, "tolist"):
                        polygon = polygon.tolist()
                    prediction["mask_polygon"] = [
                        [round(float(x), 2), round(float(y), 2)] for x, y in polygon
                    ]
                predictions.append(prediction)

        true_ids = sorted({gt["class_id"] for gt in ground_truth})
        pred_ids = sorted({p["class_id"] for p in predictions})
        samples.append({
            "image_path": str(image_path),
            "split": "val",
            "image_width": orig_shape[1],
            "image_height": orig_shape[0],
            "true_classes": true_ids,
            "true_class_names": [names.get(cid, f"unknown_{cid}") for cid in true_ids],
            "predicted_classes": pred_ids,
            "predicted_class_names": [names.get(cid, f"unknown_{cid}") for cid in pred_ids],
            "ground_truth": ground_truth,
            "predictions": predictions,
        })

    payload = {
        "experiment_id": experiment_id,
        "model": str(model_path),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "num_samples": len(samples),
        "samples": samples,
    }
    return save_json(payload, output_path)
