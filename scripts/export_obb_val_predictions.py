"""导出 OBB e50 冻结权重在正式 val 上的逐样本预测（只读推理，不训练）。

阶段收尾周任务：向张劲焱交付 OBB val 预测与共同 Box 分析入口，支持其在
统一尺寸口径下计算 seg Box 与 OBB 的 small/medium/large 指标。本脚本：

- 使用 ``runs/v3_yolo11s_obb_960_e50/weights/best.pt`` 冻结权重，加载前核对
  run_manifest 中的 ``best_weights_sha256``；
- 只读取正式 val 索引（``datasets/blade-v3-grouped-obb/val.txt``），
  不读取 test，不启动任何训练；
- 逐张推理（避免整清单载入导致 OOM），导出与 seg 侧 ``validation_predictions.json``
  兼容的逐样本结构：GT 为 OBB 四点 polygon（归一化），预测含四点 polygon
  （像素坐标）、置信度、类别，以及为共同 Box 口径准备的轴对齐外接框 ``bbox``；
- 自动派生 OBB 来自 seg polygon 最小外接矩形，不等同人工旋转框真值。

用法：
    python scripts/export_obb_val_predictions.py
    python scripts/export_obb_val_predictions.py --output results/obb_v3_val_predictions/obb_val_predictions.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from blade_defect.data.indexed_splits import load_index_config, load_indexed_split
from blade_defect.experiment.metadata import (
    collect_git_metadata,
    load_json,
    sha256_file,
    utc_timestamp,
)
from blade_defect.utils.device import resolve_device
from blade_defect.utils.files import save_json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = PROJECT_ROOT / "runs" / "v3_yolo11s_obb_960_e50"
DEFAULT_DATA_YAML = PROJECT_ROOT / "datasets" / "blade-v3-grouped-obb" / "data.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "obb_v3_val_predictions" / "obb_val_predictions.json"


def parse_obb_label(label_path: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    """解析 YOLO OBB 标签（class + 8 个归一化坐标），保留类别与四点 polygon。"""
    instances: list[dict[str, Any]] = []
    if not label_path.is_file():
        return instances
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        if not line.strip():
            continue
        parts = line.strip().split()
        if len(parts) != 9:
            continue
        try:
            class_id = int(parts[0])
            points = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        instances.append(
            {
                "class_id": class_id,
                "class_name": names.get(class_id, f"unknown_{class_id}"),
                "polygon": points,
            }
        )
    return instances


def obb_points_to_envelope(points: list[list[float]]) -> list[float]:
    """由四点 polygon（像素坐标）计算轴对齐外接框 xyxy，用于共同 Box 口径。"""
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalized_polygon_to_envelope(polygon: list[float], width: int, height: int) -> list[float]:
    """由归一化四点 polygon 计算像素轴对齐外接框 xyxy，用于共同 Box 口径。"""
    xs = polygon[0::2]
    ys = polygon[1::2]
    return [min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height]


def export_obb_val_predictions(
    *,
    run_dir: Path,
    data_yaml: Path,
    output_path: Path,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 960,
    device: str = "0",
) -> Path:
    run_manifest = load_json(run_dir / "run_manifest.json")
    best_weights = run_dir / "weights" / "best.pt"
    expected_sha = run_manifest.get("artifacts", {}).get("best_weights_sha256")
    actual_sha = sha256_file(best_weights)
    if expected_sha and expected_sha != actual_sha:
        raise RuntimeError(
            f"best.pt SHA-256 与 run_manifest 不一致：{actual_sha} != {expected_sha}，终止导出"
        )

    _, data_config = load_index_config(data_yaml)
    names_raw = data_config.get("names", {})
    names = {int(key): str(value) for key, value in dict(names_raw).items()}
    samples_index = load_indexed_split(data_yaml, "val")

    from blade_defect.models.trainer import _load_yolo

    model = _load_yolo()(str(best_weights))
    resolved_device = resolve_device(device)

    samples: list[dict[str, Any]] = []
    for sample in samples_index:
        results = model.predict(
            source=str(sample.image_path),
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=resolved_device,
            task="obb",
            save=False,
            save_txt=False,
            verbose=False,
        )
        result = next(iter(results))
        orig_shape = getattr(result, "orig_shape", None) or (None, None)
        ground_truth = parse_obb_label(sample.label_path, names)

        predictions: list[dict[str, Any]] = []
        obb = getattr(result, "obb", None)
        if obb is not None:
            xyxyxyxy = getattr(obb, "xyxyxyxy", None)
            points_list = xyxyxyxy.tolist() if hasattr(xyxyxyxy, "tolist") else []
            for index in range(len(points_list)):
                cls_id = int(obb.cls[index].item())
                conf_val = float(obb.conf[index].item())
                points = [[round(float(x), 2), round(float(y), 2)] for x, y in points_list[index]]
                predictions.append(
                    {
                        "class_id": cls_id,
                        "class_name": names.get(cls_id, f"unknown_{cls_id}"),
                        "confidence": round(conf_val, 4),
                        "obb_points": points,
                        "bbox": [round(v, 2) for v in obb_points_to_envelope(points)],
                    }
                )

        true_ids = sorted({gt["class_id"] for gt in ground_truth})
        pred_ids = sorted({p["class_id"] for p in predictions})
        samples.append(
            {
                "image_path": str(sample.image_path),
                "split": "val",
                "image_width": orig_shape[1],
                "image_height": orig_shape[0],
                "true_classes": true_ids,
                "true_class_names": [names.get(cid, f"unknown_{cid}") for cid in true_ids],
                "predicted_classes": pred_ids,
                "predicted_class_names": [names.get(cid, f"unknown_{cid}") for cid in pred_ids],
                "ground_truth": ground_truth,
                "predictions": predictions,
            }
        )

    payload = {
        "experiment_id": run_manifest.get("experiment_id", "v3_yolo11s_obb_960_e50"),
        "task": "obb",
        "model": str(best_weights),
        "model_sha256": actual_sha,
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256_file(data_yaml),
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "num_samples": len(samples),
        "generated_at": utc_timestamp(),
        "git": collect_git_metadata(PROJECT_ROOT),
        "common_box_note": (
            "GT 为 seg polygon 最小外接矩形派生的四点 OBB（归一化）；预测含四点 polygon"
            "（像素）与轴对齐外接框 bbox（像素，由 obb_points 包络得到）。共同 Box 口径"
            "分析统一使用 bbox/envelope + IoU≥0.5 类别感知匹配；自动派生 OBB 不等同"
            "人工旋转框真值；仅正式 val，test 未读取。"
        ),
        "samples": samples,
    }
    return save_json(payload, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--run", default=str(DEFAULT_RUN_DIR), help="OBB e50 运行目录")
    parser.add_argument("--data", default=str(DEFAULT_DATA_YAML), help="OBB 数据集 data.yaml")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="预测 JSON 输出路径")
    parser.add_argument("--conf", type=float, default=0.25, help="导出置信度阈值（与 seg 导出一致）")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值（与 seg 导出一致）")
    parser.add_argument("--imgsz", type=int, default=960, help="推理输入尺寸")
    parser.add_argument("--device", default="0", help="推理设备")
    args = parser.parse_args()
    output = export_obb_val_predictions(
        run_dir=Path(args.run),
        data_yaml=Path(args.data),
        output_path=Path(args.output),
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"exported: {output}")


if __name__ == "__main__":
    main()
