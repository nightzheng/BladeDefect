#!/usr/bin/env python
"""类 13 漏检样本的形态分析：把"为什么漏"从假设变成可验证的证据。

回答三件事：
  1. 训练集里类 13 的 115 个实例，形态分布是什么样（面积/长宽比/填充率/位置/亮度）
  2. 19 个漏检实例，形态上离训练集有多远（是否落在分布外 → 分布缺口）
  3. 漏检 GT 是否与同图其它类的标注重叠（→ 标注口径混用的直接证据）

产物：
  class13_train_distribution.csv   训练集 115 个实例的形态指标
  class13_missed_profile.csv       19 个漏检实例的形态指标 + 最近邻距离
  class13_overlap_check.csv        漏检 GT 与其它类 GT 的最大 IoU
  class13_missed_summary.json      汇总统计
  sheets/class13_missed_review.jpg 19 个漏检样本的可视化图版
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.data.indexed_splits import load_indexed_split  # noqa: E402

CLASS_ID = 13


def _coords(points):
    """polygon 支持 [[x,y],...] 与扁平 [x1,y1,...]。"""
    if points and isinstance(points[0], (list, tuple)):
        return [(float(p[0]), float(p[1])) for p in points]
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points) - 1, 2)]


def _shoelace(pts):
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _bbox(pts):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _crop_stats(gray, hsv, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h, w = gray.shape
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, w), min(y2, h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    patch = gray[y1:y2, x1:x2]
    sat = hsv[y1:y2, x1:x2, 1]
    gx = np.abs(np.diff(patch.astype(np.float32), axis=1)).mean()
    gy = np.abs(np.diff(patch.astype(np.float32), axis=0)).mean()
    return {
        "brightness": float(patch.mean()),
        "brightness_std": float(patch.std()),
        "saturation": float(sat.mean()),
        "edge_density": float((gx + gy) / 2.0),
    }


def _instance_features(image_path: Path, polygon_norm):
    from PIL import Image

    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    pts = [(x * w, y * h) for x, y in polygon_norm]
    area = _shoelace(pts)
    x1, y1, x2, y2 = _bbox(pts)
    bw, bh = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)

    import cv2  # 环境已装 opencv

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    cx, cy = np.mean([p[0] for p in pts]) / w, np.mean([p[1] for p in pts]) / h
    stats = _crop_stats(gray, hsv, (x1, y1, x2, y2)) or {}
    context = _crop_stats(gray, hsv, (x1 - bw, y1 - bh, x2 + bw, y2 + bh)) or {}

    return {
        "image_width": w,
        "image_height": h,
        "area_px": round(area, 1),
        "area_ratio": round(area / (w * h), 6),
        "aspect": round(max(bw, bh) / min(bw, bh), 3),
        "fill_ratio": round(area / (bw * bh), 3),
        "vertices": len(pts),
        "centroid_x": round(float(cx), 3),
        "centroid_y": round(float(cy), 3),
        "touches_border": bool(min(x1, y1) <= 1 or x2 >= w - 1 or y2 >= h - 1),
        "brightness": round(stats.get("brightness", float("nan")), 1),
        "brightness_std": round(stats.get("brightness_std", float("nan")), 1),
        "saturation": round(stats.get("saturation", float("nan")), 1),
        "edge_density": round(stats.get("edge_density", float("nan")), 2),
        "context_brightness": round(context.get("brightness", float("nan")), 1),
        "contrast_vs_context": round(
            stats.get("brightness", 0.0) - context.get("brightness", 0.0), 1
        ),
    }


def _iou(pts_a, pts_b):
    from shapely.geometry import Polygon

    pa, pb = Polygon(pts_a), Polygon(pts_b)
    if not pa.is_valid:
        pa = pa.buffer(0)
    if not pb.is_valid:
        pb = pb.buffer(0)
    if pa.area == 0 or pb.area == 0:
        return 0.0
    inter = pa.intersection(pb).area
    return inter / (pa.area + pb.area - inter) if inter > 0 else 0.0


def main() -> int:
    from PIL import Image, ImageDraw

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--predictions", required=True, help="B-1 的 validation_predictions.json")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    out = Path(args.out_dir)
    (out / "sheets").mkdir(parents=True, exist_ok=True)

    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    samples = payload["samples"]

    # ---------- 训练集类 13 分布 ----------
    train_rows = []
    for sample in load_indexed_split(args.data_yaml, "train"):
        if not sample.label_path.is_file():
            continue
        for line in sample.label_path.read_text(encoding="utf-8-sig").splitlines():
            parts = line.split()
            if len(parts) < 7 or int(float(parts[0])) != CLASS_ID:
                continue
            coords = [(float(parts[i]), float(parts[i + 1])) for i in range(1, len(parts) - 1, 2)]
            features = _instance_features(sample.image_path, coords)
            features.update({"image": str(sample.image_path), "split": "train"})
            train_rows.append(features)
    print(f"训练集类 13 实例：{len(train_rows)} 个")

    # ---------- 漏检实例 ----------
    missed_rows, overlap_rows = [], []
    for sample in samples:
        width, height = sample["image_width"], sample["image_height"]
        pts_list = []
        for gt in sample.get("ground_truth", []):
            if gt.get("polygon"):
                pts_list.append((gt["class_id"], _coords(gt["polygon"])))
        # GT 是归一化坐标，预测是像素坐标：匹配/重叠必须在同一坐标系里算
        pts_px_list = [
            (cls, [(x * width, y * height) for x, y in pts]) for cls, pts in pts_list
        ]
        for gt_index, (cls, pts) in enumerate(pts_list):
            if cls != CLASS_ID:
                continue
            pts_px = pts_px_list[gt_index][1]
            matched = False
            for prediction in sample.get("predictions", []):
                pp = prediction.get("mask_polygon") or prediction.get("obb_points")
                if not pp:
                    continue
                if _iou(pts_px, _coords(pp)) >= 0.5:
                    matched = True
                    break
            if matched:
                continue
            path = Path(sample["image_path"])
            features = _instance_features(path, pts)
            features.update({"image": str(path), "gt_index": gt_index, "split": "val", "sheet": path.name})
            missed_rows.append(features)

            best_iou, best_cls = 0.0, ""
            for other_index, (other_cls, other_px) in enumerate(pts_px_list):
                if other_index == gt_index:
                    continue
                value = _iou(pts_px, other_px)
                if value > best_iou:
                    best_iou, best_cls = value, other_cls
            overlap_rows.append(
                {
                    "image": str(path),
                    "gt_index": gt_index,
                    "max_overlap_with_other_class": round(best_iou, 4),
                    "other_class_id": best_cls,
                    "overlap_gt_0.3": best_iou > 0.3,
                }
            )
    print(f"漏检实例：{len(missed_rows)} 个")

    # ---------- 最近邻距离 ----------
    def feature_vector(row):
        return np.array(
            [
                math.log(max(row["area_px"], 1.0)),
                math.log(max(row["aspect"], 1.0)),
                row["brightness"] if not math.isnan(row["brightness"]) else 0.0,
                row["edge_density"] if not math.isnan(row["edge_density"]) else 0.0,
            ]
        )

    if train_rows:
        scale = np.std([feature_vector(r) for r in train_rows], axis=0)
        scale[scale == 0] = 1.0

        def normalized(vector):
            return vector / scale

        train_vectors = [normalized(feature_vector(r)) for r in train_rows]
        for row in missed_rows:
            vector = normalized(feature_vector(row))
            distances = [float(np.linalg.norm(vector - t)) for t in train_vectors]
            row["nearest_train_distance"] = round(min(distances), 3)
            row["nearest_train_median_distance"] = round(float(np.median(distances)), 3)
    else:
        for row in missed_rows:
            row["nearest_train_distance"] = None
            row["nearest_train_median_distance"] = None

    def write_csv(path, rows):
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(out / "class13_train_distribution.csv", train_rows)
    write_csv(out / "class13_missed_profile.csv", missed_rows)
    write_csv(out / "class13_overlap_check.csv", overlap_rows)

    # ---------- 汇总 ----------
    def summary(rows, key):
        values = [r[key] for r in rows if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])]
        if not values:
            return {}
        values = sorted(values)
        return {
            "n": len(values),
            "min": round(values[0], 2),
            "p25": round(values[len(values) // 4], 2),
            "median": round(values[len(values) // 2], 2),
            "p75": round(values[(3 * len(values)) // 4], 2),
            "max": round(values[-1], 2),
            "mean": round(sum(values) / len(values), 2),
        }

    result = {
        "train_instances": len(train_rows),
        "missed_instances": len(missed_rows),
        "train": {k: summary(train_rows, k) for k in ("area_px", "aspect", "fill_ratio", "brightness", "edge_density", "area_ratio")},
        "missed": {k: summary(missed_rows, k) for k in ("area_px", "aspect", "fill_ratio", "brightness", "edge_density", "area_ratio")},
        "overlap": {
            "n": len(overlap_rows),
            "with_other_class_gt_over_0.3": sum(1 for r in overlap_rows if r["overlap_gt_0.3"]),
            "max": max((r["max_overlap_with_other_class"] for r in overlap_rows), default=0),
        },
        "nearest_train_distance": summary(missed_rows, "nearest_train_distance"),
    }
    (out / "class13_missed_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- 图版 ----------
    cols, cell_w, cell_h, caption_h = 4, 300, 220, 22
    rows_per_sheet = cols * 2
    for start in range(0, len(missed_rows), rows_per_sheet):
        chunk = missed_rows[start : start + rows_per_sheet]
        sheet = Image.new("RGB", (cols * cell_w, (cell_h + caption_h) * 2), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for slot, row in enumerate(chunk):
            with Image.open(row["image"]) as image:
                full = image.convert("RGB")
                width, height = full.size
                thumb = full.copy()
                thumb.thumbnail((cell_w - 4, cell_h - 4))
                origin_x = (slot % cols) * cell_w
                origin_y = (slot // cols) * (cell_h + caption_h)
                sheet.paste(thumb, (origin_x + 2, origin_y + 2))
                draw.text(
                    (origin_x + 4, origin_y + cell_h + 3),
                    f"{row['sheet'][:26]} area={row['area_px']:.0f}",
                    fill=(20, 20, 20),
                )
        sheet.save(out / "sheets" / f"class13_missed_{start // rows_per_sheet + 1:02d}.jpg", quality=90)

    print(json.dumps({k: result[k] for k in ("train_instances", "missed_instances", "overlap")}, ensure_ascii=False, indent=2))
    print(f"out: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
