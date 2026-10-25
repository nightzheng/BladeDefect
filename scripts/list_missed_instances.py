#!/usr/bin/env python
"""列出指定类别的漏检实例（GT 未被任何预测以 polygon IoU>=0.5 匹配）。

匹配逻辑与 scripts/decompose_val_errors.py 完全一致（贪心、IoU>=0.5），
只多输出逐实例明细：图片、GT 序号、GT 面积、最近预测的类别与 IoU，
用于"这 19 个漏检到底长什么样、为什么没检出来"的人工反查。

用法：
  python scripts/list_missed_instances.py \
    --predictions /data/runs/v3_seg_cp_class13_960_e25/validation_predictions.json \
    --class-id 13 --out-csv /data/backup/round3/missed_class13_b1.csv \
    --out-json /data/backup/round3/missed_class13_b1.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

IOU_MATCH = 0.5


def _poly(points, width: float = 1.0, height: float = 1.0, normalized: bool = False):
    """polygon 可能是 [[x,y],...] 或扁平 [x1,y1,x2,y2,...]，两种都要接受（与 decompose 一致）。"""
    from shapely.geometry import Polygon

    if points and isinstance(points[0], (list, tuple)):
        coords = [(float(p[0]), float(p[1])) for p in points]
    else:
        coords = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points) - 1, 2)]
    if normalized:
        coords = [(x * width, y * height) for x, y in coords]
    if len(coords) < 3:
        return Polygon()
    polygon = Polygon(coords)
    return polygon if polygon.is_valid else polygon.buffer(0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--class-id", type=int, default=13)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    rows: list[dict] = []
    total_gt = 0
    for sample in payload["samples"]:
        width, height = sample["image_width"], sample["image_height"]
        gts = [
            (g["class_id"], _poly(g["polygon"], width, height, normalized=True))
            for g in sample.get("ground_truth", [])
            if g.get("polygon")
        ]
        preds = []
        for prediction in sample.get("predictions", []):
            points = prediction.get("mask_polygon") or prediction.get("obb_points")
            if points:
                preds.append((prediction["class_id"], _poly(points)))

        target_idx = [i for i, (cls, poly) in enumerate(gts) if cls == args.class_id and poly.area > 0]
        total_gt += len(target_idx)
        if not target_idx:
            continue

        pairs = []
        for gi in target_idx:
            gp = gts[gi][1]
            for pi, (_, pp) in enumerate(preds):
                if pp.area == 0:
                    continue
                inter = gp.intersection(pp).area
                if inter > 0:
                    iou = inter / (gp.area + pp.area - inter)
                    if iou >= IOU_MATCH:
                        pairs.append((iou, gi, pi))
        pairs.sort(reverse=True)
        used_g, used_p = set(), set()
        for iou, gi, pi in pairs:
            if gi in used_g or pi in used_p:
                continue
            used_g.add(gi)
            used_p.add(pi)

        for gi in target_idx:
            if gi in used_g:
                continue
            gp = gts[gi][1]
            mins = gp.bounds
            best_iou, best_class = 0.0, None
            for pc, pp in preds:
                if pp.area == 0:
                    continue
                inter = gp.intersection(pp).area
                if inter <= 0:
                    continue
                iou = inter / (gp.area + pp.area - inter)
                if iou > best_iou:
                    best_iou, best_class = iou, pc
            rows.append(
                {
                    "image": sample["image_path"],
                    "gt_index": gi,
                    "gt_area_px": round(gp.area, 1),
                    "bbox_xyxy_px": ",".join(f"{v:.1f}" for v in mins),
                    "n_predictions_in_image": len(preds),
                    "nearest_pred_class": "" if best_class is None else best_class,
                    "nearest_pred_iou": round(best_iou, 4),
                }
            )

    rows.sort(key=lambda r: (r["image"], r["gt_index"]))
    csv_path = Path(args.out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else ["image"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "predictions": args.predictions,
        "class_id": args.class_id,
        "gt_instances": total_gt,
        "missed_instances": len(rows),
        "missed_rate": round(len(rows) / total_gt, 4) if total_gt else None,
        "ioU_match_threshold": IOU_MATCH,
        "rows": rows,
    }
    Path(args.out_json).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("class_id", "gt_instances", "missed_instances", "missed_rate")}, ensure_ascii=False, indent=2))
    print(f"csv:  {csv_path}")
    print(f"json: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
