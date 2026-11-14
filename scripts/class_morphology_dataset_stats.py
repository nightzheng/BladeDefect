#!/usr/bin/env python
"""全数据集按类别的形态分布统计（回答"细长型是不是只有类 13 的问题"）。

对 train+val 全部实例计算：面积(px²)、占画面比、长宽比、填充率、顶点数，
按类别汇总，并给出"长宽比>2 / 面积>500k"的占比——用来判断形态长尾是局部问题还是全局问题。

用法：
  python scripts/class_morphology_dataset_stats.py \
    --data-yaml <原 data.yaml> \
    --out-csv /data/backup/round3/morphology_instances.csv \
    --out-json /data/backup/round3/morphology_by_class.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.data.indexed_splits import load_indexed_split, load_index_config  # noqa: E402


def _shoelace(pts):
    area = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _stats(values):
    if not values:
        return {}
    values = sorted(values)
    return {
        "n": len(values),
        "min": round(values[0], 2),
        "p50": round(values[len(values) // 2], 2),
        "p90": round(values[int(len(values) * 0.9)], 2),
        "max": round(values[-1], 2),
    }


def main() -> int:
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    _, config = load_index_config(args.data_yaml)
    raw_names = config.get("names") or {}
    names = (
        {int(k): str(v) for k, v in raw_names.items()}
        if isinstance(raw_names, dict)
        else {i: str(v) for i, v in enumerate(raw_names)}
    )

    rows = []
    size_cache: dict[str, tuple[int, int]] = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        samples = load_indexed_split(args.data_yaml, split)
        print(f"[{split}] {len(samples)} samples", flush=True)
        for sample in samples:
            label_path = sample.label_path
            if not label_path.is_file():
                continue
            lines = [ln.split() for ln in label_path.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
            if not lines:
                continue
            key = str(sample.image_path)
            if key not in size_cache:
                try:
                    with Image.open(sample.image_path) as image:
                        size_cache[key] = image.size
                except Exception:
                    continue
            width, height = size_cache[key]
            for parts in lines:
                if len(parts) < 7:
                    continue
                class_id = int(float(parts[0]))
                pts = [(float(parts[i]) * width, float(parts[i + 1]) * height) for i in range(1, len(parts) - 1, 2)]
                if len(pts) < 3:
                    continue
                area = _shoelace(pts)
                if area <= 0:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                bw = max(max(xs) - min(xs), 1e-6)
                bh = max(max(ys) - min(ys), 1e-6)
                rows.append(
                    {
                        "split": split,
                        "class_id": class_id,
                        "class_name": names.get(class_id, str(class_id)),
                        "image": key,
                        "area_px": round(area, 1),
                        "area_ratio": round(area / (width * height), 6),
                        "aspect": round(max(bw, bh) / min(bw, bh), 3),
                        "fill_ratio": round(area / (bw * bh), 3),
                        "vertices": len(pts),
                    }
                )
    print(f"总实例: {len(rows)}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    by_class: dict[str, dict] = {}
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["class_id"]].append(row)
    for class_id, items in sorted(grouped.items()):
        aspects = [r["aspect"] for r in items]
        areas = [r["area_px"] for r in items]
        fills = [r["fill_ratio"] for r in items]
        by_class[str(class_id)] = {
            "class_name": items[0]["class_name"],
            "instances": len(items),
            "area_px": _stats(areas),
            "aspect": _stats(aspects),
            "fill_ratio": _stats(fills),
            "pct_aspect_gt2": round(100 * sum(1 for v in aspects if v > 2) / len(aspects), 1),
            "pct_aspect_gt5": round(100 * sum(1 for v in aspects if v > 5) / len(aspects), 1),
            "pct_area_gt500k": round(100 * sum(1 for v in areas if v > 500_000) / len(areas), 1),
        }
    Path(args.out_json).write_text(json.dumps({"total_instances": len(rows), "by_class": by_class}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'类':<4}{'实例数':>7}{'面积p50':>12}{'长宽比p50':>10}{'aspect>2':>10}{'aspect>5':>10}{'>500k':>8}  类名")
    for class_id in sorted(grouped):
        info = by_class[str(class_id)]
        print(
            f"{class_id:<4}{info['instances']:>7}{info['area_px']['p50']:>12,.0f}{info['aspect']['p50']:>10.2f}"
            f"{info['pct_aspect_gt2']:>9.1f}%{info['pct_aspect_gt5']:>9.1f}%{info['pct_area_gt500k']:>7.1f}%  {info['class_name']}"
        )
    print(f"\ncsv:  {out_csv}\njson: {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
