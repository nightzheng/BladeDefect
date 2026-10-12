#!/usr/bin/env python
"""按 GT 实例做错误分解：未达匹配阈值（漏检）/ 误分类 / 定位偏差 / 正确。

对 export_validation_predictions / export_obb_val_predictions 导出的全量 val
预测 JSON，逐 GT 实例用 polygon IoU（>=0.5 匹配阈值，conf 为导出阈值）贪心匹配：
  - ok            ：IoU>=0.75 且类别正确
  - mislocalized  ：0.5<=IoU<0.75 且类别正确（"类对但框不够准"）
  - misclassified ：IoU>=0.5 但类别错误（"看见但分错类"）
  - missed        ：无 IoU>=0.5 的预测（未达匹配阈值；注意这是阈值口径，
                    不代表模型完全无响应——评审意见 2026-10-10）
另统计 false_positive（无任何 GT 匹配的预测）。
口径为近似分析，非官方 AP；官方指标以 metrics.json 为准。
保留归档边界口径：缺失/空几何不参与统计；零面积 GT 不进入分母，
有几何字段但零面积的预测计 FP；自交 polygon 沿用 buffer(0) 修复。
坐标格式错误、少于三点或非有限坐标明确报错，避免静默生成不完整统计。
无有效 GT 时摘要百分比显示 N/A，仍输出 JSON（含 FP）。

用法：
  python scripts/decompose_val_errors.py out.json seg=runs/exp/validation_predictions.json obb=...
  python scripts/decompose_val_errors.py out.json exp2=path.json exp3=path.json ...
输出：JSON（per-class 计数 + top 混淆对 + 误报计数），stdout 打印总体摘要。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from math import isfinite
from pathlib import Path

from shapely.geometry import Polygon

IOU_MATCH = 0.5
IOU_GOOD = 0.75


def _poly(points, w=1.0, h=1.0, normalized=False):
    if not points:
        return Polygon()
    try:
        if isinstance(points[0], (list, tuple)):
            if any(len(p) != 2 for p in points):
                raise ValueError("expected coordinate pairs")
            pts = [(float(p[0]), float(p[1])) for p in points]
        else:
            if len(points) % 2:
                raise ValueError("expected an even coordinate count")
            pts = [(float(points[i]), float(points[i + 1])) for i in range(0, len(points), 2)]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("Invalid polygon: expected numeric x/y coordinate pairs") from exc
    if normalized:
        pts = [(x * w, y * h) for x, y in pts]
    if len(pts) < 3 or not all(isfinite(x) and isfinite(y) for x, y in pts):
        raise ValueError("Invalid polygon: expected at least three points with finite coordinates")
    p = Polygon(pts)
    return p if p.is_valid else p.buffer(0)


def decompose(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    stats = defaultdict(Counter)
    conf_pairs = Counter()
    fp = Counter()
    for s in d["samples"]:
        w, h = s["image_width"], s["image_height"]
        gts, preds = [], []
        for g in s.get("ground_truth", []):
            if g.get("polygon"):
                gts.append((g["class_id"], _poly(g["polygon"], w, h, normalized=True)))
        for pr in s.get("predictions", []):
            pts = pr.get("mask_polygon") or pr.get("obb_points")
            if pts:
                preds.append((pr["class_id"], _poly(pts)))
        pairs = []
        for gi, (_, gp) in enumerate(gts):
            if gp.area == 0:
                continue
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
            gc, pc = gts[gi][0], preds[pi][0]
            if gc == pc:
                stats[gc]["ok" if iou >= IOU_GOOD else "mislocalized"] += 1
            else:
                stats[gc]["misclassified"] += 1
                conf_pairs[(gc, pc)] += 1
        for gi, (gc, gp) in enumerate(gts):
            if gi not in used_g and gp.area > 0:
                stats[gc]["missed"] += 1
        for pi, (pc, _) in enumerate(preds):
            if pi not in used_p:
                fp[pc] += 1
    return {
        "per_class": {str(k): dict(v) for k, v in sorted(stats.items())},
        "confusion_pairs_top30": {f"{a}>{b}": n for (a, b), n in conf_pairs.most_common(30)},
        "false_positives": {str(k): n for k, n in fp.most_common()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("output", type=Path, help="结果 JSON 输出路径")
    ap.add_argument("models", nargs="+", help="name=validation_predictions.json 路径，可多个")
    args = ap.parse_args()
    out = {}
    for spec in args.models:
        name, _, p = spec.partition("=")
        result = decompose(Path(p))
        out[name] = result
        tot = Counter()
        for v in result["per_class"].values():
            tot.update(v)
        n = sum(tot.values())
        rates = {
            key: f"{tot[key]/n:.1%}" if n else "N/A"
            for key in ("missed", "misclassified", "mislocalized", "ok")
        }
        print(
            f"{name}: GT={n} 未达阈值(missed)={tot['missed']}({rates['missed']}) "
            f"误分类={tot['misclassified']}({rates['misclassified']}) "
            f"定位偏差={tot['mislocalized']}({rates['mislocalized']}) "
            f"ok={tot['ok']}({rates['ok']})",
            flush=True,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
