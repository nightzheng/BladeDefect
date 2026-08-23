"""Size-stratified analysis with real polygon area and polygon IoU.

Week 6: replace bbox-area proxy with true polygon area (shoelace) and
raster-based polygon IoU (ray-casting + grid sampling). bbox helpers kept
for backward compatibility.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

SMALL = 32 * 32
MEDIUM = 96 * 96


def area_of(bbox):
    """Bounding-box area proxy (legacy)."""
    if not bbox or len(bbox) != 4:
        return 0.0
    return abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])


def size_bucket(bbox) -> str:
    """Size bucket from bbox area (legacy)."""
    return size_bucket_from_area(area_of(bbox))


def polygon_area(poly) -> float:
    """Shoelace polygon area. poly is (N,2) array."""
    p = np.asarray(poly, dtype=float)
    if len(p) < 3:
        return 0.0
    x = p[:, 0]; y = p[:, 1]
    return 0.5 * abs(float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))))


def size_bucket_from_area(area: float) -> str:
    if area < SMALL:
        return "small"
    if area < MEDIUM:
        return "medium"
    return "large"


def points_in_polygon(points, poly):
    """Vectorized ray-casting point-in-polygon. points (M,2), poly (N,2)."""
    p = np.asarray(poly, dtype=float)
    x = points[:, 0]; y = points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    n = len(p)
    for i in range(n):
        x1, y1 = p[i]; x2, y2 = p[(i + 1) % n]
        y_ok = (y1 > y) != (y2 > y)
        if not np.any(y_ok):
            continue
        denom = y2 - y1
        with np.errstate(divide="ignore", invalid="ignore"):
            x_int = x1 + (y - y1) * (x2 - x1) / denom
        inside ^= y_ok & (x < x_int)
    return inside


def raster_iou(poly_a, poly_b, grid: int = 60) -> float:
    """Raster-based polygon IoU (robust to non-convex/self-intersecting)."""
    a = np.asarray(poly_a, dtype=float); b = np.asarray(poly_b, dtype=float)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    allpts = np.vstack([a, b])
    minx, miny = allpts.min(axis=0); maxx, maxy = allpts.max(axis=0)
    step = max(maxx - minx, maxy - miny) / float(grid)
    step = max(step, 1.0)
    xs = np.arange(minx, maxx + step, step)
    ys = np.arange(miny, maxy + step, step)
    if len(xs) == 0 or len(ys) == 0:
        return 0.0
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    in_a = points_in_polygon(pts, a)
    in_b = points_in_polygon(pts, b)
    inter = int((in_a & in_b).sum()); union = int((in_a | in_b).sum())
    return inter / union if union > 0 else 0.0


def polygon_size_stratified(predictions, grid: int = 60):
    """Per-size GT/matched counts using real GT polygon area and raster IoU."""
    size_gt = Counter(); size_match = Counter()
    for s in predictions.get("samples", []):
        w = s.get("image_width", 0); h = s.get("image_height", 0)
        if w <= 0 or h <= 0:
            continue
        gt_polys = []
        for g in s.get("ground_truth", []):
            pts = np.asarray(g.get("polygon", []), dtype=float).reshape(-1, 2)
            pts[:, 0] *= w; pts[:, 1] *= h
            gt_polys.append(pts)
        pred_polys = [np.asarray(p.get("mask_polygon", []), dtype=float)
                      for p in s.get("predictions", [])]
        tids = [g.get("class_id") for g in s.get("ground_truth", [])]
        pids = [p.get("class_id") for p in s.get("predictions", [])]
        for gpoly in gt_polys:
            size_gt[size_bucket_from_area(polygon_area(gpoly))] += 1
        for i, gpoly in enumerate(gt_polys):
            best = 0.0
            for j, ppoly in enumerate(pred_polys):
                if pids[j] == tids[i]:
                    best = max(best, raster_iou(gpoly, ppoly, grid))
            if best > 0.05:
                size_match[size_bucket_from_area(polygon_area(gpoly))] += 1
    rows = []
    for b in ("small", "medium", "large"):
        gt = size_gt.get(b, 0); m = size_match.get(b, 0)
        rows.append({"size": b, "gt_count": gt, "matched_count": m,
                     "recall": round(m / gt, 4) if gt else 0.0,
                     "note": "real GT polygon area + raster IoU"})
    return rows


def size_stratified(predictions, image_index=None):
    """Legacy bbox-area stratification (kept for backward compatibility)."""
    counts = Counter()
    for s in predictions.get("samples", []):
        tid = set(s.get("true_classes", []))
        for p in s.get("predictions", []):
            b = size_bucket(p.get("bbox", []))
            counts[(b, "pred")] += 1
            if p["class_id"] in tid:
                counts[(b, "matched")] += 1
    rows = []
    for b in ("small", "medium", "large"):
        rows.append({"size": b, "pred_count": counts[(b, "pred")],
                     "matched_count": counts[(b, "matched")],
                     "note": "bbox area bucket; GT denominator requires image-index"})
    return rows


__all__ = ["SMALL", "MEDIUM", "area_of", "size_bucket", "polygon_area",
           "size_bucket_from_area", "points_in_polygon", "raster_iou",
           "polygon_size_stratified", "size_stratified"]
