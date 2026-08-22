"""Size-stratified recall/failure analysis for bbox area buckets."""
from __future__ import annotations
from collections import Counter

SMALL = 32 * 32
MEDIUM = 96 * 96

def area_of(bbox):
    if not bbox or len(bbox) != 4: return 0.0
    return abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1])

def size_bucket(bbox) -> str:
    a = area_of(bbox)
    if a < SMALL: return "small"
    if a < MEDIUM: return "medium"
    return "large"

def size_stratified(predictions, image_index=None):
    """Compute per-size predicted/matched counts.
    image_index (optional) provides GT size denominator per image.
    """
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

__all__ = ["size_bucket", "size_stratified"]
