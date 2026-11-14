"""小样本类标注审计：把可疑标注按优先级排出来，供人工逐张核对。

审计不是"自动判定标注对错"——类别边界算不算缺陷只有人（和企业工程师）能定。
这里做的事情是：把 647 条实例（类 10/13/14）里**最可能有问题的那几十条**排到前面，
并给出每一类的问题结构，让 2 小时的人工核对花在刀刃上。

检查项与默认权重（可调）：
  ZERO_AREA       多边形退化为零面积          +5
  OUT_OF_BOUNDS   坐标越界（<0 或 >1）        +4
  DUPLICATE       同类实例互相重叠 IoU>=0.85   +3
  TINY            像素面积 < 32x32            +3
  THIN            包围盒最短边 < 8px          +2
  CONFUSABLE      同图出现易混类              +2
  NO_LABEL_FILE   图有记录但标签文件缺失      +2   （仅记录，不给实例行）

易混类默认表来自 Round 1/2 的混淆矩阵（top pairs），可用 --confusable 覆盖：
  10 维修痕迹        <-> 7 表面油污
  13 结构损伤        <-> 11 玻纤损伤 / 12 叶片开裂
  14 接闪器脱落      <-> 11 玻纤损伤
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from blade_defect.data.class_copy_paste import (
    LabelInstance,
    polygon_area,
    polygon_bbox,
    polygon_iou,
    read_label_file,
)
from blade_defect.data.indexed_splits import IndexedSample

DEFAULT_CONFUSABLE: dict[int, tuple[int, ...]] = {
    10: (7,),
    13: (11, 12),
    14: (11,),
}

DEFAULT_WEIGHTS: dict[str, float] = {
    "ZERO_AREA": 5.0,
    "OUT_OF_BOUNDS": 4.0,
    "DUPLICATE": 3.0,
    "TINY": 3.0,
    "THIN": 2.0,
    "CONFUSABLE": 2.0,
    "NO_LABEL_FILE": 2.0,
}

TINY_AREA_PX = 32 * 32
THIN_SIDE_PX = 8
DUPLICATE_IOU = 0.85


@dataclass
class AuditRow:
    """一条待人工核对的实例（或图片级问题）。"""

    split: str
    class_id: int
    class_name: str
    image_path: str
    label_path: str
    instance_index: int
    vertices: int
    area_px: float
    bbox_w_px: float
    bbox_h_px: float
    aspect: float
    co_occurring: str
    flags: list[str]
    priority: float
    review_verdict: str = ""
    reviewer_note: str = ""


def _image_size(path: Path) -> tuple[int, int] | None:
    from PIL import Image

    try:
        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _co_occurring(
    instances: Sequence[LabelInstance], class_id: int, confusable: dict[int, tuple[int, ...]]
) -> list[int]:
    neighbours = set(confusable.get(class_id, ()))
    return sorted({inst.class_id for inst in instances if inst.class_id in neighbours})


def audit_split(
    samples: Iterable[IndexedSample],
    target_classes: Sequence[int],
    names: dict[int, str],
    *,
    split: str = "train",
    confusable: dict[int, tuple[int, ...]] | None = DEFAULT_CONFUSABLE,
    weights: dict[str, float] | None = DEFAULT_WEIGHTS,
) -> list[AuditRow]:
    """扫描一个 split，返回按优先级降序排列的待核对行。"""
    confusable = DEFAULT_CONFUSABLE if confusable is None else confusable
    weights = DEFAULT_WEIGHTS if weights is None else weights
    targets = set(int(value) for value in target_classes)
    rows: list[AuditRow] = []

    for sample in samples:
        if not sample.label_path.is_file():
            if sample.image_path.is_file():
                rows.append(
                    AuditRow(
                        split=split,
                        class_id=-1,
                        class_name="(图片级)",
                        image_path=str(sample.image_path),
                        label_path=str(sample.label_path),
                        instance_index=-1,
                        vertices=0,
                        area_px=0.0,
                        bbox_w_px=0.0,
                        bbox_h_px=0.0,
                        aspect=0.0,
                        co_occurring="",
                        flags=["NO_LABEL_FILE"],
                        priority=weights["NO_LABEL_FILE"],
                    )
                )
            continue

        instances = read_label_file(sample.label_path)
        size = _image_size(sample.image_path)
        width, height = size if size else (0, 0)

        for index, instance in enumerate(instances):
            if instance.class_id not in targets:
                continue

            flags: list[str] = []
            rel_area = polygon_area(instance.polygon)
            area_px = rel_area * width * height if width and height else 0.0
            x1, y1, x2, y2 = polygon_bbox(instance.polygon)
            bw = (x2 - x1) * width if width else 0.0
            bh = (y2 - y1) * height if height else 0.0
            aspect = (max(bw, bh) / max(min(bw, bh), 1e-6)) if bw and bh else 0.0

            if rel_area <= 1e-9:
                flags.append("ZERO_AREA")
            if instance.polygon.min() < 0 or instance.polygon.max() > 1:
                flags.append("OUT_OF_BOUNDS")
            if area_px and area_px < TINY_AREA_PX:
                flags.append("TINY")
            if min(bw, bh) and min(bw, bh) < THIN_SIDE_PX:
                flags.append("THIN")

            for other in instances:
                if other is instance or other.class_id != instance.class_id:
                    continue
                if polygon_iou(instance.polygon, other.polygon) >= DUPLICATE_IOU:
                    flags.append("DUPLICATE")
                    break

            co = _co_occurring(instances, instance.class_id, confusable)
            if co:
                flags.append("CONFUSABLE")

            priority = float(sum(weights.get(flag, 0.0) for flag in flags))

            rows.append(
                AuditRow(
                    split=split,
                    class_id=int(instance.class_id),
                    class_name=names.get(int(instance.class_id), str(instance.class_id)),
                    image_path=str(sample.image_path),
                    label_path=str(sample.label_path),
                    instance_index=index,
                    vertices=int(instance.polygon.shape[0]),
                    area_px=round(float(area_px), 1),
                    bbox_w_px=round(float(bw), 1),
                    bbox_h_px=round(float(bh), 1),
                    aspect=round(float(aspect), 2),
                    co_occurring=",".join(names.get(value, str(value)) for value in co),
                    flags=sorted(set(flags)),
                    priority=priority,
                )
            )

    rows.sort(key=lambda row: (-row.priority, row.class_id, row.image_path, row.instance_index))
    return rows


def summarize(rows: Sequence[AuditRow]) -> dict[str, Any]:
    """按类别统计各 flag 的命中数，给出审计工作量的结构画像。"""
    by_class: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.class_id)
        bucket = by_class.setdefault(
            key, {"class_name": row.class_name, "instances": 0, "flagged": 0, "flags": {}}
        )
        bucket["instances"] += 1
        if row.flags:
            bucket["flagged"] += 1
        for flag in row.flags:
            bucket["flags"][flag] = bucket["flags"].get(flag, 0) + 1
    return {
        "total_rows": len(rows),
        "flagged_rows": sum(1 for row in rows if row.flags),
        "by_class": by_class,
    }


__all__ = [
    "AuditRow",
    "DEFAULT_CONFUSABLE",
    "DEFAULT_WEIGHTS",
    "audit_split",
    "summarize",
]
