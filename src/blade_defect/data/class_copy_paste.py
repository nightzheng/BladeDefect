"""按类定向 copy-paste：离线合成少数类训练样本（不侵入 ultralytics dataloader）。

为什么不用 ultralytics 原生 ``copy_paste``：
1. 它是**全类生效**，无法按类筛选（方案 B 的核心诉求就是"只补少数类"）；
2. 它的实现位于 dataloader 变换链内部，按类筛选必须子类化/替换内部变换，
   而 Round 2 已经因为 dataloader 侧并发（``labels/train.cache`` 竞态）翻过车，
   不想再往那条路上加风险。

因此改为**离线合成**：从源图裁出目标类实例的像素与多边形，随机缩放/位移后贴到
目标图上，写出新的 image/label 对，并把合成样本追加进 train 索引。好处：

- **可复现**：随机性只来自 ``seed`` 与稳定排序；
- **可审计**：``manifest.json`` 逐条记录源图/目标图/类别/缩放/落点；
- **零侵入**：训练侧只是"train 多了一批图"，与既有 preflight、数据门禁、
  缓存机制完全兼容，不存在 dataloader 回调不稳定问题；
- **不污染验证集**：val 索引原样引用，合成样本只进 train。

标签格式遵循 YOLO segmentation：每行 ``cls x1 y1 x2 y2 ...``，坐标归一化到 [0,1]。
本模块对单实例只输出**一个多边形**（不拆洞、不合并多段），与数据集现状一致。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from blade_defect.data.indexed_splits import (
    IndexedSample,
    load_index_config,
    load_indexed_split,
)
from blade_defect.utils.files import save_json


# --------------------------------------------------------------------------- #
# 标签读写
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LabelInstance:
    """一条 YOLO-seg 标注实例（单多边形，坐标已归一化）。"""

    class_id: int
    polygon: np.ndarray  # (N, 2) float64，取值 [0, 1]

    def __post_init__(self) -> None:
        polygon = np.asarray(self.polygon, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2:
            raise ValueError(f"多边形至少需要 3 个点，收到 shape={polygon.shape}")
        object.__setattr__(self, "polygon", polygon)


def parse_label_line(line: str) -> LabelInstance | None:
    """解析一行 YOLO-seg 标注；空行返回 ``None``。"""
    text = line.strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        raise ValueError(f"标注行点数不合法（需要 cls + 成对坐标）：{text[:60]}")
    class_id = int(float(parts[0]))
    coords = np.asarray([float(value) for value in parts[1:]], dtype=np.float64)
    return LabelInstance(class_id=class_id, polygon=coords.reshape(-1, 2))


def read_label_file(path: str | Path) -> list[LabelInstance]:
    """读取一个标签文件；文件不存在视为空标注（背景图）。"""
    label_path = Path(path)
    if not label_path.is_file():
        return []
    instances: list[LabelInstance] = []
    for line in label_path.read_text(encoding="utf-8-sig").splitlines():
        instance = parse_label_line(line)
        if instance is not None:
            instances.append(instance)
    return instances


def format_label_line(instance: LabelInstance) -> str:
    """序列化为 YOLO-seg 一行；坐标裁剪到 [0, 1] 并保留 6 位小数。"""
    clipped = np.clip(instance.polygon, 0.0, 1.0)
    coords = " ".join(f"{value:.6f}" for value in clipped.reshape(-1))
    return f"{int(instance.class_id)} {coords}"


def write_label_file(path: str | Path, instances: Sequence[LabelInstance]) -> None:
    """写出标签文件（父目录自动创建）。"""
    label_path = Path(path)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [format_label_line(instance) for instance in instances]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 几何
# --------------------------------------------------------------------------- #
def polygon_bbox(polygon: np.ndarray) -> tuple[float, float, float, float]:
    """归一化多边形的 (x_min, y_min, x_max, y_max)。"""
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def polygon_area(polygon: np.ndarray) -> float:
    """鞋带公式面积（归一化坐标下为相对面积）。"""
    x = polygon[:, 0]
    y = polygon[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_iou(left: np.ndarray, right: np.ndarray) -> float:
    """用栅格近似计算两个归一化多边形的 IoU（用于查重，不要求高精度）。"""
    from PIL import Image, ImageDraw

    size = 128
    canvas = []
    for polygon in (left, right):
        image = Image.new("L", (size, size), 0)
        ImageDraw.Draw(image).polygon(
            [(float(x) * (size - 1), float(y) * (size - 1)) for x, y in polygon], fill=1
        )
        canvas.append(np.asarray(image, dtype=bool))
    union = np.logical_or(canvas[0], canvas[1]).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(canvas[0], canvas[1]).sum() / union)


def _polygon_mask(polygon_px: np.ndarray, width: int, height: int) -> np.ndarray:
    """在给定画布上栅格化多边形，返回 bool 掩膜。"""
    from PIL import Image, ImageDraw

    image = Image.new("L", (width, height), 0)
    ImageDraw.Draw(image).polygon([(float(x), float(y)) for x, y in polygon_px], fill=1)
    return np.asarray(image, dtype=bool)


# --------------------------------------------------------------------------- #
# 合成
# --------------------------------------------------------------------------- #
@dataclass
class PasteRecord:
    """一条合成的审计记录。"""

    out_image: str
    dst_image: str
    src_image: str
    class_id: int
    scale: float
    offset_px: tuple[int, int]
    area_px: float
    anchor_class_id: int = -1


@dataclass
class SynthesizeConfig:
    """合成任务参数（全部显式，便于复现与报备）。"""

    data_yaml: str | Path
    out_dir: str | Path
    target_classes: Sequence[int]
    copies_per_instance: int = 3
    seed: int = 42
    scale_range: tuple[float, float] = (0.8, 1.2)
    max_synthetic: int | None = None
    image_quality: int = 92
    split: str = "train"
    border_margin_ratio: float = 0.02


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _save_rgb(path: Path, array: np.ndarray, quality: int) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=quality)


def _target_instances(
    samples: Iterable[IndexedSample], target_classes: set[int]
) -> list[tuple[IndexedSample, LabelInstance]]:
    """挑出携带目标类实例的 (样本, 实例) 列表，顺序稳定。"""
    picked: list[tuple[IndexedSample, LabelInstance]] = []
    for sample in samples:
        for instance in read_label_file(sample.label_path):
            if instance.class_id in target_classes:
                picked.append((sample, instance))
    return picked


def _paste_one(
    dst_rgb: np.ndarray,
    src_rgb: np.ndarray,
    src_polygon: np.ndarray,
    rng: random.Random,
    scale_range: tuple[float, float],
    anchor_norm: tuple[float, float] | None = None,
    margin_ratio: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int]]:
    """把一个源实例贴到目标图上，返回 (新图, 新多边形归一化坐标, 缩放, 像素落点)。"""
    dst_h, dst_w = dst_rgb.shape[:2]
    src_h, src_w = src_rgb.shape[:2]
    poly_px = np.column_stack(
        [src_polygon[:, 0] * (src_w - 1), src_polygon[:, 1] * (src_h - 1)]
    )

    # 源实例在源图上的像素包围盒
    bx1, by1 = int(np.floor(poly_px[:, 0].min())), int(np.floor(poly_px[:, 1].min()))
    bx2, by2 = int(np.ceil(poly_px[:, 0].max())) + 1, int(np.ceil(poly_px[:, 1].max())) + 1
    bx1, by1 = max(bx1, 0), max(by1, 0)
    bx2, by2 = min(bx2, src_w), min(by2, src_h)
    if bx2 - bx1 < 4 or by2 - by1 < 4:
        raise ValueError("实例像素包围盒过小（<4px），跳过")

    scale = rng.uniform(*scale_range)
    new_w = max(int(round((bx2 - bx1) * scale)), 2)
    new_h = max(int(round((by2 - by1) * scale)), 2)
    if new_w > dst_w or new_h > dst_h:
        raise ValueError("缩放后实例大于目标图，跳过")

    # 在裁剪块坐标系里重建多边形
    poly_local = np.column_stack([poly_px[:, 0] - bx1, poly_px[:, 1] - by1])
    mask_local = _polygon_mask(poly_local, bx2 - bx1, by2 - by1)

    from PIL import Image

    patch = Image.fromarray(src_rgb[by1:by2, bx1:bx2])
    mask_image = Image.fromarray((mask_local.astype(np.uint8) * 255))
    patch = patch.resize((new_w, new_h), Image.BILINEAR)
    mask_image = mask_image.resize((new_w, new_h), Image.NEAREST)
    patch_array = np.asarray(patch)
    mask_array = np.asarray(mask_image, dtype=bool)

    # 落点策略（v2）：若给定锚点（目标图某个标注多边形内部的随机点），
    # 就把贴入块的**中心**对到锚点上，并留出边界余量；否则退回全图随机（v1 行为）。
    margin_px = int(round(min(dst_w, dst_h) * margin_ratio))
    max_x = max(dst_w - new_w - margin_px, margin_px)
    max_y = max(dst_h - new_h - margin_px, margin_px)
    if anchor_norm is None:
        offset_x = rng.randint(min(margin_px, max_x), max(max_x, min(margin_px, max_x)))
        offset_y = rng.randint(min(margin_px, max_y), max(max_y, min(margin_px, max_y)))
    else:
        centre_x = int(round(anchor_norm[0] * dst_w - new_w / 2))
        centre_y = int(round(anchor_norm[1] * dst_h - new_h / 2))
        offset_x = int(np.clip(centre_x, margin_px, max_x))
        offset_y = int(np.clip(centre_y, margin_px, max_y))

    out = dst_rgb.copy()
    region = out[offset_y : offset_y + new_h, offset_x : offset_x + new_w]
    region[mask_array] = patch_array[mask_array]

    # 新多边形：裁剪块坐标 -> 缩放 -> 平移到目标图 -> 归一化
    poly_scaled = np.column_stack(
        [poly_local[:, 0] * (new_w / (bx2 - bx1)), poly_local[:, 1] * (new_h / (by2 - by1))]
    )
    poly_dst = np.column_stack([poly_scaled[:, 0] + offset_x, poly_scaled[:, 1] + offset_y])
    poly_norm = np.column_stack([poly_dst[:, 0] / (dst_w - 1), poly_dst[:, 1] / (dst_h - 1)])
    return out, np.clip(poly_norm, 0.0, 1.0), scale, (offset_x, offset_y)


def _sample_anchor(
    dst_instances: list[LabelInstance], image_path: Path, rng: random.Random, tries: int = 40
) -> tuple[float, float, int] | None:
    """在目标图某个标注多边形内部随机取一个锚点（归一化坐标 + 该多边形的类别）。

    为什么必须落在标注内：本数据集的标注全部在叶面上，标注之外基本是天空/草地/水面/立柱。
    v1 用全图随机落点，导致约 8 成样本把叶片碎块贴到背景上（2026-10-23 抽检发现）。
    """
    from PIL import Image, ImageDraw

    if not dst_instances:
        return None
    with Image.open(image_path) as image:
        width, height = image.size
    canvas = 256
    masks: list[np.ndarray] = []
    for instance in dst_instances:
        board = Image.new("L", (canvas, canvas), 0)
        ImageDraw.Draw(board).polygon(
            [(float(x) * (canvas - 1), float(y) * (canvas - 1)) for x, y in instance.polygon], fill=1
        )
        masks.append(np.asarray(board, dtype=bool))

    for _ in range(tries):
        index = rng.randrange(len(dst_instances))
        mask = masks[index]
        ys, xs = np.nonzero(mask)
        if ys.size == 0:
            continue
        pick = rng.randrange(ys.size)
        return (xs[pick] / (canvas - 1), ys[pick] / (canvas - 1), int(dst_instances[index].class_id))
    return None


def synthesize(config: SynthesizeConfig) -> dict[str, Any]:
    """执行合成，返回 manifest（同时写入 ``out_dir/copy_paste_manifest.json``）。"""
    data_yaml = Path(config.data_yaml)
    out_dir = Path(config.out_dir)
    target_classes = sorted({int(value) for value in config.target_classes})
    if not target_classes:
        raise ValueError("target_classes 不能为空")

    config_path, data_config = load_index_config(data_yaml)
    dataset_root = Path(data_config["path"])
    samples = load_indexed_split(data_yaml, config.split)
    picked = _target_instances(samples, set(target_classes))

    rng = random.Random(config.seed)
    out_images_dir = out_dir / "images" / config.split
    out_labels_dir = out_dir / "labels" / config.split
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    records: list[PasteRecord] = []
    per_class: dict[int, int] = {class_id: 0 for class_id in target_classes}
    skipped: list[dict[str, str]] = []

    for sample, instance in picked:
        for copy_index in range(config.copies_per_instance):
            if config.max_synthetic is not None and len(records) >= config.max_synthetic:
                skipped.append({"reason": "max_synthetic_reached"})
                break
            # 目标图必须带标注（标注都在叶面上），否则贴上去就是悬空碎块
            anchor = None
            dst_sample = None
            dst_instances: list[LabelInstance] = []
            for _ in range(12):
                candidate = samples[rng.randrange(len(samples))]
                if candidate.label_path == sample.label_path:
                    continue
                candidate_instances = read_label_file(candidate.label_path)
                if not candidate_instances:
                    continue
                anchor = _sample_anchor(candidate_instances, candidate.image_path, rng)
                if anchor is None:
                    continue
                dst_sample, dst_instances = candidate, candidate_instances
                break
            if dst_sample is None or anchor is None:
                skipped.append({"reason": "no_annotated_destination"})
                continue
            try:
                src_rgb = _load_rgb(sample.image_path)
                dst_rgb = _load_rgb(dst_sample.image_path)
                pasted, polygon, scale, offset = _paste_one(
                    dst_rgb,
                    src_rgb,
                    instance.polygon,
                    rng,
                    config.scale_range,
                    anchor_norm=(anchor[0], anchor[1]),
                    margin_ratio=config.border_margin_ratio,
                )
            except Exception as exc:  # 单条失败不影响整体
                skipped.append({"image": str(sample.image_path), "reason": f"{type(exc).__name__}: {exc}"})
                continue

            stem = f"cp_{sample.image_path.stem}_{instance.class_id}_{copy_index:02d}_{len(records):05d}"
            out_image = out_images_dir / f"{stem}.jpg"
            out_label = out_labels_dir / f"{stem}.txt"

            _save_rgb(out_image, pasted, config.image_quality)
            write_label_file(
                out_label,
                [
                    *dst_instances,
                    LabelInstance(class_id=instance.class_id, polygon=polygon),
                ],
            )

            area_px = polygon_area(polygon) * pasted.shape[0] * pasted.shape[1]
            records.append(
                PasteRecord(
                    out_image=str(out_image),
                    dst_image=str(dst_sample.image_path),
                    src_image=str(sample.image_path),
                    class_id=int(instance.class_id),
                    scale=round(scale, 4),
                    offset_px=offset,
                    area_px=round(float(area_px), 2),
                    anchor_class_id=int(anchor[2]),
                )
            )
            per_class[int(instance.class_id)] += 1

    # train 索引 = 原 train + 合成样本（val 原样引用，不写新文件）
    # train 索引 = 原 train 样本（写成**绝对路径**）+ 合成样本
    # 注意：原索引里的条目是相对路径（../blade-v2/images/train/...），相对的是**原索引所在目录**。
    # 派生索引换了目录，直接抄相对路径会让解析基准改变，33804 张图全部"缺失"（2026-10-23 踩过）。
    train_lines = [str(sample.image_path) for sample in samples]
    train_lines.extend(record.out_image for record in records)
    train_index = out_dir / "train.txt"
    train_index.write_text("\n".join(train_lines) + "\n", encoding="utf-8")

    val_entry = data_config.get("val")
    derived_data_yaml = out_dir / "data.yaml"

    def _resolve_entry(entry: object) -> str:
        """把原 data.yaml 的分割条目解析成绝对路径，保证派生集指回原数据集。"""
        if entry is None:
            return ""
        return str(entry) if Path(str(entry)).is_absolute() else str(Path(config_path).parent / str(entry))

    resolved_val = _resolve_entry(val_entry)
    # test 原样指回原数据集：派生集绝不新增、不改写 test（红线：test 锁定）
    resolved_test = _resolve_entry(data_config.get("test"))
    names = data_config.get("names") or {}
    name_lines = (
        [f"  {index}: {name}" for index, name in names.items()]
        if isinstance(names, dict)
        else [f"  {index}: {name}" for index, name in enumerate(names)]
    )
    derived = {
        "path": str(out_dir.resolve()),
        "train": str(train_index.resolve()),
        "val": resolved_val,
    }
    derived_data_yaml.write_text(
        "\n".join(
            [
                f"path: {derived['path']}",
                f"train: {derived['train']}",
                f"val: {derived['val']}",
                *([f"test: {resolved_test}"] if resolved_test else []),
                "names:",
                *name_lines,
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "kind": "class_targeted_copy_paste",
        "seed": config.seed,
        "source_data_yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "split": config.split,
        "target_classes": target_classes,
        "copies_per_instance": config.copies_per_instance,
        "scale_range": list(config.scale_range),
        # 写死的不变量：本实现故意不做旋转/翻转，避免与 A0 的几何增强混淆
        "rotation": "none",
        "flip": "none",
        "pastes_per_synthetic_image": 1,
        "dest_sampling": "random over split, src_image != dst_image, destination must carry >=1 annotation",
        "paste_anchor_policy": "anchor point sampled inside an existing annotated polygon of the destination image (v2)",
        "border_margin_ratio": config.border_margin_ratio,
        "mosaic_participation": "synthetic images join train index; ultralytics mosaic=1.0 applies to them equally",
        "source_instances": len(picked),
        "synthetic_images": len(records),
        "per_class_synthetic": {str(key): value for key, value in per_class.items()},
        "skipped": skipped[:50],
        "skipped_total": len(skipped),
        "derived_data_yaml": str(derived_data_yaml),
        "derived_train_index": str(train_index),
        "records": [record.__dict__ for record in records],
    }
    save_json(manifest, out_dir / "copy_paste_manifest.json")
    return manifest


__all__ = [
    "LabelInstance",
    "PasteRecord",
    "SynthesizeConfig",
    "format_label_line",
    "parse_label_line",
    "polygon_area",
    "polygon_bbox",
    "polygon_iou",
    "read_label_file",
    "synthesize",
    "write_label_file",
]
