"""Reproducible YOLO dataset splitting."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from blade_defect.utils.files import find_images, find_labels
from blade_defect.utils.paths import resolve_path


def split_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
    copy: bool = True,
) -> dict[str, int]:
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios 必须是和为 1 的 train/val/test 三元组")

    images_root = resolve_path(images_dir)
    labels_root = resolve_path(labels_dir)
    output_root = resolve_path(output_dir)
    labels_by_key = {
        path.relative_to(labels_root).with_suffix("").as_posix().casefold(): path
        for path in find_labels(labels_root)
    }
    pairs: list[tuple[Path, Path]] = []
    images = find_images(images_root)
    for image_path in images:
        relative = image_path.relative_to(images_root)
        key = relative.with_suffix("").as_posix().casefold()
        label_path = labels_by_key.get(key)
        if label_path is not None:
            pairs.append((image_path, label_path))

    random.Random(seed).shuffle(pairs)
    train_end = int(len(pairs) * ratios[0])
    val_end = train_end + int(len(pairs) * ratios[1])
    groups = {"train": pairs[:train_end], "val": pairs[train_end:val_end], "test": pairs[val_end:]}
    operation = shutil.copy2 if copy else shutil.move

    for split_name, split_pairs in groups.items():
        for image_path, label_path in split_pairs:
            relative = image_path.relative_to(images_root)
            image_target = output_root / "images" / split_name / relative
            label_target = output_root / "labels" / split_name / relative.with_suffix(".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            operation(image_path, image_target)
            operation(label_path, label_target)
    counts = {name: len(items) for name, items in groups.items()}
    counts["unmatched_images"] = len(images) - len(pairs)
    return counts
