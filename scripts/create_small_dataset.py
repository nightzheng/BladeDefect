"""Create a reproducible small sample from a YOLO segmentation dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import yaml

from blade_defect.data import DEFECT_CLASSES
from blade_defect.utils import resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
SPLITS = ("train", "val")


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _label_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() == ".txt")


def _paired_samples(dataset_root: Path, split: str) -> tuple[list[tuple[Path, Path, Path]], int]:
    images_root = dataset_root / "images" / split
    labels_root = dataset_root / "labels" / split
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(
            f"数据集缺少 {split} 目录；需要 {images_root} 和 {labels_root}"
        )

    labels_by_key = {
        path.relative_to(labels_root).with_suffix("").as_posix().casefold(): path
        for path in _label_files(labels_root)
    }
    images = _image_files(images_root)
    pairs: list[tuple[Path, Path, Path]] = []
    for image_path in images:
        relative = image_path.relative_to(images_root)
        key = relative.with_suffix("").as_posix().casefold()
        label_path = labels_by_key.get(key)
        if label_path is not None:
            pairs.append((image_path, label_path, relative))
    return pairs, len(images) - len(pairs)


def _write_data_yaml(output_root: Path) -> Path:
    data_yaml = output_root / "data.yaml"
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": DEFECT_CLASSES,
    }
    with data_yaml.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)
    return data_yaml


def create_small_dataset(
    source: str | Path,
    output: str | Path,
    train_count: int = 200,
    val_count: int = 50,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Copy paired train/val samples into a compact YOLO dataset."""
    if train_count < 0 or val_count < 0:
        raise ValueError("train_count 和 val_count 不能为负数")

    source_root = resolve_path(source)
    output_root = resolve_path(output)
    if not source_root.is_dir():
        raise FileNotFoundError(f"原始数据集目录不存在：{source_root}")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"输出目录必须为空：{output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    requested_counts = {"train": train_count, "val": val_count}
    statistics: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        pairs, unmatched_images = _paired_samples(source_root, split)
        requested = requested_counts[split]
        selected = rng.sample(pairs, min(requested, len(pairs)))
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for image_path, label_path, relative in selected:
            image_target = output_root / "images" / split / relative
            label_target = output_root / "labels" / split / relative.with_suffix(".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, image_target)
            shutil.copy2(label_path, label_target)

        statistics[split] = {
            "requested": requested,
            "available_pairs": len(pairs),
            "copied": len(selected),
            "unmatched_images": unmatched_images,
        }

    _write_data_yaml(output_root)
    return statistics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=resolve_path, help="原始 YOLO-seg 数据集根目录")
    parser.add_argument("--output", required=True, type=resolve_path, help="小样本数据集输出目录")
    parser.add_argument("--train-count", type=int, default=200, help="训练集抽样数量")
    parser.add_argument("--val-count", type=int, default=50, help="验证集抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    statistics = create_small_dataset(
        source=args.source,
        output=args.output,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
    )
    print(json.dumps(statistics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
