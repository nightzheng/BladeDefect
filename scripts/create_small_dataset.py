"""从 YOLO-seg 数据集创建可复现、可直接训练的小样本数据集。"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from blade_defect.data import DEFECT_CLASSES, check_dataset
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


def _is_decodable_image(path: Path) -> bool:
    """使用兼容 Unicode 路径的方式检查 OpenCV 能否解码图片。"""
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        return encoded.size > 0 and cv2.imdecode(encoded, cv2.IMREAD_COLOR) is not None
    except (OSError, ValueError, cv2.error):
        return False


def _paired_samples(
    dataset_root: Path, split: str,
) -> tuple[list[tuple[Path, Path, Path]], int, int, int]:
    """返回有效的图片/标注配对，以及两个方向的缺失统计。"""
    images_root = dataset_root / "images" / split
    labels_root = dataset_root / "labels" / split
    if not images_root.is_dir() or not labels_root.is_dir():
        raise FileNotFoundError(
            f"数据集缺少 {split} 目录；需要 {images_root} 和 {labels_root}"
        )

    labels_by_key = {
        path.relative_to(labels_root).with_suffix("").as_posix().casefold(): path
        for path in _label_files(labels_root)
        if path.exists()
    }
    images = _image_files(images_root)
    image_keys = {
        path.relative_to(images_root).with_suffix("").as_posix().casefold()
        for path in images
        if path.exists()
    }
    pairs: list[tuple[Path, Path, Path]] = []
    missing_labels = 0
    corrupt_images = 0
    for image_path in images:
        relative = image_path.relative_to(images_root)
        key = relative.with_suffix("").as_posix().casefold()
        label_path = labels_by_key.get(key)
        # 显式复查可防止文件在目录扫描后、配对建立前被移动或删除。
        if not image_path.exists():
            continue
        if label_path is None or not label_path.exists():
            missing_labels += 1
            continue
        # 零字节文件可通过元数据快速拒绝，不进入后续解码候选池。
        if image_path.stat().st_size == 0:
            corrupt_images += 1
            continue
        pairs.append((image_path, label_path, relative))
    missing_images = sum(key not in image_keys for key in labels_by_key)
    return pairs, missing_images, missing_labels, corrupt_images


def _select_decodable_samples(
    pairs: list[tuple[Path, Path, Path]],
    requested: int,
    rng: random.Random,
    split: str,
) -> tuple[list[tuple[Path, Path, Path]], int, int]:
    """随机惰性解码候选，失败时自动补位，直到满足目标数量。"""
    candidates = list(pairs)
    rng.shuffle(candidates)
    selected: list[tuple[Path, Path, Path]] = []
    rejected = 0
    checked = 0
    # 进度按成功选中的样本推进；坏图只增加 checked/rejected，随后自动补位。
    with tqdm(total=requested, desc=f"{split} 解码筛选", unit="张", disable=None) as progress:
        for sample in candidates:
            if len(selected) >= requested:
                break
            checked += 1
            if _is_decodable_image(sample[0]):
                selected.append(sample)
                progress.update(1)
            else:
                rejected += 1
            progress.set_postfix(checked=checked, rejected=rejected, refresh=False)
    return selected, rejected, checked


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


def _clean_generated_split(output_root: Path, split: str) -> dict[str, int]:
    """修复轻微越界 polygon、删除无效样本，并执行严格验收。"""
    images_root = output_root / "images" / split
    labels_root = output_root / "labels" / split
    repair_report = check_dataset(
        images_root,
        labels_root,
        num_classes=len(DEFECT_CLASSES),
        polygon_mode="auto-fix",
    )

    # auto-fix 会删除严重坐标越界样本；其余格式错误同样无法用于分割训练，
    # 因此在生成阶段一并删除完整的 image/label 样本对。
    manually_removed = 0
    invalid_labels = {issue.file for issue in repair_report.issues}
    images_by_key = {
        path.relative_to(images_root).with_suffix("").as_posix().casefold(): path
        for path in _image_files(images_root)
    }
    for relative_label in invalid_labels:
        label_path = labels_root / relative_label
        if not label_path.exists():
            continue
        key = Path(relative_label).with_suffix("").as_posix().casefold()
        label_path.unlink()
        image_path = images_by_key.get(key)
        if image_path is not None:
            image_path.unlink(missing_ok=True)
        manually_removed += 1

    # 只有最终输出满足与 experiment run-all 相同的 strict 约束，生成才算成功。
    final_report = check_dataset(
        images_root,
        labels_root,
        num_classes=len(DEFECT_CLASSES),
        polygon_mode="strict",
        dry_run=True,
    )
    if final_report.missing_labels or final_report.orphan_labels or final_report.issues:
        raise RuntimeError(f"Generated {split} split failed strict dataset validation")
    return {
        "fixed_points": repair_report.fixed_points,
        "fixed_files": repair_report.fixed_files,
        "removed_files": repair_report.removed_files + manually_removed,
        "decode_removed": 0,
    }


def create_small_dataset(
    source: str | Path,
    output: str | Path,
    train_count: int = 200,
    val_count: int = 50,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """将经过配对、解码和 polygon 清洗的 train/val 样本复制到新数据集。"""
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
        pairs, missing_images, missing_labels, corrupt_images = _paired_samples(source_root, split)
        requested = requested_counts[split]
        selected, decode_rejected, decode_checked = _select_decodable_samples(
            pairs, requested, rng, split
        )
        corrupt_images += decode_rejected
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        for image_path, label_path, relative in selected:
            # 复制前再次检查，避免并发移动/删除产生只有单边文件的样本。
            if not image_path.exists():
                missing_images += 1
                continue
            if not label_path.exists():
                missing_labels += 1
                continue
            image_target = output_root / "images" / split / relative
            label_target = output_root / "labels" / split / relative.with_suffix(".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(image_path, image_target)
                shutil.copy2(label_path, label_target)
            except FileNotFoundError:
                # 源文件可能在检查后消失；清理半复制目标，确保输出始终成对。
                image_target.unlink(missing_ok=True)
                label_target.unlink(missing_ok=True)
                if not image_path.exists():
                    missing_images += 1
                if not label_path.exists():
                    missing_labels += 1
                continue

        cleaning = _clean_generated_split(output_root, split)
        copied = sum(
            1
            for image_path in (output_root / "images" / split).rglob("*")
            if image_path.is_file()
            and (output_root / "labels" / split
                 / image_path.relative_to(output_root / "images" / split).with_suffix(".txt")).is_file()
        )

        statistics[split] = {
            "requested": requested,
            "available_pairs": len(pairs),
            "copied": copied,
            # 保留旧字段别名，避免影响已有调用方。
            "unmatched_images": missing_labels,
            "missing_images": missing_images,
            "missing_labels": missing_labels,
            "decode_checked": decode_checked,
            "corrupt_images": corrupt_images + cleaning["decode_removed"],
            **cleaning,
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
