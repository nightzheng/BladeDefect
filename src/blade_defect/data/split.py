"""Reproducible YOLO dataset splitting."""

from __future__ import annotations

import random
import shutil
from pathlib import Path

from blade_defect.data.dataset_filter import load_dataset_filter
from blade_defect.data.label_check import check_dataset
from blade_defect.utils.files import find_images, find_labels
from blade_defect.utils.paths import resolve_path


def _clean_output_split(output_root: Path, split_name: str) -> dict[str, int]:
    """只修正新数据集中的 soft error；hard error 必须人工确认。"""

    images_root = output_root / "images" / split_name
    labels_root = output_root / "labels" / split_name
    repair_report = check_dataset(
        images_root,
        labels_root,
        polygon_mode="soft",
    )

    if repair_report.issues:
        affected_files = len({issue.file for issue in repair_report.issues})
        raise RuntimeError(
            f"Generated {split_name} split contains {affected_files} hard/invalid label files; "
            "人工确认后修复，或通过 dataset_filter 明确排除，程序不会自动删除"
        )

    final_report = check_dataset(
        images_root,
        labels_root,
        polygon_mode="strict",
        dry_run=True,
    )
    if final_report.missing_labels or final_report.orphan_labels or final_report.issues:
        raise RuntimeError(f"Generated {split_name} split failed strict dataset validation")
    return {
        "samples": final_report.images,
        "fixed_points": repair_report.fixed_points,
        "fixed_label_files": repair_report.fixed_files,
        "removed_invalid_samples": 0,
    }


def split_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    output_dir: str | Path,
    ratios: tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42,
    copy: bool = True,
    filter_config: str | Path | None = None,
) -> dict[str, int]:
    if len(ratios) != 3 or any(value < 0 for value in ratios) or abs(sum(ratios) - 1.0) > 1e-8:
        raise ValueError("ratios 必须是和为 1 的 train/val/test 三元组")

    images_root = resolve_path(images_dir)
    labels_root = resolve_path(labels_dir)
    output_root = resolve_path(output_dir)
    dataset_filter = load_dataset_filter(filter_config) if filter_config is not None else None
    labels_by_key = {
        path.relative_to(labels_root).with_suffix("").as_posix().casefold(): path
        for path in find_labels(labels_root)
    }
    pairs: list[tuple[Path, Path | None]] = []
    images = find_images(images_root)
    unmatched_images = 0
    action_counts = {action: 0 for action in ("exclude", "review", "keep_negative")}
    for image_path in images:
        relative = image_path.relative_to(images_root)
        key = relative.with_suffix("").as_posix().casefold()
        label_path = labels_by_key.get(key)
        decision = dataset_filter.decision_for(image_path) if dataset_filter is not None else None
        if decision is not None and decision.action in {"exclude", "review"}:
            action_counts[decision.action] += 1
            continue
        if decision is not None and decision.action == "keep_negative":
            action_counts[decision.action] += 1
            pairs.append((image_path, None))
        elif label_path is not None:
            pairs.append((image_path, label_path))
        else:
            unmatched_images += 1

    random.Random(seed).shuffle(pairs)
    train_end = int(len(pairs) * ratios[0])
    val_end = train_end + int(len(pairs) * ratios[1])
    groups = {"train": pairs[:train_end], "val": pairs[train_end:val_end], "test": pairs[val_end:]}
    operation = shutil.copy2 if copy else shutil.move

    for split_name, split_pairs in groups.items():
        (output_root / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split_name).mkdir(parents=True, exist_ok=True)
        for image_path, label_path in split_pairs:
            relative = image_path.relative_to(images_root)
            image_target = output_root / "images" / split_name / relative
            label_target = output_root / "labels" / split_name / relative.with_suffix(".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            operation(image_path, image_target)
            if label_path is None:
                label_target.write_text("", encoding="utf-8")
            else:
                operation(label_path, label_target)
    cleaning = {
        split_name: _clean_output_split(output_root, split_name)
        for split_name in groups
    }
    counts = {name: cleaning[name]["samples"] for name in groups}
    counts["unmatched_images"] = unmatched_images
    counts["fixed_points"] = sum(item["fixed_points"] for item in cleaning.values())
    counts["fixed_label_files"] = sum(item["fixed_label_files"] for item in cleaning.values())
    counts["removed_invalid_samples"] = sum(
        item["removed_invalid_samples"] for item in cleaning.values()
    )
    if dataset_filter is not None:
        counts.update(
            {
                "excluded_images": action_counts["exclude"],
                "review_images": action_counts["review"],
                "negative_images": action_counts["keep_negative"],
            }
        )
    return counts
