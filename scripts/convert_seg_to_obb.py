"""将 YOLO 分割多边形标签转换为 YOLO 定向边界框标签。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import yaml
from PIL import Image

from blade_defect.data import DEFECT_CLASSES
from blade_defect.data.indexed_splits import (
    load_index_config,
    load_indexed_splits,
    membership_hash,
)


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val")
INDEX_SPLITS = ("train", "val", "test")
DEFAULT_MIN_AREA = 1e-8
CSV_FIELDS = ("split", "label", "line", "class_id", "severity", "reason", "count", "details")


IssueRow = dict[str, Any]


WarningRecord = tuple[str, str, int]


class ConditionalSourceParser(argparse.ArgumentParser):
    """Require conversion paths unless txt-index dry-run mode is selected."""

    def parse_args(self, args: Sequence[str] | None = None, namespace: argparse.Namespace | None = None) -> argparse.Namespace:
        parsed = super().parse_args(args, namespace)
        if parsed.dry_run_index_check:
            if parsed.source_data is None:
                self.error("--dry-run-index-check requires --source-data")
        elif parsed.output is None or (parsed.source is None) == (parsed.source_data is None):
            self.error("conversion requires --output and exactly one of --source/--source-data")
        return parsed


class PolygonConversionError(ValueError):
    """多边形无法生成有效定向边界框时抛出的异常。"""

    def __init__(
        self,
        reason: str,
        details: str,
        warnings: Sequence[WarningRecord] = (),
    ) -> None:
        super().__init__(details)
        self.reason = reason
        self.details = details
        self.warnings = tuple(warnings)


@dataclass(frozen=True)
class PolygonResult:
    corners: np.ndarray
    warnings: tuple[WarningRecord, ...] = ()


def _deduplicate_points(points: np.ndarray) -> tuple[np.ndarray, int]:
    unique: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for point in points:
        key = (round(float(point[0]), 12), round(float(point[1]), 12))
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return np.asarray(unique, dtype=np.float32).reshape(-1, 2), len(points) - len(unique)


def order_obb_corners(corners: np.ndarray) -> np.ndarray:
    """在图像坐标系中顺时针排序角点，并从顶部最左角点开始。"""
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    minimum_y = float(ordered[:, 1].min())
    candidates = np.flatnonzero(np.isclose(ordered[:, 1], minimum_y, atol=1e-9, rtol=0.0))
    start = int(candidates[np.argmin(ordered[candidates, 0])])
    return np.roll(ordered, -start, axis=0)


def convert_polygon_to_obb(
    polygon: Sequence[Sequence[float]] | np.ndarray,
    min_area: float = DEFAULT_MIN_AREA,
    image_size: tuple[int, int] | None = None,
) -> PolygonResult:
    """将归一化多边形点转换为顺序确定的归一化 OBB。"""
    try:
        points = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    except (TypeError, ValueError) as exc:
        raise PolygonConversionError("conversion_failed", f"invalid coordinates: {exc}") from exc

    if len(points) < 3:
        raise PolygonConversionError("insufficient_points", f"polygon has {len(points)} points; need >= 3")
    if not np.isfinite(points).all():
        raise PolygonConversionError("conversion_failed", "polygon contains NaN or infinite coordinates")
    width, height = image_size or (1, 1)
    if width <= 0 or height <= 0:
        raise PolygonConversionError("conversion_failed", f"invalid image size: {width}x{height}")

    warnings: list[WarningRecord] = []
    out_of_bounds_mask = (points < 0.0) | (points > 1.0)
    clipped_coordinates = int(np.count_nonzero(out_of_bounds_mask))
    if clipped_coordinates:
        clipped_points = int(np.count_nonzero(np.any(out_of_bounds_mask, axis=1)))
        warnings.append(
            (
                "out_of_bounds",
                f"clipped {clipped_points} points ({clipped_coordinates} coordinate values) to [0, 1]",
                clipped_points,
            )
        )
        points = np.clip(points, 0.0, 1.0)

    points, duplicate_count = _deduplicate_points(points)
    if duplicate_count:
        warnings.append(
            ("duplicate_points", f"removed {duplicate_count} repeated points", duplicate_count)
        )
    if len(points) < 3:
        raise PolygonConversionError(
            "insufficient_points",
            f"only {len(points)} unique points remain after repair",
            warnings,
        )

    polygon_area = abs(float(cv2.contourArea(points)))
    if polygon_area < min_area:
        raise PolygonConversionError(
            "area_too_small",
            f"polygon area {polygon_area:.12g} is below {min_area:.12g}",
            warnings,
        )

    try:
        scale = np.asarray([width, height], dtype=np.float32)
        rectangle = cv2.minAreaRect(points * scale)
        corners = cv2.boxPoints(rectangle).astype(np.float64) / scale
    except (cv2.error, TypeError, ValueError) as exc:
        raise PolygonConversionError("conversion_failed", str(exc), warnings) from exc

    rectangle_area = float(rectangle[1][0] * rectangle[1][1]) / (width * height)
    if not np.isfinite(corners).all() or rectangle_area < min_area:
        raise PolygonConversionError(
            "area_too_small",
            f"minimum rectangle area {rectangle_area:.12g} is below {min_area:.12g}",
            warnings,
        )
    # 选择起始角点前先裁剪坐标。两个略微越过同一图像边缘的角点在裁剪后会变为等高，
    # 此时必须选择最左侧角点，确保输出始终遵循约定的角点顺序。
    clipped_corners = np.clip(corners, 0.0, 1.0)
    # 角点顺序必须基于最终序列化值；高精度坐标可能在格式化为 8 位小数后才出现并列。
    serialized_corners = np.round(clipped_corners, 8)
    return PolygonResult(corners=order_obb_corners(serialized_corners), warnings=tuple(warnings))


def _format_number(value: float) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".") or "0"


def format_obb_label(class_id: int, corners: np.ndarray) -> str:
    values = " ".join(_format_number(value) for value in np.asarray(corners).reshape(-1))
    return f"{class_id} {values}"


def _record(
    rows: list[IssueRow],
    split: str,
    relative_label: Path,
    line_number: int,
    class_id: str,
    reason: str,
    details: str,
    *,
    severity: str = "error",
    count: int = 1,
) -> None:
    rows.append(
        {
            "split": split,
            "label": relative_label.as_posix(),
            "line": line_number,
            "class_id": class_id,
            "severity": severity,
            "reason": reason,
            "count": count,
            "details": details,
        }
    )


def convert_label_text(
    text: str,
    *,
    split: str = "",
    relative_label: Path = Path(""),
    min_area: float = DEFAULT_MIN_AREA,
    image_size: tuple[int, int] | None = None,
) -> tuple[str, list[IssueRow], int]:
    """转换单个分割标签文件中所有可恢复的实例。

    转换失败时只删除对应的单个多边形，同一图像内其他有效实例仍会保留；
    每次修复和失败都会以结构化记录返回。
    """
    output_lines: list[str] = []
    issues: list[IssueRow] = []
    converted_instances = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        tokens = raw_line.split()
        if not tokens:
            continue
        class_token = tokens[0]
        try:
            class_value = float(class_token)
            class_id = int(class_value)
            if (
                not np.isfinite(class_value)
                or class_value != class_id
                or class_id not in DEFECT_CLASSES
            ):
                raise ValueError(f"class id must be one of {sorted(DEFECT_CLASSES)}")
            coordinates = [float(token) for token in tokens[1:]]
            if len(coordinates) % 2:
                raise ValueError("polygon has an odd coordinate count")
            polygon = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
            result = convert_polygon_to_obb(
                polygon, min_area=min_area, image_size=image_size
            )
        except PolygonConversionError as exc:
            for reason, details, count in exc.warnings:
                _record(
                    issues,
                    split,
                    relative_label,
                    line_number,
                    class_token,
                    reason,
                    details,
                    severity="warning",
                    count=count,
                )
            _record(issues, split, relative_label, line_number, class_token, exc.reason, exc.details)
            continue
        except (TypeError, ValueError) as exc:
            _record(
                issues,
                split,
                relative_label,
                line_number,
                class_token,
                "conversion_failed",
                str(exc),
            )
            continue

        for reason, details, count in result.warnings:
            _record(
                issues,
                split,
                relative_label,
                line_number,
                str(class_id),
                reason,
                details,
                severity="warning",
                count=count,
            )
        output_lines.append(format_obb_label(class_id, result.corners))
        converted_instances += 1
    output = "\n".join(output_lines)
    return (output + "\n" if output else ""), issues, converted_instances


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _label_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".txt")


def _write_data_yaml(output_root: Path) -> None:
    payload = {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": DEFECT_CLASSES,
    }
    with (output_root / "data.yaml").open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)


def _split_issue_statistics(rows: list[IssueRow]) -> dict[str, int]:
    errors = [row for row in rows if row["severity"] == "error" and row["line"] > 0]
    return {
        "failed_instances": len(errors),
        "clipped_points": sum(row["count"] for row in rows if row["reason"] == "out_of_bounds"),
        "duplicate_points": sum(row["count"] for row in rows if row["reason"] == "duplicate_points"),
        "insufficient_points": sum(1 for row in rows if row["reason"] == "insufficient_points"),
        "area_too_small": sum(1 for row in rows if row["reason"] == "area_too_small"),
        "conversion_failed": sum(1 for row in rows if row["reason"] == "conversion_failed"),
    }


def dry_run_index_check(source_data: str | Path, *, min_area: float = DEFAULT_MIN_AREA) -> dict[str, object]:
    """Validate all indexed segmentation labels without writing an OBB dataset."""
    indexed = load_indexed_splits(source_data, ("train", "val", "test"))
    errors: list[str] = []
    source_instances = 0
    converted_instances = 0
    for split, samples in indexed.items():
        for sample in samples:
            if not sample.image_path.is_file():
                errors.append(f"missing image: {sample.sample_id}")
                continue
            if not sample.label_path.is_file():
                errors.append(f"missing label: {sample.sample_id}")
                continue
            label_text = sample.label_path.read_text(encoding="utf-8-sig")
            source_instances += sum(1 for line in label_text.splitlines() if line.split())
            _, issues, converted = convert_label_text(
                label_text,
                split=split,
                relative_label=Path(sample.sample_id).with_suffix(".txt"),
                min_area=min_area,
            )
            converted_instances += converted
            errors.extend(
                f"{sample.sample_id}:{row['line']}:{row['reason']}"
                for row in issues
                if row["severity"] == "error"
            )
    return {
        "tool": "convert_seg_to_obb",
        "splits": {split: len(samples) for split, samples in indexed.items()},
        "total_samples": sum(len(samples) for samples in indexed.values()),
        "source_instances": source_instances,
        "converted_instances": converted_instances,
        "membership_sha256": membership_hash(indexed),
        "errors": errors,
        "valid": not errors and converted_instances == source_instances,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sample_identity_hash(sample_ids: Sequence[str]) -> str:
    """Hash a sample identity set independently of index order and storage paths."""
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        digest.update(f"{sample_id}\n".encode("utf-8"))
    return digest.hexdigest()


def _git_commit(project_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, encoding="utf-8"
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty(project_root: Path) -> bool:
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=project_root,
                text=True,
                encoding="utf-8",
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return True


def _relative_after_images(image_path: Path) -> Path:
    lowered = [part.casefold() for part in image_path.parts]
    try:
        position = lowered.index("images")
    except ValueError as exc:
        raise ValueError(f"image path has no 'images' component: {image_path}") from exc
    relative = Path(*image_path.parts[position + 1 :])
    if not relative.parts:
        raise ValueError(f"image path has no relative name after 'images': {image_path}")
    return relative


def _materialize_image(source: Path, target: Path, mode: str) -> str:
    """Materialize without touching the parent; auto prefers a space-saving hard link."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode in {"auto", "hardlink"}:
        try:
            os.link(source, target)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise
    shutil.copy2(source, target)
    return "copy"


def _write_issue_reports(
    results_root: Path, summary: dict[str, object], issue_rows: list[IssueRow]
) -> None:
    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / "invalid_obb_labels.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(CSV_FIELDS)
        writer.writerows(
            [
                [
                    row["split"], row["label"], row["line"], row["class_id"],
                    row["severity"], row["reason"], row["count"], row["details"],
                ]
                for row in issue_rows
            ]
        )
    (results_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def convert_indexed_dataset(
    source_data: str | Path,
    output: str | Path,
    results: str | Path = "results/obb",
    *,
    parent_dataset_id: str | None = None,
    derived_dataset_id: str = "blade-v3-grouped-obb",
    min_area: float = DEFAULT_MIN_AREA,
    image_mode: str = "auto",
    overwrite: bool = False,
) -> dict[str, object]:
    """Derive train/val/test OBB data from immutable txt indexes without resplitting."""
    if min_area <= 0:
        raise ValueError("min_area must be positive")
    if image_mode not in {"auto", "copy", "hardlink"}:
        raise ValueError("image_mode must be one of: auto, copy, hardlink")

    source_config_path, source_config = load_index_config(source_data)
    parent_root = Path(source_config["path"])
    parent_manifest_path = parent_root / "dataset_manifest.json"
    if not parent_manifest_path.is_file():
        raise FileNotFoundError(f"parent dataset manifest not found: {parent_manifest_path}")
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8-sig"))
    indexed = load_indexed_splits(source_config_path, INDEX_SPLITS)

    all_samples = [sample for split in INDEX_SPLITS for sample in indexed[split]]
    all_ids = [sample.sample_id for sample in all_samples]
    duplicate_ids = sorted(sample_id for sample_id, count in Counter(all_ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(
            f"parent split indexes overlap or repeat {len(duplicate_ids)} samples; "
            f"first={duplicate_ids[0]}"
        )

    output_root = Path(output).resolve()
    results_root = Path(results).resolve()
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_root}; use --overwrite to replace it"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    split_entries: dict[str, list[str]] = {split: [] for split in INDEX_SPLITS}
    split_summaries: dict[str, dict[str, int]] = {}
    issue_rows: list[IssueRow] = []
    materialization_counts: Counter[str] = Counter()

    for split in INDEX_SPLITS:
        split_rows: list[IssueRow] = []
        source_instances = 0
        converted_instances = 0
        labels_found = 0
        negative_images = 0
        empty_labels = 0
        materialized_images = 0

        for sample in indexed[split]:
            relative_image = _relative_after_images(sample.image_path)
            target_image = output_root / "images" / relative_image
            target_label = output_root / "labels" / relative_image.with_suffix(".txt")
            index_entry = f"./{target_image.relative_to(output_root).as_posix()}"
            split_entries[split].append(index_entry)

            if not sample.image_path.is_file():
                _record(
                    split_rows, split, relative_image.with_suffix(".txt"), 0, "",
                    "missing_image", f"parent image does not exist: {sample.image_path}",
                )
                continue
            if not sample.label_path.is_file():
                _record(
                    split_rows, split, relative_image.with_suffix(".txt"), 0, "",
                    "missing_label", f"parent segmentation label does not exist: {sample.label_path}",
                )
                continue

            labels_found += 1
            label_text = sample.label_path.read_text(encoding="utf-8-sig")
            current_instances = sum(1 for line in label_text.splitlines() if line.split())
            source_instances += current_instances
            if current_instances == 0:
                negative_images += 1
            try:
                with Image.open(sample.image_path) as image:
                    image_size = image.size
                converted, current_issues, current_converted = convert_label_text(
                    label_text,
                    split=split,
                    relative_label=relative_image.with_suffix(".txt"),
                    min_area=min_area,
                    image_size=image_size,
                )
                method = _materialize_image(sample.image_path, target_image, image_mode)
                materialization_counts[method] += 1
                materialized_images += 1
            except (OSError, ValueError) as exc:
                converted = ""
                current_converted = 0
                current_issues = []
                _record(
                    split_rows, split, relative_image.with_suffix(".txt"), 0, "",
                    "image_read_failed", str(exc),
                )
            target_label.parent.mkdir(parents=True, exist_ok=True)
            target_label.write_text(converted, encoding="utf-8", newline="\n")
            converted_instances += current_converted
            split_rows.extend(current_issues)
            if not converted:
                empty_labels += 1

        issue_rows.extend(split_rows)
        issue_stats = _split_issue_statistics(split_rows)
        degenerate_boxes = issue_stats["insufficient_points"] + issue_stats["area_too_small"]
        split_summaries[split] = {
            "images": len(indexed[split]),
            "materialized_images": materialized_images,
            "labels": labels_found,
            "source_instances": source_instances,
            "converted_instances": converted_instances,
            "failed_instances": issue_stats["failed_instances"],
            "duplicate_points_removed": issue_stats["duplicate_points"],
            "clipped_points": issue_stats["clipped_points"],
            "degenerate_boxes": degenerate_boxes,
            "empty_labels": empty_labels,
            "negative_images": negative_images,
            "sample_errors": sum(row["line"] == 0 and row["severity"] == "error" for row in split_rows),
            "issue_records": len(split_rows),
        }

    for split in INDEX_SPLITS:
        (output_root / f"{split}.txt").write_text(
            "\n".join(split_entries[split]) + "\n", encoding="utf-8", newline="\n"
        )
    data_payload = {
        "path": ".",
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": source_config.get("names", DEFECT_CLASSES),
    }
    (output_root / "data.yaml").write_text(
        yaml.safe_dump(data_payload, allow_unicode=True, sort_keys=False), encoding="utf-8", newline="\n"
    )

    derived_indexed = load_indexed_splits(output_root / "data.yaml", INDEX_SPLITS)
    parent_identity_hashes = {
        split: _sample_identity_hash([sample.sample_id for sample in indexed[split]])
        for split in INDEX_SPLITS
    }
    derived_identity_hashes = {
        split: _sample_identity_hash([sample.sample_id for sample in derived_indexed[split]])
        for split in INDEX_SPLITS
    }
    exact_split_inheritance = {
        split: ({sample.sample_id for sample in indexed[split]} ==
                {sample.sample_id for sample in derived_indexed[split]})
        for split in INDEX_SPLITS
    }
    manifest_counts = parent_manifest.get("samples", {})
    count_matches_manifest = {
        split: int(manifest_counts.get(split, len(indexed[split]))) == len(indexed[split])
        for split in INDEX_SPLITS
    }
    parent_index_paths = {split: indexed[split][0].index_file for split in INDEX_SPLITS if indexed[split]}
    project_root = Path(__file__).resolve().parents[1]
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    code_commit = _git_commit(project_root)
    dataset_hash_payload = {
        "parent_manifest_sha256": _sha256_file(parent_manifest_path),
        "split_identity_sha256": derived_identity_hashes,
        "conversion_code_sha256": _sha256_file(Path(__file__)),
        "conversion_type": "seg_polygon_to_min_area_obb",
    }
    dataset_hash = hashlib.sha256(
        json.dumps(dataset_hash_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    manifest: dict[str, object] = {
        "dataset_id": derived_dataset_id,
        "derived_dataset_id": derived_dataset_id,
        "parent_dataset_id": parent_dataset_id or str(parent_manifest.get("dataset_id") or parent_root.name),
        "parent_dataset_manifest_id": str(parent_manifest.get("dataset_id") or parent_root.name),
        "conversion_type": "seg_polygon_to_min_area_obb",
        "generated_at": generated_at,
        "conversion_time": generated_at,
        "conversion_commit": code_commit,
        "conversion_worktree_dirty": _git_dirty(project_root),
        "conversion_code": str(Path(__file__).relative_to(project_root).as_posix()),
        "conversion_code_sha256": _sha256_file(Path(__file__)),
        "dataset_hash": dataset_hash,
        "dataset_hash_components": dataset_hash_payload,
        "parent_dataset_manifest": os.path.relpath(parent_manifest_path, output_root).replace("\\", "/"),
        "parent_manifest_sha256": _sha256_file(parent_manifest_path),
        "samples": {split: len(indexed[split]) for split in INDEX_SPLITS},
        "labels": {split: split_summaries[split]["labels"] for split in INDEX_SPLITS},
        "instances": sum(split_summaries[split]["source_instances"] for split in INDEX_SPLITS),
        "converted_instances": sum(split_summaries[split]["converted_instances"] for split in INDEX_SPLITS),
        "parent_split_sha256": {
            split: _sha256_file(parent_index_paths[split]) for split in INDEX_SPLITS
        },
        "derived_split_sha256": {
            split: _sha256_file(output_root / f"{split}.txt") for split in INDEX_SPLITS
        },
        "parent_split_identity_sha256": parent_identity_hashes,
        "derived_split_identity_sha256": derived_identity_hashes,
        "parent_membership_sha256": membership_hash(indexed),
        "derived_membership_sha256": membership_hash(derived_indexed),
        "split_inheritance": {
            "exact_by_sample_identity": exact_split_inheritance,
            "no_added_or_missing_samples": all(exact_split_inheritance.values()),
            "no_cross_split_movement": all(exact_split_inheritance.values()),
            "group_identity_preserved": all(exact_split_inheritance.values()),
            "method": "one-to-one sample derivation; no split or group assignment operation",
        },
        "test_policy": {
            "converted": True,
            "validated_for_data_integrity": True,
            "allowed_for_training": False,
            "allowed_for_model_selection": False,
            "allowed_for_threshold_selection": False,
        },
        "storage": {
            "type": "txt_indexes_with_derived_obb_labels_and_materialized_parent_images",
            "image_materialization": dict(sorted(materialization_counts.items())),
            "modifies_parent": False,
        },
    }
    (output_root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    totals = {
        key: sum(split_summaries[split][key] for split in INDEX_SPLITS)
        for key in (
            "images", "materialized_images", "labels", "source_instances",
            "converted_instances", "failed_instances", "duplicate_points_removed",
            "clipped_points", "degenerate_boxes", "empty_labels", "negative_images",
            "sample_errors", "issue_records",
        )
    }
    valid = (
        all(exact_split_inheritance.values())
        and all(count_matches_manifest.values())
        and totals["materialized_images"] == totals["images"]
        and totals["labels"] == totals["images"]
        and totals["failed_instances"] == 0
        and totals["sample_errors"] == 0
        and totals["converted_instances"] == totals["source_instances"]
    )
    summary: dict[str, object] = {
        "valid": valid,
        "source_data": str(source_config_path),
        "source_dataset": str(parent_manifest.get("dataset_id") or parent_root.name),
        "output_dataset": derived_dataset_id,
        "output_path": str(output_root),
        "failure_policy": "drop_failed_instance_keep_image_and_other_valid_instances",
        "corner_order": "clockwise_from_topmost_then_leftmost",
        "min_area": min_area,
        "splits": split_summaries,
        "totals": totals,
        "parent_manifest_counts_match": count_matches_manifest,
        "split_identity_match": exact_split_inheritance,
        "parent_membership_sha256": membership_hash(indexed),
        "derived_membership_sha256": membership_hash(derived_indexed),
        "issue_counts": dict(sorted(Counter(row["reason"] for row in issue_rows).items())),
    }
    _write_issue_reports(results_root, summary, issue_rows)
    return summary


def convert_dataset(
    source: str | Path,
    output: str | Path,
    results: str | Path = "results/obb",
    *,
    min_area: float = DEFAULT_MIN_AREA,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """转换训练集和验证集、复制图像，并生成完整审计报告。"""
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    results_root = Path(results).resolve()
    if min_area <= 0:
        raise ValueError("min_area must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if source_root == output_root:
        raise ValueError("source and output directories must be different")
    for split in SPLITS:
        if not (source_root / "images" / split).is_dir():
            raise FileNotFoundError(f"missing image directory: {source_root / 'images' / split}")
        if not (source_root / "labels" / split).is_dir():
            raise FileNotFoundError(f"missing label directory: {source_root / 'labels' / split}")

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_root}; use --overwrite to replace it"
            )
        shutil.rmtree(output_root)
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    issue_rows: list[IssueRow] = []
    split_summaries: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        images_root = source_root / "images" / split
        labels_root = source_root / "labels" / split
        all_images = _image_files(images_root)
        all_labels = _label_files(labels_root)
        images = all_images[:limit] if limit is not None else all_images
        selected_keys = {
            image.relative_to(images_root).with_suffix("").as_posix().casefold()
            for image in images
        }
        labels_by_key = {
            label.relative_to(labels_root).with_suffix("").as_posix().casefold(): label
            for label in all_labels
        }
        selected_labels = {
            key: label for key, label in labels_by_key.items() if key in selected_keys
        }
        split_rows: list[IssueRow] = []
        source_instances = 0
        converted_instances = 0
        empty_after_conversion = 0
        negative_images = 0

        for image_path in images:
            relative_image = image_path.relative_to(images_root)
            relative_label = relative_image.with_suffix(".txt")
            key = relative_image.with_suffix("").as_posix().casefold()
            label_path = selected_labels.get(key)
            target_image = output_root / "images" / split / relative_image
            target_label = output_root / "labels" / split / relative_label
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)

            if label_path is None:
                _record(
                    split_rows, split, relative_label, 0, "", "missing_label",
                    "image has no matching source label",
                )
                target_label.write_text("", encoding="utf-8")
                continue
            label_text = label_path.read_text(encoding="utf-8-sig")
            current_instances = sum(1 for line in label_text.splitlines() if line.split())
            source_instances += current_instances
            if current_instances == 0:
                negative_images += 1
            try:
                with Image.open(image_path) as image:
                    image_size = image.size
                converted, current_issues, current_converted = convert_label_text(
                    label_text,
                    split=split,
                    relative_label=relative_label,
                    min_area=min_area,
                    image_size=image_size,
                )
            except (OSError, ValueError) as exc:
                converted = ""
                current_converted = 0
                current_issues = []
                _record(
                    split_rows, split, relative_label, 0, "", "image_read_failed", str(exc)
                )
            target_label.write_text(converted, encoding="utf-8", newline="\n")
            converted_instances += current_converted
            split_rows.extend(current_issues)
            if current_instances > 0 and current_converted == 0:
                empty_after_conversion += 1

        if limit is None:
            for key, label_path in labels_by_key.items():
                if key not in selected_keys:
                    _record(
                        split_rows,
                        split,
                        label_path.relative_to(labels_root),
                        0,
                        "",
                        "orphan_label",
                        "label has no matching source image",
                    )

        issue_rows.extend(split_rows)
        issue_stats = _split_issue_statistics(split_rows)
        split_summaries[split] = {
            "images": len(images),
            "labels": len(selected_labels),
            "instances": source_instances,
            "converted_instances": converted_instances,
            **issue_stats,
            "empty_after_conversion": empty_after_conversion,
            "negative_images": negative_images,
            "output_images": len(images),
            "output_labels": len(images),
            # 保留初版转换测试使用的字段别名，维持向后兼容。
            "label_files": len(selected_labels),
            "converted_objects": converted_instances,
            "issue_records": len(split_rows),
            "files_with_issues": len({row["label"] for row in split_rows}),
        }

    _write_data_yaml(output_root)
    reason_counts = Counter(row["reason"] for row in issue_rows)
    summary: dict[str, object] = {
        "source_dataset": source_root.name,
        "output_dataset": output_root.name,
        "source_path": str(source_root),
        "output_path": str(output_root),
        "failure_policy": "drop_failed_instance_keep_image_and_other_valid_instances",
        "corner_order": "clockwise_from_topmost_then_leftmost",
        "min_area": min_area,
        "splits": split_summaries,
        "totals": {
            "images": sum(item["images"] for item in split_summaries.values()),
            "labels": sum(item["labels"] for item in split_summaries.values()),
            "instances": sum(item["instances"] for item in split_summaries.values()),
            "converted_instances": sum(
                item["converted_instances"] for item in split_summaries.values()
            ),
            "failed_instances": sum(item["failed_instances"] for item in split_summaries.values()),
            "clipped_points": sum(item["clipped_points"] for item in split_summaries.values()),
            "empty_after_conversion": sum(
                item["empty_after_conversion"] for item in split_summaries.values()
            ),
            "converted_objects": sum(
                item["converted_instances"] for item in split_summaries.values()
            ),
            "issue_records": len(issue_rows),
        },
        "issue_counts": dict(sorted(reason_counts.items())),
    }

    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / "invalid_obb_labels.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(CSV_FIELDS)
        writer.writerows(
            [
                [
                    row["split"],
                    row["label"],
                    row["line"],
                    row["class_id"],
                    row["severity"],
                    row["reason"],
                    row["count"],
                    row["details"],
                ]
                for row in issue_rows
            ]
        )
    (results_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = ConditionalSourceParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="YOLO-seg dataset root")
    parser.add_argument("--output", type=Path, help="new YOLO-OBB dataset root")
    parser.add_argument("--source-data", type=Path, help="data.yaml backed by train/val/test txt indexes")
    parser.add_argument("--dry-run-index-check", action="store_true", help="check all indexed labels without writing files")
    parser.add_argument("--parent-dataset-id", help="explicit parent identity stored in derived manifest")
    parser.add_argument("--derived-dataset-id", default="blade-v3-grouped-obb")
    parser.add_argument(
        "--image-mode", choices=("auto", "copy", "hardlink"), default="auto",
        help="materialize indexed parent images; auto prefers hard links and falls back to copies",
    )
    parser.add_argument("--results", type=Path, default=Path("results/obb"))
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument("--limit", type=int, help="maximum images per split (tests only)")
    parser.add_argument(
        "--overwrite", action="store_true", help="delete and replace an existing output directory"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.dry_run_index_check:
        if args.source_data is None:
            raise SystemExit("--dry-run-index-check requires --source-data")
        report = dry_run_index_check(args.source_data, min_area=args.min_area)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["valid"]:
            raise SystemExit(1)
        return
    output = args.output.resolve()
    results = args.results.resolve()
    print(f"Output dataset: {output}")
    print(f"Audit results:  {results}")
    if args.source_data is not None:
        print(f"Source index:   {args.source_data.resolve()}")
        summary = convert_indexed_dataset(
            args.source_data,
            output,
            results,
            parent_dataset_id=args.parent_dataset_id,
            derived_dataset_id=args.derived_dataset_id,
            min_area=args.min_area,
            image_mode=args.image_mode,
            overwrite=args.overwrite,
        )
    else:
        if args.source is None:
            raise SystemExit("directory conversion requires --source")
        source = args.source.resolve()
        print(f"Source dataset: {source}")
        summary = convert_dataset(
            source,
            output,
            results,
            min_area=args.min_area,
            limit=args.limit,
            overwrite=args.overwrite,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary.get("valid", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
