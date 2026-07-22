"""Convert YOLO segmentation polygons into YOLO oriented bounding boxes."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict

import cv2
import numpy as np
import yaml
from PIL import Image

from blade_defect.data import DEFECT_CLASSES


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val")
DEFAULT_MIN_AREA = 1e-8
CSV_FIELDS = ("split", "label", "line", "class_id", "severity", "reason", "count", "details")


class IssueRow(TypedDict):
    split: str
    label: str
    line: int
    class_id: str
    severity: str
    reason: str
    count: int
    details: str


WarningRecord = tuple[str, str, int]


class PolygonConversionError(ValueError):
    """A polygon cannot produce a valid oriented bounding box."""

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
    """Order corners clockwise in image coordinates, starting at top/left."""
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
    """Convert normalized polygon points into a deterministic normalized OBB."""
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
    # Clip before choosing the start corner. Two corners that were microscopically
    # outside the same image edge become tied after clipping; the left-most one
    # must then be selected to preserve the published ordering convention.
    clipped_corners = np.clip(corners, 0.0, 1.0)
    # Ordering must describe the serialized values, not higher-precision values
    # that can become tied only after the 8-decimal label formatting step.
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
    """Convert all recoverable instances in one segmentation label file.

    A failed polygon is removed as a single instance. Other valid instances in
    the same image are retained, and every repair/failure is returned in rows.
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


def convert_dataset(
    source: str | Path,
    output: str | Path,
    results: str | Path = "results/obb",
    *,
    min_area: float = DEFAULT_MIN_AREA,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict[str, object]:
    """Convert train/val splits, copy images and emit complete audit reports."""
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
            # Backward-compatible aliases used by the first conversion tests.
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
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(issue_rows)
    (results_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="YOLO-seg dataset root")
    parser.add_argument("--output", required=True, type=Path, help="new YOLO-OBB dataset root")
    parser.add_argument("--results", type=Path, default=Path("results/obb"))
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument("--limit", type=int, help="maximum images per split (tests only)")
    parser.add_argument(
        "--overwrite", action="store_true", help="delete and replace an existing output directory"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    results = args.results.resolve()
    print(f"Source dataset: {source}")
    print(f"Output dataset: {output}")
    print(f"Audit results:  {results}")
    summary = convert_dataset(
        source,
        output,
        results,
        min_area=args.min_area,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
