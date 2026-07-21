"""将 YOLO 分割数据集转换为 YOLO 定向边界框标签。

输出标签格式为 ``class x1 y1 x2 y2 x3 y3 x4 y4``。角点按照图像坐标顺时针排列，
从最顶部（如果齐平则从最左侧）角点开始。输入坐标超出 [0, 1] 范围时会被裁剪并记录。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence, TypedDict, cast

import cv2
import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "val")
DEFAULT_MIN_AREA = 1e-8
CSV_FIELDS = ("split", "label", "line", "class_id", "reason", "details")


class IssueRow(TypedDict):
    split: str
    label: str
    line: int
    class_id: str
    reason: str
    details: str


class PolygonConversionError(ValueError):
    """多边形无法生成有效的定向边界框。"""

    def __init__(
        self,
        reason: str,
        details: str,
        warnings: Sequence[tuple[str, str]] = (),
    ) -> None:
        super().__init__(details)
        self.reason = reason
        self.details = details
        self.warnings = tuple(warnings)


@dataclass(frozen=True)
class PolygonResult:
    corners: np.ndarray
    warnings: tuple[tuple[str, str], ...] = ()


def _deduplicate_points(points: np.ndarray) -> tuple[np.ndarray, int]:
    """删除重复点，并保留首次出现的坐标。"""
    unique: list[np.ndarray] = []
    seen: set[tuple[float, float]] = set()
    for point in points:
        # 通过四舍五入使重复点检测在文本 -> 浮点解析后保持稳定。
        key = (round(float(point[0]), 12), round(float(point[1]), 12))
        if key not in seen:
            seen.add(key)
            unique.append(point)
    return np.asarray(unique, dtype=np.float32).reshape(-1, 2), len(points) - len(unique)


def order_obb_corners(corners: np.ndarray) -> np.ndarray:
    """返回按顺时针顺序排列的四个角点，从最顶部/左侧的角点开始。"""
    points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    ordered = points[np.argsort(angles)]
    start = int(np.lexsort((ordered[:, 0], ordered[:, 1]))[0])
    return np.roll(ordered, -start, axis=0)


def convert_polygon_to_obb(
    polygon: Sequence[Sequence[float]] | np.ndarray,
    min_area: float = DEFAULT_MIN_AREA,
    image_size: tuple[int, int] | None = None,
) -> PolygonResult:
    """将归一化的多边形点转换为确定性、归一化的 OBB。

    超出范围的坐标和重复点会修复并作为警告返回。无法修复的多边形会抛出
    :class:`PolygonConversionError`。
    """
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

    warnings: list[tuple[str, str]] = []
    out_of_bounds = int(np.count_nonzero((points < 0.0) | (points > 1.0)))
    if out_of_bounds:
        warnings.append(("out_of_bounds", f"clipped {out_of_bounds} coordinate values to [0, 1]"))
        points = np.clip(points, 0.0, 1.0)

    points, duplicate_count = _deduplicate_points(points)
    if duplicate_count:
        warnings.append(("duplicate_points", f"removed {duplicate_count} repeated points"))
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
    corners = np.clip(order_obb_corners(corners), 0.0, 1.0)
    return PolygonResult(corners=corners, warnings=tuple(warnings))


def _format_number(value: float) -> str:
    return f"{float(value):.8f}".rstrip("0").rstrip(".") or "0"


def format_obb_label(class_id: int, corners: np.ndarray) -> str:
    values = " ".join(_format_number(value) for value in np.asarray(corners).reshape(-1))
    return f"{class_id} {values}"


def _record(
    rows: list[IssueRow], split: str, relative_label: Path, line_number: int,
    class_id: str, reason: str, details: str,
) -> None:
    rows.append(
        {
            "split": split,
            "label": relative_label.as_posix(),
            "line": line_number,
            "class_id": class_id,
            "reason": reason,
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
    """将一个 YOLO 分割标签文件中的所有有效多边形转换为定向边界框。"""
    output_lines: list[str] = []
    issues: list[IssueRow] = []
    object_count = 0
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        tokens = raw_line.split()
        if not tokens:
            continue
        class_token = tokens[0]
        try:
            class_value = float(class_token)
            class_id = int(class_value)
            if not np.isfinite(class_value) or class_value != class_id or class_id < 0:
                raise ValueError("class id must be a non-negative integer")
            coordinates = [float(token) for token in tokens[1:]]
            if len(coordinates) % 2:
                raise ValueError("polygon has an odd coordinate count")
            polygon = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
            result = convert_polygon_to_obb(
                polygon, min_area=min_area, image_size=image_size
            )
        except PolygonConversionError as exc:
            for reason, details in exc.warnings:
                _record(
                    issues, split, relative_label, line_number, class_token, reason, details
                )
            _record(issues, split, relative_label, line_number, class_token, exc.reason, exc.details)
            continue
        except (TypeError, ValueError) as exc:
            _record(
                issues, split, relative_label, line_number, class_token,
                "conversion_failed", str(exc),
            )
            continue

        for reason, details in result.warnings:
            _record(issues, split, relative_label, line_number, str(class_id), reason, details)
        output_lines.append(format_obb_label(class_id, result.corners))
        object_count += 1
    output = "\n".join(output_lines)
    return (output + "\n" if output else ""), issues, object_count


def _image_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def convert_dataset(
    source: str | Path,
    output: str | Path,
    results: str | Path,
    *,
    min_area: float = DEFAULT_MIN_AREA,
    limit: int | None = None,
) -> dict[str, object]:
    """转换 train/val 划分，复制配对图像，并写入审计报告。"""
    source_root = Path(source).resolve()
    output_root = Path(output).resolve()
    results_root = Path(results).resolve()
    if min_area <= 0:
        raise ValueError("min_area must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    for split in SPLITS:
        if not (source_root / "images" / split).is_dir():
            raise FileNotFoundError(f"missing image directory: {source_root / 'images' / split}")
        if not (source_root / "labels" / split).is_dir():
            raise FileNotFoundError(f"missing label directory: {source_root / 'labels' / split}")
    if source_root == output_root:
        raise ValueError("source and output directories must be different")

    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    issue_rows: list[IssueRow] = []
    split_summaries: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        images_root = source_root / "images" / split
        labels_root = source_root / "labels" / split
        images = _image_files(images_root)
        if limit is not None:
            images = images[:limit]
        converted_objects = 0
        files_with_issues: set[str] = set()

        for image_path in images:
            relative_image = image_path.relative_to(images_root)
            relative_label = relative_image.with_suffix(".txt")
            label_path = labels_root / relative_label
            target_image = output_root / "images" / split / relative_image
            target_label = output_root / "labels" / split / relative_label
            target_image.parent.mkdir(parents=True, exist_ok=True)
            target_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target_image)
            with Image.open(image_path) as image:
                image_size = image.size

            # 缺失的 YOLO 标签表示未标注/负样本图像。
            label_text = label_path.read_text(encoding="utf-8-sig") if label_path.is_file() else ""
            converted, current_issues, current_objects = convert_label_text(
                label_text,
                split=split,
                relative_label=relative_label,
                min_area=min_area,
                image_size=image_size,
            )
            target_label.write_text(converted, encoding="utf-8", newline="\n")
            converted_objects += current_objects
            issue_rows.extend(current_issues)
            if current_issues:
                files_with_issues.add(relative_label.as_posix())

        split_summaries[split] = {
            "images": len(images),
            "label_files": len(images),
            "converted_objects": converted_objects,
            "issue_records": sum(1 for row in issue_rows if row["split"] == split),
            "files_with_issues": len(files_with_issues),
        }

    source_yaml = source_root / "data.yaml"
    if source_yaml.is_file():
        shutil.copy2(source_yaml, output_root / "data.yaml")

    reason_counts = Counter(str(row["reason"]) for row in issue_rows)
    summary: dict[str, object] = {
        "source": str(source_root),
        "output": str(output_root),
        "corner_order": "clockwise_from_topmost_then_leftmost",
        "min_area": min_area,
        "splits": split_summaries,
        "totals": {
            "images": sum(item["images"] for item in split_summaries.values()),
            "converted_objects": sum(item["converted_objects"] for item in split_summaries.values()),
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
        writer.writerows(cast(Iterable[Mapping[Literal['split', 'label', 'line', 'class_id', 'reason', 'details'], Any]], issue_rows))
    (results_root / "conversion_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/blade-small"))
    parser.add_argument("--output", type=Path, default=Path("datasets/blade-obb"))
    parser.add_argument("--results", type=Path, default=Path("results/obb"))
    parser.add_argument("--min-area", type=float, default=DEFAULT_MIN_AREA)
    parser.add_argument("--limit", type=int, help="maximum images per split (for smoke tests)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = convert_dataset(
        args.source, args.output, args.results, min_area=args.min_area, limit=args.limit
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
