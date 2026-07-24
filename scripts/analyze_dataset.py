"""Generate reusable statistics and charts for a filtered YOLO-seg dataset.

The source image and label directories are read-only.  Soft coordinate overflow
within [-0.01, 1.01] is clipped in memory for geometry statistics; source label
files are never rewritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SOFT_COORDINATE_MIN = -0.01
SOFT_COORDINATE_MAX = 1.01
SMALL_AREA_MAX = 32**2
MEDIUM_AREA_MAX = 96**2


@dataclass(frozen=True)
class ParsedPolygon:
    class_id: int
    line_number: int
    coordinates: tuple[float, ...]
    geometry_status: str


@dataclass(frozen=True)
class LabelParseResult:
    class_ids: tuple[int, ...]
    polygons: tuple[ParsedPolygon, ...]
    errors: tuple[dict[str, Any], ...]
    coordinate_issues: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="统计 YOLO-seg 数据集的类别、几何和质量分布")
    parser.add_argument("--images", required=True, type=Path, help="图片根目录，例如 D:\\images")
    parser.add_argument("--labels", required=True, type=Path, help="标签根目录，例如 D:\\labels")
    parser.add_argument("--data", required=True, type=Path, help="包含 names 的 data.yaml")
    parser.add_argument("--output", required=True, type=Path, help="统计结果输出目录")
    parser.add_argument(
        "--filter-config",
        type=Path,
        help="可选；按文件名应用 exclude、review 和 keep_negative 规则",
    )
    parser.add_argument("--workers", type=int, default=16, help="并行读取线程数，默认16")
    parser.add_argument("--skip-charts", action="store_true", help="只生成 CSV/JSON，不生成 PNG 图表")
    return parser.parse_args()


def load_class_names(config_path: Path) -> dict[int, str]:
    text = config_path.read_text(encoding="utf-8-sig")
    names: dict[int, str] = {}
    in_names = False
    names_indent = 0
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        if stripped == "names:":
            in_names = True
            names_indent = indent
            continue
        if in_names and indent <= names_indent:
            break
        if in_names:
            match = re.match(r"(\d+)\s*:\s*(.+?)\s*$", stripped)
            if match:
                names[int(match.group(1))] = match.group(2).strip().strip("\"'")
    if not names:
        raise ValueError(f"{config_path} 中缺少有效的 names 映射配置")
    if sorted(names) != list(range(len(names))):
        raise ValueError("类别 ID 必须从 0 开始且连续")
    return names


def load_filter_decisions(config_path: Path) -> dict[str, dict[str, str]]:
    """Read the images list in dataset_filter.yaml without a YAML dependency."""

    decisions: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    in_images = False

    def commit(entry: dict[str, str] | None) -> None:
        if not entry:
            return
        filename = entry.get("filename", "").strip()
        action = entry.get("action", "").strip()
        if not filename or not action:
            raise ValueError("dataset filter 的每个 images 项都必须包含 filename 和 action")
        if action not in {"exclude", "review", "keep_negative"}:
            raise ValueError(f"{filename} 使用了不支持的 action：{action}")
        key = Path(filename).name.casefold()
        if key in decisions:
            raise ValueError(f"dataset filter 包含重复文件名：{filename}")
        decisions[key] = dict(entry)

    for raw_line in config_path.read_text(encoding="utf-8-sig").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip())
        if indent == 0:
            if in_images:
                commit(current)
                current = None
            in_images = stripped == "images:"
            continue
        if not in_images:
            continue
        if stripped.startswith("- "):
            commit(current)
            current = {}
            stripped = stripped[2:].strip()
        if current is None or ":" not in stripped:
            continue
        key, value = stripped.split(":", maxsplit=1)
        current[key.strip()] = value.strip().strip("\"'")
    if in_images:
        commit(current)
    return decisions


def class_group(class_name: str) -> str:
    return class_name.split("--", maxsplit=1)[0]


def relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix().casefold()


def collect_files(root: Path, extensions: set[str]) -> tuple[dict[str, Path], list[str]]:
    files: dict[str, Path] = {}
    duplicates: list[str] = []
    for directory, _, filenames in os.walk(root):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            if path.suffix.casefold() not in extensions:
                continue
            key = relative_key(path, root)
            if key in files:
                duplicates.append(path.relative_to(root).as_posix())
            else:
                files[key] = path
    return files, sorted(duplicates)


def coordinate_offset(value: float) -> float:
    if value < 0:
        return -value
    if value > 1:
        return value - 1
    return 0.0


def make_error(
    path: Path,
    line_number: int,
    reason: str,
    issue_type: str,
    severity: str,
    class_id: int | None = None,
    coordinates: Iterable[float] | None = None,
) -> dict[str, Any]:
    values = list(coordinates or [])
    return {
        "file": str(path),
        "line": line_number,
        "class_id": "" if class_id is None else class_id,
        "issue_type": issue_type,
        "severity": severity,
        "reason": reason,
        "min_coordinate": min(values) if values else "",
        "max_coordinate": max(values) if values else "",
        "max_offset": max((coordinate_offset(value) for value in values), default=""),
    }


def parse_label(path: Path, class_names: dict[int, str]) -> LabelParseResult:
    class_ids: list[int] = []
    polygons: list[ParsedPolygon] = []
    errors: list[dict[str, Any]] = []
    coordinate_issues: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        tokens = line.split()
        class_id: int | None = None
        try:
            class_value = float(tokens[0])
            if not class_value.is_integer():
                raise ValueError("类别 ID 不是整数")
            class_id = int(class_value)
            if class_id not in class_names:
                raise ValueError(f"类别 ID {class_id} 不在配置范围内")
        except (IndexError, ValueError) as error:
            errors.append(make_error(path, line_number, str(error), "class_id", "hard"))
            continue

        class_ids.append(class_id)
        if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
            errors.append(
                make_error(
                    path,
                    line_number,
                    "多边形至少需要3个点，且坐标必须成对",
                    "polygon_coordinate_count",
                    "hard",
                    class_id,
                )
            )
            continue
        try:
            coordinates = [float(value) for value in tokens[1:]]
        except ValueError:
            errors.append(
                make_error(path, line_number, "多边形包含非数值坐标", "polygon_coordinate_value", "hard", class_id)
            )
            continue

        out_of_range = [value for value in coordinates if not 0 <= value <= 1]
        geometry_status = "valid"
        if out_of_range:
            soft = all(SOFT_COORDINATE_MIN <= value <= SOFT_COORDINATE_MAX for value in coordinates)
            severity = "soft" if soft else "hard"
            for coordinate_index, value in enumerate(coordinates, start=1):
                if 0 <= value <= 1:
                    continue
                boundary = 0.0 if value < 0 else 1.0
                point_index = (coordinate_index + 1) // 2
                axis = "x" if coordinate_index % 2 == 1 else "y"
                coordinate_severity = (
                    "soft" if SOFT_COORDINATE_MIN <= value <= SOFT_COORDINATE_MAX else "hard"
                )
                coordinate_issues.append(
                    {
                        "file": str(path),
                        "line": line_number,
                        "class_id": class_id,
                        "class_name": class_names[class_id],
                        "point_index": point_index,
                        "axis": axis,
                        "coordinate_position": f"point_{point_index}_{axis}",
                        "coordinate_index": coordinate_index,
                        "original_value": value,
                        "boundary": boundary,
                        "overflow_direction": "below_0" if value < 0 else "above_1",
                        "overflow_amount": coordinate_offset(value),
                        "coordinate_severity": coordinate_severity,
                        "line_severity": severity,
                        "reset_value": boundary if soft else "",
                        "action": "reset_to_boundary" if soft else "manual_review_then_fix_or_filter",
                    }
                )
            errors.append(
                make_error(
                    path,
                    line_number,
                    "坐标轻微越界，可在统计时裁剪" if soft else "坐标严重越界，超出可裁剪范围",
                    "polygon_coordinate_range",
                    severity,
                    class_id,
                    coordinates,
                )
            )
            if not soft:
                continue
            coordinates = [min(1.0, max(0.0, value)) for value in coordinates]
            geometry_status = "soft_clipped"

        points = list(zip(coordinates[::2], coordinates[1::2], strict=True))
        if len(set(points)) < 3:
            errors.append(
                make_error(
                    path,
                    line_number,
                    "多边形少于3个不同的点",
                    "polygon_geometry",
                    "hard",
                    class_id,
                    coordinates,
                )
            )
            continue
        if polygon_area_relative(coordinates) <= 0:
            errors.append(
                make_error(path, line_number, "多边形面积为0", "polygon_geometry", "hard", class_id, coordinates)
            )
            continue
        polygons.append(ParsedPolygon(class_id, line_number, tuple(coordinates), geometry_status))
    return LabelParseResult(
        tuple(class_ids), tuple(polygons), tuple(errors), tuple(coordinate_issues)
    )


def polygon_area_relative(coordinates: Iterable[float]) -> float:
    values = list(coordinates)
    points = list(zip(values[::2], values[1::2], strict=True))
    area_twice = sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1])
    )
    return abs(area_twice) / 2.0


def polygon_geometry(coordinates: Iterable[float], width: int, height: int) -> dict[str, float | str]:
    values = list(coordinates)
    xs = values[::2]
    ys = values[1::2]
    bbox_width_relative = max(xs) - min(xs)
    bbox_height_relative = max(ys) - min(ys)
    relative_bbox_area = bbox_width_relative * bbox_height_relative
    relative_mask_area = polygon_area_relative(values)
    bbox_width_px = bbox_width_relative * width
    bbox_height_px = bbox_height_relative * height
    bbox_area_px = bbox_width_px * bbox_height_px
    mask_area_px = relative_mask_area * width * height
    if bbox_area_px < SMALL_AREA_MAX:
        size_category = "small"
    elif bbox_area_px < MEDIUM_AREA_MAX:
        size_category = "medium"
    else:
        size_category = "large"
    return {
        "bbox_width_px": bbox_width_px,
        "bbox_height_px": bbox_height_px,
        "bbox_area_px": bbox_area_px,
        "relative_bbox_area": relative_bbox_area,
        "mask_area_px": mask_area_px,
        "relative_mask_area": relative_mask_area,
        "size_category": size_category,
    }


def read_image_size(path: Path) -> tuple[int, int, str]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("图片尺寸必须为正数")
        return width, height, ""
    except Exception as error:  # Pillow exposes several decoder-specific exception types.
        return 0, 0, str(error)


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_class_csv(path: Path, class_names: dict[int, str], split_counts: dict[str, Counter[int]]) -> None:
    splits = sorted(split_counts)
    rows: list[dict[str, Any]] = []
    for class_id, class_name in class_names.items():
        row: dict[str, Any] = {
            "class_id": class_id,
            "class_name": class_name,
            "group": class_group(class_name),
        }
        row.update({split: split_counts[split][class_id] for split in splits})
        row["total"] = sum(split_counts[split][class_id] for split in splits)
        rows.append(row)
    write_csv(path, ["class_id", "class_name", "group", *splits, "total"], rows)


def safe_mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def safe_median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def build_object_size_rows(
    object_records: list[dict[str, Any]], class_names: dict[int, str], split_names: list[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in [*split_names, "total"]:
        split_records = object_records if split == "total" else [row for row in object_records if row["split"] == split]
        by_class: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for record in split_records:
            by_class[int(record["class_id"])].append(record)
        for class_id, class_name in class_names.items():
            records = by_class[class_id]
            counts = Counter(str(record["size_category"]) for record in records)
            total = len(records)
            relative_bbox = [float(record["relative_bbox_area"]) for record in records]
            relative_mask = [float(record["relative_mask_area"]) for record in records]
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "group": class_group(class_name),
                    "small_count": counts["small"],
                    "medium_count": counts["medium"],
                    "large_count": counts["large"],
                    "total_geometry_instances": total,
                    "small_ratio": counts["small"] / total if total else 0,
                    "medium_ratio": counts["medium"] / total if total else 0,
                    "large_ratio": counts["large"] / total if total else 0,
                    "mean_relative_bbox_area": safe_mean(relative_bbox),
                    "median_relative_bbox_area": safe_median(relative_bbox),
                    "mean_relative_mask_area": safe_mean(relative_mask),
                    "median_relative_mask_area": safe_median(relative_mask),
                }
            )
    return rows


def load_font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def save_horizontal_bars(
    path: Path,
    title: str,
    labels: list[str],
    series: list[tuple[str, list[int], str]],
    x_label: str,
) -> None:
    from PIL import Image, ImageDraw

    width = 1900
    row_height = 52
    top = 150
    bottom = 120
    left = 610
    right = 100
    height = top + bottom + max(1, len(labels)) * row_height
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(38, bold=True)
    label_font = load_font(23)
    value_font = load_font(20)
    legend_font = load_font(22)
    draw.text((60, 42), title, fill="#17365D", font=title_font)
    chart_width = width - left - right
    totals = [sum(values[index] for _, values, _ in series) for index in range(len(labels))]
    maximum = max(totals, default=1) or 1
    for index, label in enumerate(labels):
        y = top + index * row_height
        draw.text((40, y + 9), label, fill="#222222", font=label_font)
        x = left
        for _, values, color in series:
            value = values[index]
            segment_width = int(chart_width * value / maximum)
            if value and segment_width < 2:
                segment_width = 2
            if segment_width:
                draw.rectangle((x, y + 8, x + segment_width, y + 38), fill=color)
            x += segment_width
        draw.text((x + 10, y + 10), f"{totals[index]:,}", fill="#333333", font=value_font)
    legend_x = left
    for name, _, color in series:
        draw.rectangle((legend_x, height - 72, legend_x + 26, height - 46), fill=color)
        draw.text((legend_x + 36, height - 74), name, fill="#333333", font=legend_font)
        legend_x += 200
    draw.text((left + chart_width // 2 - 80, height - 110), x_label, fill="#555555", font=legend_font)
    image.save(path)


def save_histogram(
    path: Path,
    title: str,
    labels: list[str],
    counts: list[int],
    x_label: str,
    y_label: str = "数量",
) -> None:
    from PIL import Image, ImageDraw

    width, height = 1800, 1050
    left, right, top, bottom = 150, 90, 150, 190
    chart_width = width - left - right
    chart_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(38, bold=True)
    axis_font = load_font(22)
    value_font = load_font(18)
    draw.text((60, 42), title, fill="#17365D", font=title_font)
    draw.line((left, top, left, top + chart_height), fill="#666666", width=2)
    draw.line((left, top + chart_height, left + chart_width, top + chart_height), fill="#666666", width=2)
    maximum = max(counts, default=1) or 1
    gap = 16
    bar_width = max(10, (chart_width - gap * (len(labels) + 1)) // max(1, len(labels)))
    for index, (label, count) in enumerate(zip(labels, counts)):
        x1 = left + gap + index * (bar_width + gap)
        bar_height = int(chart_height * count / maximum)
        y1 = top + chart_height - bar_height
        draw.rectangle((x1, y1, x1 + bar_width, top + chart_height), fill="#4472C4")
        value_text = f"{count:,}"
        box = draw.textbbox((0, 0), value_text, font=value_font)
        draw.text((x1 + (bar_width - (box[2] - box[0])) / 2, max(top, y1 - 28)), value_text, fill="#333333", font=value_font)
        label_box = draw.textbbox((0, 0), label, font=axis_font)
        draw.text(
            (x1 + (bar_width - (label_box[2] - label_box[0])) / 2, top + chart_height + 22),
            label,
            fill="#333333",
            font=axis_font,
        )
    draw.text((left + chart_width // 2 - 100, height - 78), x_label, fill="#555555", font=axis_font)
    draw.text((30, top + chart_height // 2), y_label, fill="#555555", font=axis_font)
    image.save(path)


def generate_charts(
    output: Path,
    class_names: dict[int, str],
    instance_distribution: dict[str, Counter[int]],
    object_records: list[dict[str, Any]],
    image_records: list[dict[str, Any]],
) -> None:
    labels = [f"{class_id} {name}" for class_id, name in class_names.items()]
    class_values = [sum(counter[class_id] for counter in instance_distribution.values()) for class_id in class_names]
    save_horizontal_bars(
        output / "class_distribution.png",
        "各缺陷类别实例数量",
        labels,
        [("实例数", class_values, "#4472C4")],
        "实例数量",
    )

    size_counts = {size: Counter() for size in ("small", "medium", "large")}
    for record in object_records:
        size_counts[str(record["size_category"])][int(record["class_id"])] += 1
    save_horizontal_bars(
        output / "object_size_distribution.png",
        "各类别小、中、大目标分布（COCO bbox标准）",
        labels,
        [
            ("小目标", [size_counts["small"][class_id] for class_id in class_names], "#5B9BD5"),
            ("中目标", [size_counts["medium"][class_id] for class_id in class_names], "#ED7D31"),
            ("大目标", [size_counts["large"][class_id] for class_id in class_names], "#70AD47"),
        ],
        "可计算几何信息的实例数量",
    )

    mask_bins = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 1.0000001]
    mask_labels = ["<0.01%", "0.01–0.05%", "0.05–0.1%", "0.1–0.5%", "0.5–1%", "1–5%", "5–10%", "≥10%"]
    mask_counts = [0] * (len(mask_bins) - 1)
    for record in object_records:
        value = float(record["relative_mask_area"])
        for index, (lower, upper) in enumerate(zip(mask_bins, mask_bins[1:])):
            if lower <= value < upper:
                mask_counts[index] += 1
                break
    save_histogram(
        output / "mask_area_distribution.png",
        "Mask相对图像面积分布",
        mask_labels,
        mask_counts,
        "Mask占整张图片的面积比例",
    )

    object_count_bins = [0, 0, 0, 0, 0, 0, 0]
    for record in image_records:
        count = int(record["instance_count"])
        object_count_bins[min(count, 6)] += 1
    save_histogram(
        output / "objects_per_image.png",
        "单张图片缺陷实例数量分布",
        ["0", "1", "2", "3", "4", "5", "≥6"],
        object_count_bins,
        "单图实例数",
        "图片数量",
    )


def analyze_dataset(
    images_root: Path,
    labels_root: Path,
    config_path: Path,
    output: Path,
    filter_config: Path | None = None,
    *,
    workers: int = 16,
    charts: bool = True,
) -> dict[str, Any]:
    for directory, label in ((images_root, "图片"), (labels_root, "标签")):
        if not directory.is_dir():
            raise FileNotFoundError(f"{label}目录不存在：{directory}")
    if workers < 1:
        raise ValueError("workers 必须大于等于1")

    class_names = load_class_names(config_path)
    filter_decisions = load_filter_decisions(filter_config) if filter_config is not None else {}
    matched_decisions: set[str] = set()
    split_names = sorted(
        path.name for path in images_root.iterdir() if path.is_dir() and (labels_root / path.name).is_dir()
    )
    if not split_names:
        raise ValueError("没有找到同时存在于图片和标签目录中的数据划分")

    image_distribution = {split: Counter() for split in split_names}
    instance_distribution = {split: Counter() for split in split_names}
    split_statistics: dict[str, dict[str, Any]] = {}
    invalid_lines: list[dict[str, Any]] = []
    coordinate_issue_records: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    object_records: list[dict[str, Any]] = []
    image_read_errors: list[dict[str, Any]] = []
    filtered_corrupt_records: list[dict[str, Any]] = []

    for split in split_names:
        image_root = images_root / split
        label_root = labels_root / split
        images, duplicate_images = collect_files(image_root, IMAGE_EXTENSIONS)
        labels, duplicate_labels = collect_files(label_root, {".txt"})
        raw_image_count = len(images)
        raw_label_count = len(labels)
        action_files = {action: [] for action in ("exclude", "review", "keep_negative")}
        negative_keys: set[str] = set()

        for key, image_path in list(images.items()):
            decision_key = image_path.name.casefold()
            decision = filter_decisions.get(decision_key)
            if decision is None:
                continue
            matched_decisions.add(decision_key)
            action = decision["action"]
            action_files[action].append(image_path.relative_to(image_root).as_posix())
            if decision.get("issue") == "corrupted_file":
                filtered_corrupt_records.append(
                    {
                        "split": split,
                        "image_file": image_path.relative_to(image_root).as_posix(),
                        "action": action,
                        "issue": "corrupted_file",
                        "reason": decision.get("reason", ""),
                        "status": decision.get("status", ""),
                    }
                )
            if action in {"exclude", "review"}:
                images.pop(key)
                labels.pop(key, None)
            elif action == "keep_negative":
                negative_keys.add(key)

        image_keys = set(images)
        label_keys = set(labels)
        effective_label_keys = label_keys | negative_keys
        paired_keys = sorted(image_keys & effective_label_keys)
        missing_labels = sorted(images[key].relative_to(image_root).as_posix() for key in image_keys - effective_label_keys)
        orphan_labels = sorted(labels[key].relative_to(label_root).as_posix() for key in label_keys - image_keys)

        label_items = [(key, labels[key]) for key in paired_keys if key not in negative_keys]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            parsed_values = executor.map(lambda item: (item[0], parse_label(item[1], class_names)), label_items)
        parsed_by_key = dict(parsed_values)

        image_items = sorted(images.items())
        with ThreadPoolExecutor(max_workers=workers) as executor:
            image_sizes = dict(executor.map(lambda item: (item[0], read_image_size(item[1])), image_items))

        empty_labels = len(negative_keys)
        instances = 0
        usable_polygon_instances = 0
        soft_clippable_instances = 0
        hard_invalid_lines = 0
        split_invalid_lines = 0
        split_coordinate_issues = 0
        split_soft_reset_coordinates = 0
        split_hard_review_coordinates = 0
        split_object_start = len(object_records)

        for key, result in parsed_by_key.items():
            split_errors = [dict(error, split=split) for error in result.errors]
            invalid_lines.extend(split_errors)
            split_coordinate_rows = [dict(issue, split=split) for issue in result.coordinate_issues]
            coordinate_issue_records.extend(split_coordinate_rows)
            split_coordinate_issues += len(split_coordinate_rows)
            split_soft_reset_coordinates += sum(
                issue["line_severity"] == "soft" for issue in split_coordinate_rows
            )
            split_hard_review_coordinates += sum(
                issue["line_severity"] == "hard" for issue in split_coordinate_rows
            )
            split_invalid_lines += len(split_errors)
            hard_invalid_lines += sum(error["severity"] == "hard" for error in split_errors)
            if not result.class_ids and not result.errors:
                empty_labels += 1
            instances += len(result.class_ids)
            usable_polygon_instances += sum(polygon.geometry_status == "valid" for polygon in result.polygons)
            soft_clippable_instances += sum(polygon.geometry_status == "soft_clipped" for polygon in result.polygons)
            instance_distribution[split].update(result.class_ids)
            image_distribution[split].update(set(result.class_ids))

        for key, image_path in image_items:
            width, height, image_error = image_sizes[key]
            result = parsed_by_key.get(key, LabelParseResult((), (), (), ()))
            relative_image = image_path.relative_to(image_root).as_posix()
            if image_error:
                image_read_errors.append({"split": split, "image_file": relative_image, "reason": image_error})
            image_records.append(
                {
                    "split": split,
                    "image_file": relative_image,
                    "width": width,
                    "height": height,
                    "aspect_ratio": width / height if height else 0,
                    "instance_count": len(result.class_ids),
                    "is_negative": key in negative_keys,
                    "has_effective_label": key in effective_label_keys,
                    "image_read_error": image_error,
                }
            )
            if image_error:
                continue
            label_path = labels.get(key)
            for polygon in result.polygons:
                geometry = polygon_geometry(polygon.coordinates, width, height)
                object_records.append(
                    {
                        "split": split,
                        "image_file": relative_image,
                        "label_file": label_path.relative_to(label_root).as_posix() if label_path else "",
                        "line": polygon.line_number,
                        "class_id": polygon.class_id,
                        "class_name": class_names[polygon.class_id],
                        "group": class_group(class_names[polygon.class_id]),
                        "geometry_status": polygon.geometry_status,
                        "image_width": width,
                        "image_height": height,
                        **geometry,
                    }
                )

        split_statistics[split] = {
            "raw_images": raw_image_count,
            "raw_labels": raw_label_count,
            "images": len(images),
            "labels": len(effective_label_keys),
            "paired": len(paired_keys),
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "empty_labels": empty_labels,
            "virtual_empty_labels": len(negative_keys - label_keys),
            "excluded_by_filter": len(action_files["exclude"]),
            "review_by_filter": len(action_files["review"]),
            "negative_by_filter": len(action_files["keep_negative"]),
            "corrupted_by_filter": sum(
                row["split"] == split for row in filtered_corrupt_records
            ),
            "instances": instances,
            "usable_polygon_instances": usable_polygon_instances,
            "soft_clippable_instances": soft_clippable_instances,
            "geometry_instances": usable_polygon_instances + soft_clippable_instances,
            "sized_geometry_instances": len(object_records) - split_object_start,
            "hard_invalid_lines": hard_invalid_lines,
            "invalid_lines": split_invalid_lines,
            "out_of_bounds_coordinates": split_coordinate_issues,
            "soft_reset_coordinates": split_soft_reset_coordinates,
            "hard_review_coordinates": split_hard_review_coordinates,
            "image_read_errors": sum(1 for row in image_read_errors if row["split"] == split),
            "missing_label_files": missing_labels,
            "orphan_label_files": orphan_labels,
            "excluded_image_files": sorted(action_files["exclude"]),
            "review_image_files": sorted(action_files["review"]),
            "negative_image_files": sorted(action_files["keep_negative"]),
            "duplicate_image_keys": duplicate_images,
            "duplicate_label_keys": duplicate_labels,
        }

    output.mkdir(parents=True, exist_ok=True)
    write_class_csv(output / "class_image_distribution.csv", class_names, image_distribution)
    write_class_csv(output / "class_instance_distribution.csv", class_names, instance_distribution)
    invalid_fields = [
        "split", "file", "line", "class_id", "issue_type", "severity", "reason",
        "min_coordinate", "max_coordinate", "max_offset",
    ]
    write_csv(output / "invalid_annotations.csv", invalid_fields, invalid_lines)
    coordinate_issue_fields = [
        "split", "file", "line", "class_id", "class_name", "point_index", "axis",
        "coordinate_position", "coordinate_index", "original_value", "boundary",
        "overflow_direction", "overflow_amount", "coordinate_severity", "line_severity",
        "reset_value", "action",
    ]
    write_csv(
        output / "out_of_bounds_coordinates.csv",
        coordinate_issue_fields,
        coordinate_issue_records,
    )
    hard_coordinate_records = [
        row for row in coordinate_issue_records if row["line_severity"] == "hard"
    ]
    write_csv(
        output / "hard_error_coordinates.csv",
        coordinate_issue_fields,
        hard_coordinate_records,
    )
    hard_class_rows: list[dict[str, Any]] = []
    for class_id in sorted({int(row["class_id"]) for row in hard_coordinate_records}):
        rows = [row for row in hard_coordinate_records if int(row["class_id"]) == class_id]
        affected_images = len({row["file"] for row in rows})
        total_class_images = sum(counts[class_id] for counts in image_distribution.values())
        total_class_instances = sum(counts[class_id] for counts in instance_distribution.values())
        hard_class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_names[class_id],
                "total_class_images": total_class_images,
                "total_class_instances": total_class_instances,
                "affected_images": affected_images,
                "affected_image_ratio": affected_images / total_class_images if total_class_images else 0,
                "hard_lines": len({(row["file"], row["line"]) for row in rows}),
                "hard_coordinates": len(rows),
                "minimum_value": min(float(row["original_value"]) for row in rows),
                "maximum_value": max(float(row["original_value"]) for row in rows),
                "maximum_overflow": max(float(row["overflow_amount"]) for row in rows),
            }
        )
    hard_class_fields = [
        "class_id", "class_name", "total_class_images", "total_class_instances",
        "affected_images", "affected_image_ratio", "hard_lines", "hard_coordinates",
        "minimum_value", "maximum_value", "maximum_overflow",
    ]
    write_csv(output / "hard_error_by_class.csv", hard_class_fields, hard_class_rows)
    image_fields = [
        "split", "image_file", "width", "height", "aspect_ratio", "instance_count",
        "is_negative", "has_effective_label", "image_read_error",
    ]
    write_csv(output / "image_statistics.csv", image_fields, image_records)
    object_fields = [
        "split", "image_file", "label_file", "line", "class_id", "class_name", "group",
        "geometry_status", "image_width", "image_height", "bbox_width_px", "bbox_height_px",
        "bbox_area_px", "relative_bbox_area", "mask_area_px", "relative_mask_area", "size_category",
    ]
    write_csv(output / "object_statistics.csv", object_fields, object_records)
    size_rows = build_object_size_rows(object_records, class_names, split_names)
    size_fields = [
        "split", "class_id", "class_name", "group", "small_count", "medium_count", "large_count",
        "total_geometry_instances", "small_ratio", "medium_ratio", "large_ratio",
        "mean_relative_bbox_area", "median_relative_bbox_area", "mean_relative_mask_area",
        "median_relative_mask_area",
    ]
    write_csv(output / "object_size_by_class.csv", size_fields, size_rows)
    write_csv(output / "image_read_errors.csv", ["split", "image_file", "reason"], image_read_errors)
    write_csv(
        output / "corrupted_images_by_filter.csv",
        ["split", "image_file", "action", "issue", "reason", "status"],
        filtered_corrupt_records,
    )

    totals = {
        key: sum(int(split_stats[key]) for split_stats in split_statistics.values())
        for key in (
            "raw_images", "raw_labels", "images", "labels", "paired", "missing_labels",
            "orphan_labels", "empty_labels", "virtual_empty_labels", "excluded_by_filter",
            "review_by_filter", "negative_by_filter", "instances", "usable_polygon_instances",
            "corrupted_by_filter",
            "soft_clippable_instances", "geometry_instances", "sized_geometry_instances",
            "hard_invalid_lines", "invalid_lines", "out_of_bounds_coordinates",
            "soft_reset_coordinates", "hard_review_coordinates", "image_read_errors",
        )
    }
    coordinate_errors = [row for row in invalid_lines if row["issue_type"] == "polygon_coordinate_range"]
    coordinate_values = [
        float(value)
        for row in coordinate_errors
        for value in (row["min_coordinate"], row["max_coordinate"])
        if value != ""
    ]
    size_counts = Counter(str(record["size_category"]) for record in object_records)
    object_counts = [int(record["instance_count"]) for record in image_records]
    widths = [int(record["width"]) for record in image_records if int(record["width"]) > 0]
    heights = [int(record["height"]) for record in image_records if int(record["height"]) > 0]
    unmatched_filter_files = sorted(
        decision["filename"] for key, decision in filter_decisions.items() if key not in matched_decisions
    )
    report: dict[str, Any] = {
        "sources": {
            "images": str(images_root.resolve()),
            "labels": str(labels_root.resolve()),
            "data_config": str(config_path.resolve()),
            "filter_config": str(filter_config.resolve()) if filter_config is not None else None,
        },
        "filter": {
            "enabled": filter_config is not None,
            "configured_images": len(filter_decisions),
            "matched_images": len(matched_decisions),
            "unmatched_config_files": unmatched_filter_files,
        },
        "class_count": len(class_names),
        "classes": [
            {"class_id": class_id, "class_name": name, "group": class_group(name)}
            for class_id, name in class_names.items()
        ],
        "splits": split_statistics,
        "totals": totals,
        "coordinate_anomalies": {
            "soft_range": [SOFT_COORDINATE_MIN, SOFT_COORDINATE_MAX],
            "soft_lines": sum(row["severity"] == "soft" for row in coordinate_errors),
            "hard_lines": sum(row["severity"] == "hard" for row in coordinate_errors),
            "affected_files": len({row["file"] for row in coordinate_errors}),
            "minimum_coordinate": min(coordinate_values) if coordinate_values else None,
            "maximum_coordinate": max(coordinate_values) if coordinate_values else None,
            "maximum_offset": max((float(row["max_offset"]) for row in coordinate_errors), default=0),
            "out_of_bounds_coordinate_count": len(coordinate_issue_records),
            "soft_reset_coordinate_count": sum(
                row["line_severity"] == "soft" for row in coordinate_issue_records
            ),
            "hard_review_coordinate_count": sum(
                row["line_severity"] == "hard" for row in coordinate_issue_records
            ),
            "below_zero_count": sum(
                row["overflow_direction"] == "below_0" for row in coordinate_issue_records
            ),
            "above_one_count": sum(
                row["overflow_direction"] == "above_1" for row in coordinate_issue_records
            ),
            "detail_file": str((output / "out_of_bounds_coordinates.csv").resolve()),
            "hard_affected_images": len({row["file"] for row in hard_coordinate_records}),
            "hard_affected_classes": len(hard_class_rows),
            "hard_minimum_value": min(
                (float(row["original_value"]) for row in hard_coordinate_records),
                default=None,
            ),
            "hard_maximum_value": max(
                (float(row["original_value"]) for row in hard_coordinate_records),
                default=None,
            ),
            "hard_maximum_overflow": max(
                (float(row["overflow_amount"]) for row in hard_coordinate_records),
                default=0,
            ),
            "hard_coordinate_file": str((output / "hard_error_coordinates.csv").resolve()),
            "hard_class_file": str((output / "hard_error_by_class.csv").resolve()),
            "hard_classes": hard_class_rows,
            "source_modified": False,
        },
        "image_dimensions": {
            "readable_images": len(widths),
            "minimum_width": min(widths) if widths else None,
            "maximum_width": max(widths) if widths else None,
            "minimum_height": min(heights) if heights else None,
            "maximum_height": max(heights) if heights else None,
            "median_width": safe_median([float(value) for value in widths]),
            "median_height": safe_median([float(value) for value in heights]),
        },
        "objects_per_image": {
            "mean": safe_mean([float(value) for value in object_counts]),
            "median": safe_median([float(value) for value in object_counts]),
            "maximum": max(object_counts, default=0),
            "zero_instance_images": sum(value == 0 for value in object_counts),
        },
        "object_size": {
            "standard": "COCO-style bbox pixel area",
            "small": f"area < {SMALL_AREA_MAX} (32^2)",
            "medium": f"{SMALL_AREA_MAX} <= area < {MEDIUM_AREA_MAX} (96^2)",
            "large": f"area >= {MEDIUM_AREA_MAX}",
            "small_count": size_counts["small"],
            "medium_count": size_counts["medium"],
            "large_count": size_counts["large"],
            "total_geometry_instances": len(object_records),
        },
        "invalid_line_samples": invalid_lines[:100],
        "notes": [
            "原始图片和标签目录只读，脚本不会修改源文件。",
            "筛选后的 images 和 labels 表示可用于后续处理的有效数量。",
            "keep_negative 按虚拟空标签计入有效标签数，但不会写入原始标签目录。",
            "轻微越界坐标只在内存中裁剪后用于几何统计，严重越界不进入几何统计。",
            "geometry_instances 表示坐标可计算的实例；sized_geometry_instances 还要求图片可读取，以便计算像素面积。",
        ],
    }
    (output / "dataset_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if charts:
        generate_charts(output, class_names, instance_distribution, object_records, image_records)
    return report


def main() -> None:
    args = parse_args()
    report = analyze_dataset(
        args.images,
        args.labels,
        args.data,
        args.output,
        filter_config=args.filter_config,
        workers=args.workers,
        charts=not args.skip_charts,
    )
    print(json.dumps(report["totals"], ensure_ascii=False, indent=2))
    print(f"统计结果已写入：{args.output.resolve()}")


if __name__ == "__main__":
    main()
