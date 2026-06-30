"""YOLO segmentation annotation validation."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from blade_defect.utils.files import find_images, find_labels
from blade_defect.utils.paths import resolve_path


@dataclass
class LabelIssue:
    file: str
    line: int
    status: str
    error_type: str
    message: str


@dataclass
class DatasetCheckReport:
    images: int = 0
    labels: int = 0
    valid_objects: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)
    issues: list[LabelIssue] = field(default_factory=list)
    fixes: list[LabelIssue] = field(default_factory=list)
    fixed_points: int = 0
    fixed_files: int = 0
    max_offset: float = 0.0
    fix_float_enabled: bool = False
    dry_run: bool = False

    @property
    def valid(self) -> bool:
        return not self.missing_labels and not self.orphan_labels and not self.issues

    @property
    def error_type_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter(issue.error_type for issue in self.issues)
        if self.missing_labels:
            counts["missing_label"] = len(self.missing_labels)
        if self.orphan_labels:
            counts["orphan_label"] = len(self.orphan_labels)
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict:
        result = asdict(self)
        result["valid"] = self.valid
        result["error_type_counts"] = self.error_type_counts
        return result


@dataclass(frozen=True)
class _LineError:
    error_type: str
    message: str


@dataclass(frozen=True)
class _LineResult:
    error: _LineError | None = None
    fixed_line: str | None = None
    fixed_points: int = 0
    max_offset: float = 0.0


def _inspect_seg_line(
    line: str,
    num_classes: int | None = None,
    fix_float: bool = False,
) -> _LineResult:
    parts = line.split()
    if not parts:
        return _LineResult(error=_LineError("polygon_format", "polygon 标注行为空"))

    try:
        class_value = float(parts[0])
    except ValueError:
        return _LineResult(error=_LineError("class_id", "class_id 必须为非负整数"))
    if not class_value.is_integer() or class_value < 0:
        return _LineResult(error=_LineError("class_id", "class_id 必须为非负整数"))
    class_id = int(class_value)
    if num_classes is not None and class_id >= num_classes:
        return _LineResult(
            error=_LineError("class_id", f"class_id={class_id} 超出类别范围 [0, {num_classes - 1}]")
        )

    try:
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return _LineResult(error=_LineError("polygon_non_numeric", "polygon 坐标包含非数值字段"))

    if len(coords) == 4:
        return _LineResult(
            error=_LineError(
                "bbox_format",
                "检测到 YOLO bbox 格式（class_id x_center y_center width height），期望 YOLO-seg polygon 点序列",
            )
        )
    if len(coords) % 2:
        return _LineResult(
            error=_LineError(
                "polygon_coordinate_count",
                "polygon 坐标数量必须为偶数，每两个值表示一个 (x, y) 点",
            )
        )
    if len(coords) < 6:
        return _LineResult(error=_LineError("polygon_coordinate_count", "polygon 至少需要 3 个 (x, y) 点"))

    points = list(zip(coords[::2], coords[1::2], strict=True))
    hard_points: list[tuple[int, float, float]] = []
    soft_points: list[tuple[int, float, float]] = []
    for point_number, (x, y) in enumerate(points, start=1):
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            if -0.01 <= x <= 1.01 and -0.01 <= y <= 1.01:
                soft_points.append((point_number, x, y))
            else:
                hard_points.append((point_number, x, y))

    if hard_points:
        point_number, x, y = hard_points[0]
        return _LineResult(
            error=_LineError(
                "polygon_coordinate_range",
                f"polygon 第 {point_number} 个点 (x={x}, y={y}) 的归一化坐标必须位于 [0, 1]，"
                "且已超出 soft clip 范围 [-0.01, 1.01]",
            )
        )
    if soft_points and not fix_float:
        point_number, x, y = soft_points[0]
        return _LineResult(
            error=_LineError(
                "polygon_coordinate_range",
                f"polygon 第 {point_number} 个点 (x={x}, y={y}) 的归一化坐标必须位于 [0, 1]",
            )
        )
    if soft_points:
        max_offset = 0.0
        clipped_coords: list[float] = []
        for coord_index, value in enumerate(coords, start=1):
            clipped = min(1.0, max(0.0, value))
            clipped_coords.append(clipped)
            if clipped != value:
                parts[coord_index] = "0" if clipped == 0.0 else "1"
                max_offset = max(max_offset, abs(value - clipped))
        clipped_points = set(zip(clipped_coords[::2], clipped_coords[1::2], strict=True))
        if len(clipped_points) < 3:
            return _LineResult(
                error=_LineError("polygon_geometry", "soft clip 后 polygon 少于 3 个不同的点，不执行修复")
            )
        return _LineResult(
            fixed_line=" ".join(parts),
            fixed_points=len(soft_points),
            max_offset=max_offset,
        )

    if len(set(points)) < 3:
        return _LineResult(error=_LineError("polygon_geometry", "polygon 至少需要 3 个不同的点"))
    return _LineResult()


def _validate_seg_line(line: str, num_classes: int | None = None) -> _LineError | None:
    return _inspect_seg_line(line, num_classes).error


def validate_seg_line(line: str, num_classes: int | None = None) -> str | None:
    """Validate one ``class_id x1 y1 ... xn yn`` YOLO-seg row."""
    error = _validate_seg_line(line, num_classes)
    return error.message if error else None


def check_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    num_classes: int | None = None,
    fix_float: bool = False,
    dry_run: bool = False,
) -> DatasetCheckReport:
    images_root, labels_root = resolve_path(images_dir), resolve_path(labels_dir)
    images = find_images(images_root)
    labels = find_labels(labels_root)
    report = DatasetCheckReport(
        images=len(images),
        labels=len(labels),
        fix_float_enabled=fix_float,
        dry_run=dry_run,
    )

    image_keys = {
        path.relative_to(images_root).with_suffix("").as_posix().casefold(): path
        for path in images
    }
    label_keys = {
        path.relative_to(labels_root).with_suffix("").as_posix().casefold(): path
        for path in labels
    }
    report.missing_labels = [
        image_keys[key].relative_to(images_root).with_suffix("").as_posix()
        for key in sorted(image_keys.keys() - label_keys.keys())
    ]
    report.orphan_labels = [
        label_keys[key].relative_to(labels_root).with_suffix("").as_posix()
        for key in sorted(label_keys.keys() - image_keys.keys())
    ]

    for label_path in labels:
        relative = label_path.relative_to(labels_root).as_posix()
        content = label_path.read_text(encoding="utf-8")
        if not content.strip():
            report.empty_labels.append(relative)
            continue
        raw_lines = content.splitlines(keepends=True)
        updated_lines = list(raw_lines)
        file_fixed = False
        for line_number, raw_line in enumerate(raw_lines, 1):
            line = raw_line.strip()
            if not line:
                continue
            result = _inspect_seg_line(line, num_classes, fix_float=fix_float)
            if result.error:
                report.issues.append(
                    LabelIssue(
                        relative,
                        line_number,
                        "HARD_ERROR",
                        result.error.error_type,
                        result.error.message,
                    )
                )
            else:
                report.valid_objects += 1
                if result.fixed_line is not None:
                    file_fixed = True
                    report.fixed_points += result.fixed_points
                    report.max_offset = max(report.max_offset, result.max_offset)
                    report.fixes.append(
                        LabelIssue(
                            relative,
                            line_number,
                            "FIXED_FLOAT_ERROR",
                            "polygon_coordinate_range",
                            f"soft clip {result.fixed_points} 个 polygon 点，最大偏移量={result.max_offset}",
                        )
                    )
                    line_ending = "\r\n" if raw_line.endswith("\r\n") else "\n" if raw_line.endswith("\n") else ""
                    indentation = raw_line[: len(raw_line) - len(raw_line.lstrip())]
                    updated_lines[line_number - 1] = indentation + result.fixed_line + line_ending
        if file_fixed:
            report.fixed_files += 1
            if not dry_run:
                with label_path.open("w", encoding="utf-8", newline="") as file:
                    file.write("".join(updated_lines))
    return report
