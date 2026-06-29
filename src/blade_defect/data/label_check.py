"""YOLO segmentation annotation validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from blade_defect.utils.files import find_images, find_labels
from blade_defect.utils.paths import resolve_path


@dataclass
class LabelIssue:
    file: str
    line: int
    message: str


@dataclass
class DatasetCheckReport:
    images: int = 0
    labels: int = 0
    valid_objects: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    issues: list[LabelIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.missing_labels and not self.orphan_labels and not self.issues

    def to_dict(self) -> dict:
        result = asdict(self)
        result["valid"] = self.valid
        return result


def validate_seg_line(line: str, num_classes: int | None = None) -> str | None:
    parts = line.split()
    if len(parts) < 7:
        return "分割标注至少需要 class_id 和 3 个坐标点"
    if (len(parts) - 1) % 2:
        return "多边形坐标数量必须为偶数"
    try:
        class_value = float(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError:
        return "存在非数值字段"
    class_id = int(class_value)
    if class_value != class_id or class_id < 0:
        return "class_id 必须为非负整数"
    if num_classes is not None and class_id >= num_classes:
        return f"class_id={class_id} 超出类别范围 [0, {num_classes - 1}]"
    if any(value < 0.0 or value > 1.0 for value in coords):
        return "归一化坐标必须位于 [0, 1]"
    points = list(zip(coords[::2], coords[1::2]))
    if len(set(points)) < 3:
        return "多边形至少需要 3 个不同的点"
    return None


def check_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    num_classes: int | None = None,
) -> DatasetCheckReport:
    images_root, labels_root = resolve_path(images_dir), resolve_path(labels_dir)
    images = find_images(images_root)
    labels = find_labels(labels_root)
    report = DatasetCheckReport(images=len(images), labels=len(labels))

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
        for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw_line.strip()
            if not line:
                continue
            issue = validate_seg_line(line, num_classes)
            if issue:
                report.issues.append(LabelIssue(relative, line_number, issue))
            else:
                report.valid_objects += 1
    return report
