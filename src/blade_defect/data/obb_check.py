"""Strict validation for YOLO oriented bounding-box datasets."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from blade_defect.utils.files import find_images, find_labels
from blade_defect.utils.paths import resolve_path


@dataclass(frozen=True)
class OBBDatasetIssue:
    file: str
    error_type: str
    message: str
    line: int | None = None


@dataclass
class OBBDatasetReport:
    images: int = 0
    labels: int = 0
    valid_instances: int = 0
    negative_images: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    corrupt_images: list[str] = field(default_factory=list)
    issues: list[OBBDatasetIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def error_type_counts(self) -> dict[str, int]:
        return dict(sorted(Counter(issue.error_type for issue in self.issues).items()))

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "images": self.images,
            "labels": self.labels,
            "valid_instances": self.valid_instances,
            "negative_images": self.negative_images,
            "missing_labels": self.missing_labels,
            "orphan_labels": self.orphan_labels,
            "corrupt_images": self.corrupt_images,
            "error_type_counts": self.error_type_counts,
            "issues": [asdict(issue) for issue in self.issues],
        }


def _relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).with_suffix("").as_posix().casefold()


def _is_decodable(path: Path) -> bool:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        return encoded.size > 0 and cv2.imdecode(encoded, cv2.IMREAD_COLOR) is not None
    except (OSError, ValueError, cv2.error):
        return False


def _line_error(
    tokens: list[str], num_classes: int, min_area: float,
) -> tuple[str, str] | None:
    if len(tokens) != 9:
        return "obb_field_count", f"expected 9 fields, found {len(tokens)}"
    try:
        class_value = float(tokens[0])
        class_id = int(class_value)
        if not np.isfinite(class_value) or class_value != class_id or not 0 <= class_id < num_classes:
            return "obb_class_id", f"class id must be an integer in [0, {num_classes - 1}]"
        coordinates = np.asarray([float(token) for token in tokens[1:]], dtype=np.float64)
    except (ValueError, OverflowError):
        return "obb_numeric_format", "class id and coordinates must be finite numbers"
    if not np.isfinite(coordinates).all():
        return "obb_numeric_format", "coordinates contain NaN or infinity"
    if np.any((coordinates < 0.0) | (coordinates > 1.0)):
        return "obb_coordinate_range", "all eight coordinates must be in [0, 1]"

    points = coordinates.reshape(4, 2)
    unique = {tuple(np.round(point, 12)) for point in points}
    if len(unique) != 4:
        return "obb_duplicate_points", "the four OBB corners must be distinct"
    signed_area = float(cv2.contourArea(points.astype(np.float32), oriented=True))
    if abs(signed_area) < min_area:
        return "obb_area_too_small", f"quadrilateral area {abs(signed_area):.12g} is below {min_area:.12g}"
    if signed_area <= 0:
        return "obb_point_order", "corners must be clockwise in image coordinates"
    minimum_y = float(points[:, 1].min())
    candidates = np.flatnonzero(points[:, 1] == minimum_y)
    expected_start = int(candidates[np.argmin(points[candidates, 0])])
    if expected_start != 0:
        return "obb_point_order", "first corner must be the top-most, then left-most corner"
    return None


def check_obb_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    *,
    num_classes: int = 15,
    min_area: float = 1e-8,
) -> OBBDatasetReport:
    """Validate pairing, image integrity and every YOLO-OBB label row."""
    images_root = resolve_path(images_dir)
    labels_root = resolve_path(labels_dir)
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if min_area <= 0:
        raise ValueError("min_area must be positive")
    if not images_root.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {images_root}")
    if not labels_root.is_dir():
        raise FileNotFoundError(f"label directory does not exist: {labels_root}")

    images = find_images(images_root)
    labels = find_labels(labels_root)
    images_by_key = {_relative_key(path, images_root): path for path in images}
    labels_by_key = {_relative_key(path, labels_root): path for path in labels}
    report = OBBDatasetReport(images=len(images), labels=len(labels))

    for key, image_path in images_by_key.items():
        relative_image = image_path.relative_to(images_root).as_posix()
        if not _is_decodable(image_path):
            report.corrupt_images.append(relative_image)
            report.issues.append(
                OBBDatasetIssue(relative_image, "corrupt_image", "OpenCV could not decode image")
            )
        if key not in labels_by_key:
            report.missing_labels.append(relative_image)
            report.issues.append(
                OBBDatasetIssue(relative_image, "missing_label", "image has no matching label")
            )

    for key, label_path in labels_by_key.items():
        relative_label = label_path.relative_to(labels_root).as_posix()
        if key not in images_by_key:
            report.orphan_labels.append(relative_label)
            report.issues.append(
                OBBDatasetIssue(relative_label, "orphan_label", "label has no matching image")
            )
            continue
        text = label_path.read_text(encoding="utf-8-sig")
        nonempty_lines = [(number, line.split()) for number, line in enumerate(text.splitlines(), 1) if line.split()]
        if not nonempty_lines:
            report.negative_images += 1
            continue
        for line_number, tokens in nonempty_lines:
            error = _line_error(tokens, num_classes, min_area)
            if error is None:
                report.valid_instances += 1
            else:
                error_type, message = error
                report.issues.append(
                    OBBDatasetIssue(relative_label, error_type, message, line_number)
                )
    return report


__all__ = ["OBBDatasetIssue", "OBBDatasetReport", "check_obb_dataset"]
