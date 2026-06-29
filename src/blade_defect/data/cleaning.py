"""Conservative image and annotation cleaning utilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from blade_defect.utils.files import find_images, find_labels
from blade_defect.utils.paths import resolve_path


@dataclass
class CleaningReport:
    scanned: int = 0
    corrupt_images: list[str] = field(default_factory=list)
    duplicate_images: list[str] = field(default_factory=list)
    empty_labels: list[str] = field(default_factory=list)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_dataset(images_dir: str | Path, labels_dir: str | Path) -> CleaningReport:
    """Scan for common problems without deleting user data."""
    images_root, labels_root = resolve_path(images_dir), resolve_path(labels_dir)
    report = CleaningReport()
    hashes: dict[str, Path] = {}

    for image_path in find_images(images_root):
        report.scanned += 1
        try:
            with Image.open(image_path) as image:
                image.verify()
        except (OSError, UnidentifiedImageError):
            report.corrupt_images.append(str(image_path))
            continue
        digest = _sha256(image_path)
        if digest in hashes:
            report.duplicate_images.append(f"{image_path} == {hashes[digest]}")
        else:
            hashes[digest] = image_path

    for label_path in find_labels(labels_root):
        if not label_path.read_text(encoding="utf-8").strip():
            report.empty_labels.append(str(label_path))
    return report
