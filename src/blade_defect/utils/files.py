"""Filesystem and configuration helpers."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

from .paths import posix_path, resolve_config_paths, resolve_path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(directory: str | Path, recursive: bool = True) -> list[Path]:
    root = resolve_path(directory)
    iterator = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def find_labels(directory: str | Path) -> list[Path]:
    """Find YOLO label files case-insensitively on every platform."""
    root = resolve_path(directory)
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".txt")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with resolve_path(path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return data or {}


def load_project_config(
    path: str | Path,
    path_fields: tuple[str, ...] = ("data", "project"),
) -> tuple[dict[str, Any], Path]:
    """Load a project config and resolve its path fields to Path objects."""
    config_path = resolve_path(path)
    return resolve_config_paths(load_yaml(config_path), config_path, path_fields)


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    """Load data.yaml and expose dataset entries as absolute Path objects."""
    config_path = resolve_path(path)
    config = load_yaml(config_path)
    dataset_root = resolve_path(config.get("path", "."), config_path.parent)
    result = dict(config)
    result["path"] = dataset_root
    for field in ("train", "val", "test"):
        if result.get(field) is not None:
            result[field] = resolve_path(result[field], dataset_root)
    return result


def write_data_yaml(
    dataset_root: str | Path,
    output_path: str | Path,
    names: dict[int, str] | list[str] | None = None,
) -> Path:
    """Write a portable Ultralytics data.yaml using forward slashes."""
    output = resolve_path(output_path)
    root = resolve_path(dataset_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        serialized_root = Path(os.path.relpath(root, output.parent))
    except ValueError:  # Different Windows drives cannot be made relative.
        serialized_root = root
    payload: dict[str, Any] = {
        "path": posix_path(serialized_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }
    if names is not None:
        payload["names"] = names
    with output.open("w", encoding="utf-8", newline="\n") as file:
        yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)
    return output


@contextmanager
def resolved_data_yaml(path: str | Path) -> Iterator[str]:
    """Yield a temporary data YAML whose dataset root is absolute and portable."""
    source = resolve_path(path)
    if source.suffix.lower() not in {".yaml", ".yml"} or not source.is_file():
        yield str(source)
        return

    payload = load_yaml(source)
    payload["path"] = posix_path(resolve_path(payload.get("path", "."), source.parent))
    descriptor, temporary_name = tempfile.mkstemp(suffix=source.suffix)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            yaml.safe_dump(payload, file, allow_unicode=True, sort_keys=False)
        yield str(temporary)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(data: Any, path: str | Path) -> Path:
    output = resolve_path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2, default=lambda value: str(value))
    return output
