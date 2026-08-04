"""Portable loading helpers for YOLO txt split indexes."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from blade_defect.utils.files import load_yaml


@dataclass(frozen=True)
class IndexedSample:
    """One sample selected by a target split index."""

    target_split: str
    index_file: Path
    line_number: int
    raw_entry: str
    image_path: Path
    label_path: Path
    sample_id: str


def _absolute(path: str | Path, base: Path) -> Path:
    candidate = Path(os.path.expandvars(str(path).strip()))
    if not candidate.is_absolute():
        candidate = base / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def load_index_config(data_yaml: str | Path) -> tuple[Path, dict[str, Any]]:
    """Load a data yaml without resolving junctions or symbolic links."""
    config_path = Path(os.path.abspath(os.fspath(data_yaml)))
    config = dict(load_yaml(config_path))
    dataset_root = _absolute(config.get("path", "."), config_path.parent)
    config["path"] = dataset_root
    return config_path, config


def resolve_index_entry(index_file: Path, raw_entry: str) -> Path:
    """Resolve relative entries from the txt file location, as Ultralytics does."""
    return _absolute(raw_entry, index_file.parent)


def label_path_for_image(image_path: Path) -> Path:
    """Derive a label path by replacing the first ``images`` path component."""
    parts = image_path.parts
    lowered = [part.casefold() for part in parts]
    try:
        position = lowered.index("images")
    except ValueError as exc:
        raise ValueError(f"image path has no 'images' component: {image_path}") from exc
    return Path(*parts[:position], "labels", *parts[position + 1 :]).with_suffix(".txt")


def stable_sample_id(image_path: Path) -> str:
    """Return an identity stable across release-index locations and drive letters."""
    parts = image_path.parts
    lowered = [part.casefold() for part in parts]
    try:
        position = lowered.index("images")
    except ValueError:
        return image_path.name.casefold()
    return Path(*parts[position:]).as_posix().casefold()


def load_indexed_split(data_yaml: str | Path, split: str) -> list[IndexedSample]:
    """Load one txt-backed split from a dataset yaml."""
    _, config = load_index_config(data_yaml)
    entry = config.get(split)
    if not isinstance(entry, (str, Path)):
        raise KeyError(f"split is not declared: {split}")
    index_file = _absolute(entry, Path(config["path"]))
    if index_file.suffix.casefold() != ".txt" or not index_file.is_file():
        raise FileNotFoundError(f"split index not found: {index_file}")

    samples: list[IndexedSample] = []
    for line_number, raw_line in enumerate(index_file.read_text(encoding="utf-8-sig").splitlines(), 1):
        raw_entry = raw_line.strip()
        if not raw_entry:
            continue
        image_path = resolve_index_entry(index_file, raw_entry)
        samples.append(
            IndexedSample(
                target_split=split,
                index_file=index_file,
                line_number=line_number,
                raw_entry=raw_entry,
                image_path=image_path,
                label_path=label_path_for_image(image_path),
                sample_id=stable_sample_id(image_path),
            )
        )
    return samples


def load_indexed_splits(
    data_yaml: str | Path,
    splits: Iterable[str] = ("train", "val"),
    *,
    require_all: bool = True,
) -> dict[str, list[IndexedSample]]:
    """Load several indexed splits while preserving target split membership."""
    loaded: dict[str, list[IndexedSample]] = {}
    for split in splits:
        try:
            loaded[split] = load_indexed_split(data_yaml, split)
        except (KeyError, FileNotFoundError):
            if require_all:
                raise
    return loaded


def membership_hash(samples_by_split: dict[str, list[IndexedSample]]) -> str:
    """Hash target split plus stable sample IDs in deterministic order."""
    digest = hashlib.sha256()
    for split in sorted(samples_by_split):
        for sample in samples_by_split[split]:
            digest.update(f"{split}\t{sample.sample_id}\n".encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "IndexedSample",
    "label_path_for_image",
    "load_index_config",
    "load_indexed_split",
    "load_indexed_splits",
    "membership_hash",
    "resolve_index_entry",
    "stable_sample_id",
]
