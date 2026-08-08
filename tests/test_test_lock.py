from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from scripts.audit_test_lock import audit_test_lock


def _dataset(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    root.mkdir()
    for split in ("train", "val", "test"):
        image = root / "images" / split / f"{split}.jpg"
        label = root / "labels" / split / f"{split}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(split.encode())
        label.write_text("0 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
        (root / f"{split}.txt").write_text(f"./images/{split}/{split}.jpg\n", encoding="utf-8")
    (root / "data.yaml").write_text(
        yaml.safe_dump({"path": ".", "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": ["defect"]}),
        encoding="utf-8",
    )
    test_hash = hashlib.sha256((root / "test.txt").read_bytes()).hexdigest()
    (root / "dataset_manifest.json").write_text(
        json.dumps({"dataset_id": "release", "samples": {"train": 1, "val": 1, "test": 1}, "test_lock": {"locked_split_sha256": test_hash}}),
        encoding="utf-8",
    )
    return root


def test_clean_test_lock_passes(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "train.yaml").write_text("model: yolo.pt\ndata: data.yaml\n", encoding="utf-8")
    report = audit_test_lock(dataset, configs, tmp_path / "runs")
    assert report["valid"]


def test_test_used_for_threshold_selection_fails(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "train.yaml").write_text(
        "model: yolo.pt\ndata: data.yaml\nthreshold_split: test\n", encoding="utf-8"
    )
    report = audit_test_lock(dataset, configs, tmp_path / "runs")
    assert not report["valid"]
    assert any(item["reason"] == "test_used_for_training_or_selection" for item in report["violations"])


def test_modified_test_index_fails(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (dataset / "test.txt").write_text("./images/test/changed.jpg\n", encoding="utf-8")
    report = audit_test_lock(dataset, tmp_path / "configs", tmp_path / "runs")
    assert not report["valid"]
    assert any(item["reason"] == "test_list_hash_changed" for item in report["violations"])
