from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from blade_defect.data import DEFECT_CLASSES
from blade_defect.data.indexed_splits import load_indexed_splits, membership_hash
from scripts.remap_to_6class import convert_from_index

POLYGON = "0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3"
SPLITS = ("train", "val", "test")


def _write_grouped_source(root: Path) -> Path:
    """源数据集：v3 式分组 split，val/test 清单引用 images/train 下的文件。"""
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "a.jpg").write_bytes(b"a")
    (labels / "a.txt").write_text(f"0 {POLYGON}\n", encoding="utf-8")
    (images / "b.jpg").write_bytes(b"b")
    (labels / "b.txt").write_text(f"4 {POLYGON}\n", encoding="utf-8")
    (images / "c.jpg").write_bytes(b"c")
    (labels / "c.txt").write_text(f"14 {POLYGON}\n", encoding="utf-8")
    (images / "d.jpg").write_bytes(b"d")
    (labels / "d.txt").write_text("", encoding="utf-8")
    (root / "train.txt").write_text("./images/train/a.jpg\n./images/train/d.jpg\n", encoding="utf-8")
    (root / "val.txt").write_text("./images/train/b.jpg\n", encoding="utf-8")
    (root / "test.txt").write_text("./images/train/c.jpg\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump({
            "path": ".",
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
            "names": DEFECT_CLASSES,
        }),
        encoding="utf-8",
    )
    indexed = load_indexed_splits(data, SPLITS)
    (root / "dataset_manifest.json").write_text(
        json.dumps({
            "dataset_id": "source-grouped",
            "membership_sha256": membership_hash(indexed),
            "samples": {"train": 2, "val": 1, "test": 1},
            "instances": 3,
            "test_lock": {
                "policy": "test is readable only for owner-approved final evaluation",
                "automatic_unlock": False,
                "locked_split_sha256": "x" * 64,
            },
        }),
        encoding="utf-8",
    )
    return data


def test_index_mode_preserves_grouped_membership(tmp_path: Path) -> None:
    source_data = _write_grouped_source(tmp_path / "src")
    output = tmp_path / "out"

    summary = convert_from_index(source_data, output)

    assert (output / "images" / "val" / "b.jpg").is_file()
    assert (output / "images" / "test" / "c.jpg").is_file()
    assert sorted(p.name for p in (output / "images" / "train").iterdir()) == ["a.jpg", "d.jpg"]

    assert (output / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == f"0 {POLYGON}\n"
    assert (output / "labels" / "val" / "b.txt").read_text(encoding="utf-8") == f"1 {POLYGON}\n"
    assert (output / "labels" / "test" / "c.txt").read_text(encoding="utf-8") == f"5 {POLYGON}\n"
    assert (output / "labels" / "train" / "d.txt").read_text(encoding="utf-8") == ""

    data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"
    assert data["test"] == "images/test"
    assert len(data["names"]) == 6

    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    indexed = load_indexed_splits(source_data, SPLITS)
    assert manifest["label_level"] == "coarse"
    assert manifest["derived_from"] == "source-grouped"
    assert manifest["membership_sha256"] == membership_hash(indexed)
    assert manifest["source_membership_sha256"] == membership_hash(indexed)
    assert manifest["instances"] == 3
    assert manifest["samples"] == {"train": 2, "val": 1, "test": 1}
    assert manifest["class_counts"] == {"0": 1, "1": 1, "5": 1}
    assert manifest["storage"]["images_hardlinked"] == 4
    assert manifest["storage"]["images_copied"] == 0
    assert manifest["test_lock"]["automatic_unlock"] is False
    assert "locked_split_sha256" not in manifest["test_lock"]
    assert manifest["test_lock"]["inherited_from"] == "source-grouped"

    assert summary["instances"] == 3
    assert all(
        item["valid"] and item["membership_ok"] for item in summary["validation"].values()
    )


def test_index_mode_rejects_within_split_filename_collision(tmp_path: Path) -> None:
    root = tmp_path / "src"
    (root / "images" / "train").mkdir(parents=True)
    (root / "images" / "val").mkdir(parents=True)
    (root / "train.txt").write_text("./images/train/x.jpg\n./images/val/x.jpg\n", encoding="utf-8")
    (root / "val.txt").write_text("./images/train/y.jpg\n", encoding="utf-8")
    (root / "test.txt").write_text("./images/train/z.jpg\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text(
        yaml.safe_dump({
            "path": ".",
            "train": "train.txt",
            "val": "val.txt",
            "test": "test.txt",
            "names": DEFECT_CLASSES,
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="文件名冲突"):
        convert_from_index(data, tmp_path / "out")


def test_index_mode_detects_membership_drift(tmp_path: Path) -> None:
    source_data = _write_grouped_source(tmp_path / "src")
    manifest_path = source_data.parent / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["membership_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="成员关系"):
        convert_from_index(source_data, tmp_path / "out")


def test_index_mode_detects_instance_drift(tmp_path: Path) -> None:
    source_data = _write_grouped_source(tmp_path / "src")
    manifest_path = source_data.parent / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["instances"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="实例数"):
        convert_from_index(source_data, tmp_path / "out")


def test_index_mode_fails_strict_validation_on_invalid_polygon(tmp_path: Path) -> None:
    source_data = _write_grouped_source(tmp_path / "src")
    (source_data.parent / "labels" / "train" / "a.txt").write_text(
        "0 0.1 0.1 2.0 0.1 2.0 0.3 0.1 0.3\n", encoding="utf-8"
    )

    output = tmp_path / "out"
    with pytest.raises(RuntimeError, match="strict 校验失败"):
        convert_from_index(source_data, output)

    assert not output.exists()


def test_index_mode_requires_empty_output(tmp_path: Path) -> None:
    source_data = _write_grouped_source(tmp_path / "src")
    output = tmp_path / "out"
    output.mkdir()
    (output / "occupied.txt").write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError, match="输出目录不为空"):
        convert_from_index(source_data, output)
