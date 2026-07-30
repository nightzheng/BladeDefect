"""txt 索引数据集门禁测试：索引解析、缺失图片、重复索引、空标签负样本与路径可移植性。"""
from __future__ import annotations

from pathlib import Path

import pytest

from blade_defect.data.validation import (
    DatasetGateError,
    validate_dataset_gate,
)

VALID_LINE = "0 0.1 0.1 0.2 0.1 0.2 0.2\n"


def _write_sample(dataset: Path, split: str, name: str, label: str | None = VALID_LINE) -> Path:
    image = dataset / "images" / split / f"{name}.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8\xff")
    if label is not None:
        label_path = dataset / "labels" / split / f"{name}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(label, encoding="utf-8")
    return image


def _write_data_yaml(tmp_path: Path, dataset: Path, *, absolute_root: bool = True) -> Path:
    root = dataset.as_posix() if absolute_root else "dataset"
    data = tmp_path / "data.yaml"
    data.write_text(
        f"path: {root}\ntrain: train.txt\nval: val.txt\nnames: [defect]\n",
        encoding="utf-8",
    )
    return data


def _build_dataset(tmp_path: Path) -> Path:
    dataset = tmp_path / "dataset"
    for split in ("train", "val"):
        _write_sample(dataset, split, f"{split}_0")
        _write_sample(dataset, split, f"{split}_1")
    return dataset


def _write_index(dataset: Path, split: str, lines: list[str]) -> None:
    dataset.joinpath(f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _default_index(dataset: Path) -> None:
    for split in ("train", "val"):
        _write_index(
            dataset,
            split,
            [f"./images/{split}/{split}_0.jpg", f"./images/{split}/{split}_1.jpg"],
        )


def test_txt_index_with_relative_dot_slash_paths_passes(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    _default_index(dataset)
    report = validate_dataset_gate(_write_data_yaml(tmp_path, dataset))
    assert report.valid
    assert report.splits["train"].source == "txt_index"
    assert report.splits["train"].images == 2
    assert report.splits["train"].instances == 2
    assert report.splits["train"].class_counts == {0: 2}


def test_txt_index_with_absolute_image_paths_passes(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    for split in ("train", "val"):
        _write_index(
            dataset,
            split,
            [
                (dataset / "images" / split / f"{split}_0.jpg").as_posix(),
                (dataset / "images" / split / f"{split}_1.jpg").as_posix(),
            ],
        )
    report = validate_dataset_gate(_write_data_yaml(tmp_path, dataset))
    assert report.valid


def test_relative_data_yaml_root_is_portable(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    _default_index(dataset)
    data = _write_data_yaml(tmp_path, dataset, absolute_root=False)
    report = validate_dataset_gate(data)
    assert report.valid


def test_missing_image_in_index_fails_with_context(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    _write_index(dataset, "train", ["./images/train/train_0.jpg", "./images/train/ghost.jpg"])
    _write_index(dataset, "val", ["./images/val/val_0.jpg", "./images/val/val_1.jpg"])
    data = _write_data_yaml(tmp_path, dataset)
    with pytest.raises(DatasetGateError) as excinfo:
        validate_dataset_gate(data)
    message = str(excinfo.value)
    assert "missing_images=1" in message
    assert "ghost.jpg" in message
    assert str(dataset / "train.txt") in message
    assert f"dataset_id={dataset.name}" in message
    assert "data_yaml=" in message


def test_duplicate_index_entries_fail(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    _write_index(
        dataset,
        "train",
        ["./images/train/train_0.jpg", "./images/train/train_0.jpg", "./images/train/train_1.jpg"],
    )
    _write_index(dataset, "val", ["./images/val/val_0.jpg", "./images/val/val_1.jpg"])
    with pytest.raises(DatasetGateError, match="duplicate_index=1"):
        validate_dataset_gate(_write_data_yaml(tmp_path, dataset))


def test_empty_label_counts_as_negative_sample_and_passes(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    _write_sample(dataset, "train", "negative", label="")
    _write_index(
        dataset,
        "train",
        [
            "./images/train/train_0.jpg",
            "./images/train/train_1.jpg",
            "./images/train/negative.jpg",
        ],
    )
    _write_index(dataset, "val", ["./images/val/val_0.jpg", "./images/val/val_1.jpg"])
    report = validate_dataset_gate(_write_data_yaml(tmp_path, dataset))
    assert report.valid
    assert report.splits["train"].negative_samples == 1
    assert report.splits["train"].instances == 2


def test_missing_label_fails(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    image = dataset / "images" / "train" / "train_1.jpg"
    # 删除 train_1 的标签使其成为缺失标签样本。
    (dataset / "labels" / "train" / "train_1.txt").unlink()
    assert image.is_file()
    _default_index(dataset)
    with pytest.raises(DatasetGateError, match="missing_labels=1"):
        validate_dataset_gate(_write_data_yaml(tmp_path, dataset))


def test_orphan_label_fails(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    _write_sample(dataset, "train", "orphan")
    # 索引中不包含 orphan.jpg，标签成为孤立标签。
    _write_index(dataset, "train", ["./images/train/train_0.jpg", "./images/train/train_1.jpg"])
    _write_index(dataset, "val", ["./images/val/val_0.jpg", "./images/val/val_1.jpg"])
    with pytest.raises(DatasetGateError, match="orphan_labels=1"):
        validate_dataset_gate(_write_data_yaml(tmp_path, dataset))


def test_out_of_range_class_fails(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    _write_sample(dataset, "train", "bad", label="3 0.1 0.1 0.2 0.1 0.2 0.2\n")
    _write_index(
        dataset,
        "train",
        ["./images/train/train_0.jpg", "./images/train/train_1.jpg", "./images/train/bad.jpg"],
    )
    _write_index(dataset, "val", ["./images/val/val_0.jpg", "./images/val/val_1.jpg"])
    with pytest.raises(DatasetGateError, match="hard_error=1"):
        validate_dataset_gate(_write_data_yaml(tmp_path, dataset))


def test_directory_dataset_still_supported(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    data = tmp_path / "data.yaml"
    data.write_text(
        f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\nnames: [defect]\n",
        encoding="utf-8",
    )
    report = validate_dataset_gate(data)
    assert report.valid
    assert report.splits["val"].source == "directory"
