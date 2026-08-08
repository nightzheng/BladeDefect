from pathlib import Path

import pytest

from blade_defect.data.split_isolation import (
    assert_test_split_isolation,
    validate_test_split_isolation,
)
from blade_defect.data.validation import DatasetGateError


def _make_dataset(
    root: Path,
    train: list[str],
    val: list[str],
    test: list[str] | None,
) -> Path:
    dataset = root / "dataset"
    (dataset / "images").mkdir(parents=True)
    for split, names in (("train", train), ("val", val), ("test", test or [])):
        index_lines = []
        for name in names:
            image = dataset / "images" / split / f"{name}.jpg"
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(b"img")
            index_lines.append(image.as_posix())
        (dataset / f"{split}.txt").write_text("\n".join(index_lines), encoding="utf-8")
    lines = [
        f"path: {dataset.as_posix()}",
        "train: train.txt",
        "val: val.txt",
        "names: [defect]",
    ]
    if test is not None:
        lines.append("test: test.txt")
    data_yaml = root / "data.yaml"
    data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data_yaml


def test_isolated_splits_pass(tmp_path: Path) -> None:
    data_yaml = _make_dataset(
        tmp_path, train=["a", "b"], val=["c"], test=["d", "e"]
    )

    report = assert_test_split_isolation(data_yaml)

    assert report.isolated is True
    assert report.split_counts == {"train": 2, "val": 1, "test": 2}
    assert report.dataset_id == "dataset"


def test_same_path_in_train_and_test_fails(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    (dataset / "images" / "train").mkdir(parents=True)
    image = dataset / "images" / "train" / "leak.jpg"
    image.write_bytes(b"img")
    val_image = dataset / "images" / "val" / "clean.jpg"
    val_image.parent.mkdir(parents=True, exist_ok=True)
    val_image.write_bytes(b"img")
    line = image.as_posix()
    (dataset / "train.txt").write_text(line + "\n", encoding="utf-8")
    (dataset / "val.txt").write_text(val_image.as_posix() + "\n", encoding="utf-8")
    (dataset / "test.txt").write_text(line + "\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset.as_posix()}\ntrain: train.txt\nval: val.txt\n"
        "test: test.txt\nnames: [defect]\n",
        encoding="utf-8",
    )

    report = validate_test_split_isolation(data_yaml)

    assert report.isolated is False
    assert any(item.startswith("path:") for item in report.overlaps["train_vs_test"])
    with pytest.raises(DatasetGateError, match="split 泄漏"):
        assert_test_split_isolation(data_yaml)


def test_same_stem_different_directories_fails(tmp_path: Path) -> None:
    data_yaml = _make_dataset(
        tmp_path, train=["sample_001"], val=["sample_002"], test=["sample_001"]
    )

    report = validate_test_split_isolation(data_yaml)

    assert report.isolated is False
    assert report.overlaps["train_vs_test"] == ["stem:sample_001"]
    with pytest.raises(DatasetGateError, match="test 不得参与训练"):
        assert_test_split_isolation(data_yaml)


def test_train_val_overlap_detected_without_test_split(tmp_path: Path) -> None:
    data_yaml = _make_dataset(tmp_path, train=["a", "b"], val=["b"], test=None)

    report = validate_test_split_isolation(data_yaml)

    assert report.isolated is False
    assert report.split_counts["test"] == 0
    assert report.overlaps["train_vs_val"] == ["stem:b"]


def test_directory_style_splits_supported(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        image_dir = dataset / "images" / split
        image_dir.mkdir(parents=True)
        (image_dir / f"{split}_only.jpg").write_bytes(b"img")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset.as_posix()}\ntrain: images/train\nval: images/val\n"
        "test: images/test\nnames: [defect]\n",
        encoding="utf-8",
    )

    report = assert_test_split_isolation(data_yaml)

    assert report.isolated is True
    assert report.split_counts == {"train": 1, "val": 1, "test": 1}


def test_txt_index_in_subdirectory_resolves_relative_to_txt(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image = dataset / "images" / "train" / "leak.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"img")
    splits_dir = dataset / "splits"
    splits_dir.mkdir()
    (splits_dir / "train.txt").write_text("../images/train/leak.jpg\n", encoding="utf-8")
    (dataset / "val.txt").write_text("./images/train/val_only.jpg\n", encoding="utf-8")
    (dataset / "test.txt").write_text("./images/train/leak.jpg\n", encoding="utf-8")
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset.as_posix()}\ntrain: splits/train.txt\nval: val.txt\n"
        "test: test.txt\nnames: [defect]\n",
        encoding="utf-8",
    )

    report = validate_test_split_isolation(data_yaml)

    assert report.isolated is False
    assert any(item.startswith("path:") for item in report.overlaps["train_vs_test"])


def test_same_file_with_different_case_fails(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image = dataset / "images" / "train" / "Leak.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"img")
    (dataset / "train.txt").write_text(image.as_posix() + "\n", encoding="utf-8")
    (dataset / "val.txt").write_text(
        (dataset / "images" / "train" / "val_only.jpg").as_posix() + "\n", encoding="utf-8"
    )
    (dataset / "test.txt").write_text(
        (dataset / "images" / "train" / "leak.jpg").as_posix() + "\n", encoding="utf-8"
    )
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text(
        f"path: {dataset.as_posix()}\ntrain: train.txt\nval: val.txt\n"
        "test: test.txt\nnames: [defect]\n",
        encoding="utf-8",
    )

    report = validate_test_split_isolation(data_yaml)

    assert report.isolated is False
    assert report.overlaps["train_vs_test"]
