from pathlib import Path

import pytest
import yaml
from PIL import Image

from blade_defect.data import (
    DEFECT_CLASSES,
    DEFECT_GROUPS,
    check_dataset,
    clean_dataset,
    get_class_name,
    get_group_classes,
    get_group_name,
    split_dataset,
)
from blade_defect.data.label_check import validate_seg_line
from scripts.create_small_dataset import create_small_dataset


def test_defect_class_ids_are_contiguous() -> None:
    assert list(DEFECT_CLASSES) == list(range(15))


def test_every_defect_class_has_a_group() -> None:
    grouped_ids = [class_id for class_ids in DEFECT_GROUPS.values() for class_id in class_ids]
    assert sorted(grouped_ids) == list(DEFECT_CLASSES)
    assert all(get_group_name(class_id) in DEFECT_GROUPS for class_id in DEFECT_CLASSES)
    assert all(get_group_classes(group_name) == class_ids for group_name, class_ids in DEFECT_GROUPS.items())


def test_data_yaml_names_match_defect_classes() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "data.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["names"] == DEFECT_CLASSES


@pytest.mark.parametrize("lookup", [get_class_name, get_group_name])
@pytest.mark.parametrize("class_id", [-1, 15, True])
def test_invalid_defect_class_id_raises(lookup, class_id: int) -> None:
    with pytest.raises(ValueError, match="有效范围为 0-14"):
        lookup(class_id)


def test_create_small_dataset_copies_only_pairs_and_writes_config(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)

    for filename in ("one.jpg", "two.JPG", "three.jpeg", "four.png"):
        (source / "images" / "train" / filename).write_bytes(filename.encode())
        (source / "labels" / "train" / Path(filename).with_suffix(".txt")).write_text(
            "0 0.1 0.1 0.5 0.1 0.5 0.5\n",
            encoding="utf-8",
        )
    (source / "images" / "val" / "paired.png").write_bytes(b"paired")
    (source / "labels" / "val" / "paired.txt").write_text("10 0.1 0.1 0.5 0.1 0.5 0.5\n", encoding="utf-8")
    (source / "images" / "val" / "unmatched.jpg").write_bytes(b"unmatched")

    first_output = tmp_path / "small-first"
    second_output = tmp_path / "small-second"
    first_stats = create_small_dataset(source, first_output, train_count=2, val_count=3, seed=7)
    second_stats = create_small_dataset(source, second_output, train_count=2, val_count=3, seed=7)

    first_train_images = sorted(path.name for path in (first_output / "images" / "train").iterdir())
    second_train_images = sorted(path.name for path in (second_output / "images" / "train").iterdir())
    assert first_train_images == second_train_images
    assert len(first_train_images) == 2
    assert first_stats == second_stats
    assert first_stats == {
        "train": {"requested": 2, "available_pairs": 4, "copied": 2, "unmatched_images": 0},
        "val": {"requested": 3, "available_pairs": 1, "copied": 1, "unmatched_images": 1},
    }

    for image_path in (first_output / "images").rglob("*"):
        if image_path.is_file():
            relative = image_path.relative_to(first_output / "images")
            assert (first_output / "labels" / relative.with_suffix(".txt")).is_file()
    data_config = yaml.safe_load((first_output / "data.yaml").read_text(encoding="utf-8"))
    assert data_config["train"] == "images/train"
    assert data_config["val"] == "images/val"
    assert data_config["names"] == DEFECT_CLASSES


def test_valid_segmentation_line() -> None:
    assert validate_seg_line("0 0.1 0.1 0.5 0.1 0.5 0.5", num_classes=2) is None


def test_invalid_coordinate() -> None:
    message = validate_seg_line("0 0.1 0.1 1.2 0.1 0.5 0.5")
    assert message is not None
    assert "[0, 1]" in message


def test_invalid_class() -> None:
    message = validate_seg_line("2 0.1 0.1 0.5 0.1 0.5 0.5", num_classes=2)
    assert message is not None
    assert "class_id=2" in message


def test_uppercase_extensions_and_case_insensitive_pairing(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    (images / "Nested").mkdir(parents=True)
    (labels / "nested").mkdir(parents=True)
    (images / "Nested" / "Blade.JPG").write_bytes(b"image")
    (labels / "nested" / "blade.TXT").write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5\n",
        encoding="utf-8",
    )

    report = check_dataset(images, labels, num_classes=1)
    assert report.valid
    assert report.images == 1
    assert report.labels == 1

    output = tmp_path / "output"
    counts = split_dataset(images, labels, output, ratios=(1.0, 0.0, 0.0))
    assert counts == {"train": 1, "val": 0, "test": 0, "unmatched_images": 0}
    assert (output / "images" / "train" / "Nested" / "Blade.JPG").exists()
    assert (output / "labels" / "train" / "Nested" / "Blade.txt").exists()


def test_clean_dataset_reports_duplicates_and_empty_labels(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()

    image = Image.new("RGB", (2, 2), color="white")
    image.save(images / "first.png")
    image.save(images / "second.png")
    (labels / "first.txt").write_text("", encoding="utf-8")

    report = clean_dataset(images, labels)

    assert report.scanned == 2
    assert report.corrupt_images == []
    assert len(report.duplicate_images) == 1
    assert report.empty_labels == [str(labels / "first.txt")]
