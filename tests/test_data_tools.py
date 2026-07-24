from pathlib import Path

import pytest
import yaml
from PIL import Image

from blade_defect.data import (
    DEFECT_CLASSES,
    DEFECT_GROUPS,
    check_dataset,
    clean_dataset,
    clamp01,
    get_class_name,
    get_group_classes,
    get_group_name,
    load_dataset_filter,
    split_dataset,
)
from blade_defect.data.label_check import validate_seg_line
from scripts.create_small_dataset import create_small_dataset
from scripts import create_small_dataset as small_dataset_module


def _write_test_image(path: Path) -> None:
    """写入一张 Pillow 和 OpenCV 都能解码的小图片。"""
    Image.new("RGB", (4, 4), color="white").save(path)


def _write_filter_config(path: Path, images: list[dict[str, str]]) -> None:
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "issue_types": ["missing_label", "uncertain_annotation"],
                "actions": ["exclude", "review", "keep_negative"],
                "images": images,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )


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


def test_project_dataset_filter_contains_confirmed_decisions() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "dataset_filter.yaml"
    dataset_filter = load_dataset_filter(config_path)

    decisions = list(dataset_filter.decisions.values())
    assert len(decisions) == 24
    assert sum(decision.action == "exclude" for decision in decisions) == 19
    assert sum(decision.action == "keep_negative" for decision in decisions) == 5
    assert sum(decision.issue == "corrupted_file" for decision in decisions) == 10
    assert all(decision.status == "confirmed" for decision in decisions)


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
        _write_test_image(source / "images" / "train" / filename)
        (source / "labels" / "train" / Path(filename).with_suffix(".txt")).write_text(
            "0 0.1 0.1 0.5 0.1 0.5 0.5\n",
            encoding="utf-8",
        )
    _write_test_image(source / "images" / "val" / "paired.png")
    (source / "labels" / "val" / "paired.txt").write_text("10 0.1 0.1 0.5 0.1 0.5 0.5\n", encoding="utf-8")
    (source / "images" / "val" / "unmatched.jpg").write_bytes(b"unmatched")
    (source / "labels" / "val" / "orphan.txt").write_text(
        "0 0.1 0.1 0.5 0.1 0.5 0.5\n", encoding="utf-8"
    )

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
        "train": {"requested": 2, "available_pairs": 4, "copied": 2, "unmatched_images": 0,
                  "missing_images": 0, "missing_labels": 0, "fixed_points": 0,
                  "fixed_files": 0, "removed_files": 0, "corrupt_images": 0,
                  "decode_removed": 0, "decode_checked": 2},
        "val": {"requested": 3, "available_pairs": 1, "copied": 1, "unmatched_images": 1,
                "missing_images": 1, "missing_labels": 1, "fixed_points": 0,
                "fixed_files": 0, "removed_files": 0, "corrupt_images": 0,
                "decode_removed": 0, "decode_checked": 1},
    }

    for image_path in (first_output / "images").rglob("*"):
        if image_path.is_file():
            relative = image_path.relative_to(first_output / "images")
            assert (first_output / "labels" / relative.with_suffix(".txt")).is_file()
    data_config = yaml.safe_load((first_output / "data.yaml").read_text(encoding="utf-8"))
    assert data_config["train"] == "images/train"
    assert data_config["val"] == "images/val"
    assert data_config["names"] == DEFECT_CLASSES


def test_create_small_dataset_applies_filter_and_creates_empty_negative_label(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    _write_test_image(source / "images" / "train" / "paired.jpg")
    (source / "labels" / "train" / "paired.txt").write_text(
        "0 0 0 1 0 1 1\n", encoding="utf-8"
    )
    _write_test_image(source / "images" / "train" / "negative.jpg")
    _write_test_image(source / "images" / "train" / "excluded.jpg")
    filter_config = tmp_path / "dataset_filter.yaml"
    _write_filter_config(
        filter_config,
        [
            {
                "filename": "negative.jpg",
                "action": "keep_negative",
                "reason": "确认无损伤",
                "status": "confirmed",
            },
            {
                "filename": "excluded.jpg",
                "action": "exclude",
                "issue": "missing_label",
                "reason": "有缺陷但没有坐标",
                "status": "confirmed",
            },
        ],
    )

    output = tmp_path / "output"
    stats = create_small_dataset(
        source, output, train_count=2, val_count=0, filter_config=filter_config
    )

    assert stats["train"]["available_pairs"] == 2
    assert stats["train"]["excluded_images"] == 1
    assert stats["train"]["negative_images"] == 1
    assert (output / "labels" / "train" / "negative.txt").read_text(encoding="utf-8") == ""
    assert not (output / "images" / "train" / "excluded.jpg").exists()


def test_create_small_dataset_cleans_partial_copy_if_source_disappears(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    image = source / "images" / "train" / "sample.jpg"
    label = source / "labels" / "train" / "sample.txt"
    _write_test_image(image)
    label.write_text("0 0 0 1 0 1 1\n", encoding="utf-8")
    original_copy = small_dataset_module.shutil.copy2

    def disappearing_copy(source_path: Path, target_path: Path) -> Path | str:
        if Path(source_path) == label:
            label.unlink()
        return original_copy(source_path, target_path)

    monkeypatch.setattr(small_dataset_module.shutil, "copy2", disappearing_copy)
    output = tmp_path / "output"
    stats = create_small_dataset(source, output, train_count=1, val_count=0)

    assert stats["train"]["copied"] == 0
    assert stats["train"]["missing_labels"] == 1
    assert not any((output / "images" / "train").iterdir())
    assert not any((output / "labels" / "train").iterdir())


def test_create_small_dataset_repairs_soft_but_stops_for_hard_review(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    samples = {
        "soft": "0 -0.005 0.1 0.5 0.1 0.5 1.006\n",
        "hard": "0 0.1 0.1 1.2 0.1 0.5 0.5\n",
        "bbox": "0 0.5 0.5 0.2 0.2\n",
    }
    for name, annotation in samples.items():
        _write_test_image(source / "images" / "train" / f"{name}.jpg")
        (source / "labels" / "train" / f"{name}.txt").write_text(annotation, encoding="utf-8")

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="程序不会自动删除"):
        create_small_dataset(source, output, train_count=3, val_count=0)

    assert (output / "labels" / "train" / "soft.txt").read_text(encoding="utf-8") == (
        "0 0 0.1 0.5 0.1 0.5 1\n"
    )
    assert (output / "labels" / "train" / "hard.txt").exists()
    assert (output / "images" / "train" / "hard.jpg").exists()
    assert (source / "labels" / "hard.txt").read_text(encoding="utf-8") == samples["hard"]


def test_create_small_dataset_rejects_images_opencv_cannot_decode(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    bad_image = source / "images" / "train" / "bad.JPG"
    bad_image.write_bytes(b"not a jpeg")
    (source / "labels" / "train" / "bad.txt").write_text(
        "0 0 0 1 0 1 1\n", encoding="utf-8"
    )

    output = tmp_path / "output"
    stats = create_small_dataset(source, output, train_count=1, val_count=0)

    assert stats["train"]["available_pairs"] == 1
    assert stats["train"]["copied"] == 0
    assert stats["train"]["corrupt_images"] == 1
    assert stats["train"]["decode_checked"] == 1
    assert not any((output / "images" / "train").iterdir())


def test_create_small_dataset_decodes_only_until_target_is_filled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    for index in range(20):
        image = source / "images" / "train" / f"{index}.jpg"
        _write_test_image(image)
        (source / "labels" / "train" / f"{index}.txt").write_text(
            "0 0 0 1 0 1 1\n", encoding="utf-8"
        )
    decode_calls = 0
    original_check = small_dataset_module._is_decodable_image

    def count_decode(path: Path) -> bool:
        nonlocal decode_calls
        decode_calls += 1
        return original_check(path)

    monkeypatch.setattr(small_dataset_module, "_is_decodable_image", count_decode)
    stats = create_small_dataset(source, tmp_path / "output", train_count=3, val_count=0)

    assert stats["train"]["copied"] == 3
    assert stats["train"]["decode_checked"] == 3
    assert decode_calls == 3


def test_valid_segmentation_line() -> None:
    assert validate_seg_line("0 0.1 0.1 0.5 0.1 0.5 0.5", num_classes=2) is None
    assert validate_seg_line("14 0 0 1 0 1 1 0.5 0.75 0 1", num_classes=15) is None


def test_invalid_coordinate() -> None:
    message = validate_seg_line("0 0.1 0.1 1.2 0.1 0.5 0.5")
    assert message is not None
    assert "[0, 1]" in message


def test_invalid_class() -> None:
    message = validate_seg_line("2 0.1 0.1 0.5 0.1 0.5 0.5", num_classes=2)
    assert message is not None
    assert "class_id=2" in message


def test_label_report_distinguishes_bbox_polygon_and_empty_labels(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    rows = {
        "valid": "14 0 0 1 0 1 1 0 1\n",
        "bbox": "0 0.5 0.5 0.2 0.2\n",
        "odd": "0 0.1 0.1 0.5 0.1 0.5\n",
        "range": "0 0.1 0.1 1.01 0.2 0.5 0.5\n",
        "empty": "\n",
    }
    for name, row in rows.items():
        (images / f"{name}.jpg").write_bytes(b"image")
        (labels / f"{name}.txt").write_text(row, encoding="utf-8")

    report = check_dataset(images, labels, num_classes=15, polygon_mode="strict")

    assert report.valid_objects == 1
    assert report.empty_labels == ["empty.txt"]
    assert report.error_type_counts == {
        "bbox_format": 1,
        "polygon_coordinate_count": 1,
        "polygon_coordinate_range": 1,
    }
    assert "第 2 个点" in next(issue.message for issue in report.issues if issue.error_type == "polygon_coordinate_range")
    payload = report.to_dict()
    assert payload["error_type_counts"] == report.error_type_counts


def test_empty_label_is_recorded_without_becoming_an_error(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "negative.jpg").write_bytes(b"image")
    (labels / "negative.txt").write_text("\n", encoding="utf-8")

    report = check_dataset(images, labels, num_classes=15)

    assert report.empty_labels == ["negative.txt"]
    assert report.valid
    assert report.error_type_counts == {}


def test_fix_float_clips_soft_errors_without_changing_class_id(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "soft.jpg").write_bytes(b"image")
    label = labels / "soft.txt"
    label.write_text("14 -0.005 0.2 0.4 1.006 1 0.8\n", encoding="utf-8")

    report = check_dataset(images, labels, num_classes=15, fix_float=True)

    assert label.read_text(encoding="utf-8") == "14 0 0.2 0.4 1 1 0.8\n"
    assert report.fixed_points == 2
    assert report.fixed_files == 1
    assert report.max_offset == pytest.approx(0.006)
    assert report.fixes[0].status == "FIXED_FLOAT_ERROR"
    assert report.issues == []
    assert report.valid


def test_fix_float_dry_run_reports_but_does_not_write(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "soft.jpg").write_bytes(b"image")
    label = labels / "soft.txt"
    original = "0 0.1 0.1 1.001 0.2 0.5 0.5\n"
    label.write_text(original, encoding="utf-8")

    report = check_dataset(images, labels, fix_float=True, dry_run=True)

    assert label.read_text(encoding="utf-8") == original
    assert report.fixed_points == 1
    assert report.fixed_files == 1
    assert report.dry_run
    assert report.fixes[0].status == "FIXED_FLOAT_ERROR"


def test_auto_fix_removes_image_label_pair_with_hard_coordinate_error(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    (images / "hard.jpg").write_bytes(b"image")
    label = labels / "hard.txt"
    label.write_text("0 0.1 0.1 1.02 0.2 0.5 0.5\n", encoding="utf-8")

    report = check_dataset(images, labels, fix_float=True)

    assert report.fixed_points == 0
    assert report.fixed_files == 0
    assert report.removed_files == 1
    assert report.issues[0].status == "HARD_ERROR"
    assert not label.exists()
    assert not (images / "hard.jpg").exists()
    assert not report.valid


@pytest.mark.parametrize(("value", "expected"), [(-1.0, 0.0), (0.4, 0.4), (2.0, 1.0)])
def test_clamp01(value: float, expected: float) -> None:
    assert clamp01(value) == expected


def test_soft_mode_keeps_hard_error_sample(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    image = images / "hard.jpg"
    label = labels / "hard.txt"
    image.write_bytes(b"image")
    label.write_text("0 0.1 0.1 1.02 0.2 0.5 0.5\n", encoding="utf-8")

    report = check_dataset(images, labels, polygon_mode="soft")

    assert report.removed_files == 0
    assert image.exists() and label.exists()
    assert report.issues[0].error_type == "polygon_coordinate_range"


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
    assert counts == {
        "train": 1,
        "val": 0,
        "test": 0,
        "unmatched_images": 0,
        "fixed_points": 0,
        "fixed_label_files": 0,
        "removed_invalid_samples": 0,
    }
    assert (output / "images" / "train" / "Nested" / "Blade.JPG").exists()
    assert (output / "labels" / "train" / "Nested" / "Blade.txt").exists()


def test_split_dataset_applies_filter_actions(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for filename in ("paired.jpg", "negative.jpg", "excluded.jpg", "review.jpg"):
        _write_test_image(images / filename)
    (labels / "paired.txt").write_text("0 0 0 1 0 1 1\n", encoding="utf-8")
    filter_config = tmp_path / "dataset_filter.yaml"
    _write_filter_config(
        filter_config,
        [
            {
                "filename": "negative.jpg",
                "action": "keep_negative",
                "reason": "确认无损伤",
                "status": "confirmed",
            },
            {
                "filename": "excluded.jpg",
                "action": "exclude",
                "issue": "missing_label",
                "reason": "有缺陷但没有坐标",
                "status": "confirmed",
            },
            {
                "filename": "review.jpg",
                "action": "review",
                "issue": "uncertain_annotation",
                "reason": "等待复核",
                "status": "pending",
            },
        ],
    )

    output = tmp_path / "output"
    counts = split_dataset(
        images,
        labels,
        output,
        ratios=(1.0, 0.0, 0.0),
        filter_config=filter_config,
    )

    assert counts == {
        "train": 2,
        "val": 0,
        "test": 0,
        "unmatched_images": 0,
        "fixed_points": 0,
        "fixed_label_files": 0,
        "removed_invalid_samples": 0,
        "excluded_images": 1,
        "review_images": 1,
        "negative_images": 1,
    }
    assert (output / "images" / "train" / "paired.jpg").exists()
    assert (output / "labels" / "train" / "negative.txt").read_text(encoding="utf-8") == ""
    assert not (output / "images" / "train" / "excluded.jpg").exists()
    assert not (output / "images" / "train" / "review.jpg").exists()


def test_split_dataset_repairs_only_new_labels_and_stops_for_hard_review(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    soft_source = "0 -0.005 0.1 0.5 0.1 0.5 1.006\n"
    hard_source = "0 0.1 0.1 1.2 0.1 0.5 0.5\n"
    for name, annotation in (("soft", soft_source), ("hard", hard_source)):
        _write_test_image(images / f"{name}.jpg")
        (labels / f"{name}.txt").write_text(annotation, encoding="utf-8")

    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="程序不会自动删除"):
        split_dataset(images, labels, output, ratios=(1.0, 0.0, 0.0))

    # 学长确认的规则：原始标签不变，修正只写入新数据集。
    assert (labels / "soft.txt").read_text(encoding="utf-8") == soft_source
    assert (labels / "hard.txt").read_text(encoding="utf-8") == hard_source
    assert (output / "labels" / "train" / "soft.txt").read_text(encoding="utf-8") == (
        "0 0 0.1 0.5 0.1 0.5 1\n"
    )
    assert (output / "labels" / "train" / "hard.txt").exists()
    assert (output / "images" / "train" / "hard.jpg").exists()


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
