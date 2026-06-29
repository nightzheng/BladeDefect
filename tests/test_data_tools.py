from pathlib import Path

from PIL import Image

from blade_defect.data import check_dataset, clean_dataset, split_dataset
from blade_defect.data.label_check import validate_seg_line


def test_valid_segmentation_line() -> None:
    assert validate_seg_line("0 0.1 0.1 0.5 0.1 0.5 0.5", num_classes=2) is None


def test_invalid_coordinate() -> None:
    assert "[0, 1]" in validate_seg_line("0 0.1 0.1 1.2 0.1 0.5 0.5")


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
