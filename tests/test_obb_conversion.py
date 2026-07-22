import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from blade_defect.data import DEFECT_CLASSES, check_obb_dataset
from scripts.convert_seg_to_obb import (
    PolygonConversionError,
    build_parser,
    convert_dataset,
    convert_label_text,
    convert_polygon_to_obb,
)


VALID_OBB = "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n"


def _obb_dataset_roots(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    return images, labels


def _write_obb_test_image(path: Path) -> None:
    Image.new("RGB", (12, 8), "white").save(path)


def test_normal_polygon_conversion_has_fixed_corner_order() -> None:
    polygon = np.array([[0.2, 0.2], [0.8, 0.2], [0.8, 0.6], [0.2, 0.6]])

    result = convert_polygon_to_obb(polygon)

    assert result.corners.shape == (4, 2)
    assert result.corners[0] == pytest.approx([0.2, 0.2])
    np.testing.assert_allclose(
        result.corners, [[0.2, 0.2], [0.8, 0.2], [0.8, 0.6], [0.2, 0.6]], atol=1e-6
    )


def test_out_of_bounds_polygon_is_clipped_and_reported() -> None:
    polygon = [[-0.2, 0.2], [1.2, 0.2], [1.2, 0.8], [-0.2, 0.8]]

    result = convert_polygon_to_obb(polygon)

    assert np.all((result.corners >= 0.0) & (result.corners <= 1.0))
    assert result.warnings[0][0] == "out_of_bounds"
    np.testing.assert_allclose(
        result.corners, [[0, 0.2], [1, 0.2], [1, 0.8], [0, 0.8]], atol=1e-6
    )


def test_minimum_rectangle_is_computed_in_image_pixel_space() -> None:
    pixel_corners = np.array([[50, 20], [150, 40], [142, 80], [42, 60]], dtype=float)
    normalized = pixel_corners / [200, 100]

    result = convert_polygon_to_obb(normalized, image_size=(200, 100))

    np.testing.assert_allclose(result.corners * [200, 100], pixel_corners, atol=1e-4)


def test_clipped_edge_corners_restart_from_leftmost_tied_corner() -> None:
    polygon = np.array([[0.35, 0], [0.8, 0.74], [0.78, 0.76], [0.33, 0]])

    result = convert_polygon_to_obb(polygon, image_size=(5472, 3648))

    assert result.corners[0, 1] == pytest.approx(0.0)
    top_edge = result.corners[np.isclose(result.corners[:, 1], 0.0)]
    assert result.corners[0, 0] == pytest.approx(top_edge[:, 0].min())


@pytest.mark.parametrize(
    "polygon,reason",
    [
        ([[0.1, 0.1], [0.2, 0.2]], "insufficient_points"),
        ([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], "area_too_small"),
        ([[0.1, 0.1], [0.1, 0.1], [0.2, 0.2]], "insufficient_points"),
    ],
)
def test_degenerate_polygon_is_rejected(polygon: list[list[float]], reason: str) -> None:
    with pytest.raises(PolygonConversionError) as error:
        convert_polygon_to_obb(polygon)
    assert error.value.reason == reason


def test_output_label_format() -> None:
    output, issues, count = convert_label_text("3 0.2 0.2 0.8 0.2 0.8 0.6 0.2 0.6\n")

    tokens = output.strip().split()
    assert count == 1
    assert issues == []
    assert tokens[0] == "3"
    assert len(tokens) == 9
    assert all(0.0 <= float(value) <= 1.0 for value in tokens[1:])


def test_dataset_conversion_preserves_structure_and_writes_reports(tmp_path: Path) -> None:
    source = tmp_path / "seg"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
        Image.new("RGB", (8, 8), "white").save(source / "images" / split / "sample.png")
    (source / "labels" / "train" / "sample.txt").write_text(
        "1 -0.1 0.2 0.8 0.2 0.8 0.7 -0.1 0.7\n"
        "2 0.1 0.1 0.1 0.1 0.2 0.2\n",
        encoding="utf-8",
    )
    (source / "labels" / "val" / "sample.txt").write_text("", encoding="utf-8")
    (source / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    output = tmp_path / "obb"
    reports = tmp_path / "reports"

    summary = convert_dataset(source, output, reports)

    assert (output / "images" / "train" / "sample.png").is_file()
    label_tokens = (output / "labels" / "train" / "sample.txt").read_text().split()
    assert len(label_tokens) == 9
    assert (output / "labels" / "val" / "sample.txt").read_text() == ""
    assert (output / "data.yaml").is_file()
    data_yaml = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data_yaml == {
        "path": ".",
        "train": "images/train",
        "val": "images/val",
        "names": DEFECT_CLASSES,
    }
    totals = summary["totals"] if isinstance(summary, dict) else getattr(summary, "totals")
    converted = totals["converted_objects"] if isinstance(totals, dict) else getattr(totals, "converted_objects")
    assert converted == 1
    assert json.loads((reports / "conversion_summary.json").read_text(encoding="utf-8")) == summary
    with (reports / "invalid_obb_labels.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["reason"] for row in rows} == {
        "out_of_bounds", "duplicate_points", "insufficient_points"
    }


def test_cli_requires_source_and_output() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_existing_output_requires_explicit_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for split in ("train", "val"):
        (source / "images" / split).mkdir(parents=True)
        (source / "labels" / split).mkdir(parents=True)
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError, match="--overwrite"):
        convert_dataset(source, output, tmp_path / "results")


def test_valid_obb_and_empty_negative_label(tmp_path: Path) -> None:
    images, labels = _obb_dataset_roots(tmp_path)
    _write_obb_test_image(images / "positive.png")
    _write_obb_test_image(images / "negative.png")
    (labels / "positive.txt").write_text(VALID_OBB, encoding="utf-8")
    (labels / "negative.txt").write_text("\n", encoding="utf-8")

    report = check_obb_dataset(images, labels)

    assert report.valid
    assert report.images == report.labels == 2
    assert report.valid_instances == 1
    assert report.negative_images == 1


@pytest.mark.parametrize(
    ("label", "error_type"),
    [
        ("0 0.1 0.1 0.8 0.1 0.8 0.8\n", "obb_field_count"),
        ("0 -0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n", "obb_coordinate_range"),
        ("0 0.1 0.1 0.8 0.1 0.8 0.8 0.8 0.8\n", "obb_duplicate_points"),
        ("0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4\n", "obb_area_too_small"),
    ],
)
def test_invalid_obb_rows_are_hard_errors(
    tmp_path: Path, label: str, error_type: str,
) -> None:
    images, labels = _obb_dataset_roots(tmp_path)
    _write_obb_test_image(images / "sample.png")
    (labels / "sample.txt").write_text(label, encoding="utf-8")

    report = check_obb_dataset(images, labels)

    assert not report.valid
    assert report.error_type_counts == {error_type: 1}


def test_missing_image_and_missing_label_are_reported(tmp_path: Path) -> None:
    images, labels = _obb_dataset_roots(tmp_path)
    _write_obb_test_image(images / "without_label.png")
    (labels / "without_image.txt").write_text(VALID_OBB, encoding="utf-8")

    report = check_obb_dataset(images, labels)

    assert not report.valid
    assert report.error_type_counts == {"missing_label": 1, "orphan_label": 1}
    assert report.missing_labels == ["without_label.png"]
    assert report.orphan_labels == ["without_image.txt"]


def test_corrupt_image_is_reported(tmp_path: Path) -> None:
    images, labels = _obb_dataset_roots(tmp_path)
    (images / "broken.png").write_bytes(b"not an image")
    (labels / "broken.txt").write_text(VALID_OBB, encoding="utf-8")

    report = check_obb_dataset(images, labels)

    assert not report.valid
    assert report.error_type_counts == {"corrupt_image": 1}
    assert report.corrupt_images == ["broken.png"]
