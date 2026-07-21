import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.convert_seg_to_obb import (
    PolygonConversionError,
    convert_dataset,
    convert_label_text,
    convert_polygon_to_obb,
)


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
    totals = summary["totals"] if isinstance(summary, dict) else getattr(summary, "totals")
    converted = totals["converted_objects"] if isinstance(totals, dict) else getattr(totals, "converted_objects")
    assert converted == 1
    assert json.loads((reports / "conversion_summary.json").read_text(encoding="utf-8")) == summary
    with (reports / "invalid_obb_labels.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["reason"] for row in rows} == {
        "out_of_bounds", "duplicate_points", "insufficient_points"
    }
