from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.analyze_dataset import (
    analyze_dataset,
    load_filter_decisions,
    parse_label,
    polygon_geometry,
)


class DatasetAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_filter_config_unifies_effective_image_and_label_counts(self) -> None:
        images = self.root / "images" / "train"
        labels = self.root / "labels" / "train"
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        for filename in ("paired.jpg", "negative.jpg", "excluded.jpg"):
            Image.new("RGB", (100, 200), "white").save(images / filename)
        (labels / "paired.txt").write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n", encoding="utf-8")
        data_config = self.root / "data.yaml"
        data_config.write_text("names:\n  0: test-class\n", encoding="utf-8")
        filter_config = self.root / "dataset_filter.yaml"
        filter_config.write_text(
            """version: 1
issue_types:
  - missing_label
actions:
  - exclude
  - review
  - keep_negative
images:
  - filename: negative.jpg
    action: keep_negative
    reason: confirmed negative
    status: confirmed
  - filename: excluded.jpg
    action: exclude
    issue: missing_label
    reason: missing coordinates
    status: confirmed
""",
            encoding="utf-8",
        )

        decisions = load_filter_decisions(filter_config)
        self.assertEqual(set(decisions), {"negative.jpg", "excluded.jpg"})
        output = self.root / "results"
        report = analyze_dataset(
            self.root / "images",
            self.root / "labels",
            data_config,
            output,
            filter_config=filter_config,
            workers=2,
            charts=False,
        )

        totals = report["totals"]
        self.assertEqual(totals["raw_images"], 3)
        self.assertEqual(totals["raw_labels"], 1)
        self.assertEqual(totals["images"], 2)
        self.assertEqual(totals["labels"], 2)
        self.assertEqual(totals["paired"], 2)
        self.assertEqual(totals["missing_labels"], 0)
        self.assertEqual(totals["virtual_empty_labels"], 1)
        self.assertEqual(totals["excluded_by_filter"], 1)
        self.assertEqual(totals["instances"], 1)
        self.assertEqual(totals["geometry_instances"], 1)
        with (output / "object_statistics.csv").open(encoding="utf-8-sig", newline="") as handle:
            object_rows = list(csv.DictReader(handle))
        self.assertEqual(len(object_rows), 1)
        self.assertEqual(object_rows[0]["size_category"], "small")

    def test_known_square_geometry_is_calculated_correctly(self) -> None:
        coordinates = (0.1, 0.1, 0.3, 0.1, 0.3, 0.3, 0.1, 0.3)
        geometry = polygon_geometry(coordinates, width=100, height=200)
        self.assertAlmostEqual(float(geometry["bbox_width_px"]), 20.0)
        self.assertAlmostEqual(float(geometry["bbox_height_px"]), 40.0)
        self.assertAlmostEqual(float(geometry["bbox_area_px"]), 800.0)
        self.assertAlmostEqual(float(geometry["mask_area_px"]), 800.0)
        self.assertAlmostEqual(float(geometry["relative_bbox_area"]), 0.04)
        self.assertAlmostEqual(float(geometry["relative_mask_area"]), 0.04)
        self.assertEqual(geometry["size_category"], "small")

    def test_soft_coordinate_overflow_is_clipped_only_in_memory(self) -> None:
        label = self.root / "soft.txt"
        original = "0 -0.005 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n"
        label.write_text(original, encoding="utf-8")
        result = parse_label(label, {0: "test"})

        self.assertEqual(label.read_text(encoding="utf-8"), original)
        self.assertEqual(len(result.class_ids), 1)
        self.assertEqual(len(result.polygons), 1)
        self.assertEqual(result.polygons[0].geometry_status, "soft_clipped")
        self.assertEqual(result.polygons[0].coordinates[0], 0.0)
        self.assertEqual(result.errors[0]["severity"], "soft")
        self.assertAlmostEqual(float(result.errors[0]["max_offset"]), 0.005)
        self.assertEqual(len(result.coordinate_issues), 1)
        issue = result.coordinate_issues[0]
        self.assertEqual(issue["class_name"], "test")
        self.assertEqual(issue["point_index"], 1)
        self.assertEqual(issue["axis"], "x")
        self.assertEqual(issue["coordinate_position"], "point_1_x")
        self.assertAlmostEqual(float(issue["original_value"]), -0.005)
        self.assertAlmostEqual(float(issue["overflow_amount"]), 0.005)
        self.assertEqual(issue["line_severity"], "soft")
        self.assertEqual(issue["reset_value"], 0.0)
        self.assertEqual(issue["action"], "reset_to_boundary")

    def test_hard_coordinate_overflow_is_not_used_for_geometry(self) -> None:
        label = self.root / "hard.txt"
        label.write_text("0 0.1 0.1 1.2 0.1 0.3 0.3\n", encoding="utf-8")
        result = parse_label(label, {0: "test"})

        self.assertEqual(len(result.class_ids), 1)
        self.assertEqual(result.polygons, ())
        self.assertEqual(result.errors[0]["severity"], "hard")
        self.assertAlmostEqual(float(result.errors[0]["max_offset"]), 0.2)
        self.assertEqual(len(result.coordinate_issues), 1)
        issue = result.coordinate_issues[0]
        self.assertEqual(issue["point_index"], 2)
        self.assertEqual(issue["axis"], "x")
        self.assertEqual(issue["coordinate_position"], "point_2_x")
        self.assertAlmostEqual(float(issue["original_value"]), 1.2)
        self.assertAlmostEqual(float(issue["overflow_amount"]), 0.2)
        self.assertEqual(issue["line_severity"], "hard")
        self.assertEqual(issue["reset_value"], "")
        self.assertEqual(issue["action"], "manual_review_then_fix_or_filter")

    def test_hard_line_does_not_reset_other_soft_overflow_coordinates(self) -> None:
        label = self.root / "mixed.txt"
        label.write_text("0 -0.005 0.1 1.2 0.1 0.3 0.3\n", encoding="utf-8")
        result = parse_label(label, {0: "test"})

        self.assertEqual(result.polygons, ())
        self.assertEqual(len(result.coordinate_issues), 2)
        self.assertEqual(
            {issue["coordinate_severity"] for issue in result.coordinate_issues},
            {"soft", "hard"},
        )
        self.assertTrue(all(issue["line_severity"] == "hard" for issue in result.coordinate_issues))
        self.assertTrue(all(issue["reset_value"] == "" for issue in result.coordinate_issues))
        self.assertTrue(
            all(issue["action"] == "manual_review_then_fix_or_filter" for issue in result.coordinate_issues)
        )


if __name__ == "__main__":
    unittest.main()
