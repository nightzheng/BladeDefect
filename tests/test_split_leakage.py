from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.validate_split_leakage import validate_from_group_summary


class SplitLeakageTests(unittest.TestCase):
    def test_direct_sha_inventory_detects_cross_split_duplicate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            split_output = root / "results" / "split_candidate"
            registry_output = root / "results" / "dataset_registry"
            split_output.mkdir(parents=True)
            registry_output.mkdir(parents=True)
            entries = {
                "train": ["./images/train/a.JPG"],
                "val": ["./images/val/b.JPG"],
                "test": ["./images/train/c.JPG"],
            }
            for split, rows in entries.items():
                (split_output / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            with (split_output / "sample_group_assignments.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["relative_image", "target_split", "group_id", "duplicate_group_id", "sequence_group"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                for split, rows in entries.items():
                    for index, relative in enumerate(rows):
                        writer.writerow(
                            {
                                "relative_image": relative,
                                "target_split": split,
                                "group_id": f"group-{split}-{index}",
                                "duplicate_group_id": "",
                                "sequence_group": f"seq-{split}-{index}",
                            }
                        )
            inventory = registry_output / "image_sha256.csv"
            with inventory.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["relative_image", "sha256"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"relative_image": entries["train"][0], "sha256": "same"},
                        {"relative_image": entries["val"][0], "sha256": "same"},
                        {"relative_image": entries["test"][0], "sha256": "unique"},
                    ]
                )

            report = validate_from_group_summary(
                split_output,
                split_output / "leakage_validation.json",
                inventory,
            )
            self.assertFalse(report["valid"])
            self.assertEqual(report["risk_counts"]["sha256_cross_split"], 1)


if __name__ == "__main__":
    unittest.main()
