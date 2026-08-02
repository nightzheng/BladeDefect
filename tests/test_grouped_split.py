from __future__ import annotations

import csv
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from PIL import Image

from scripts.build_grouped_split import build_grouped_split
from scripts.validate_split_leakage import validate_dataset_files, validate_from_group_summary


def _image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (128, 128), color).save(path)


def _label(path: Path, class_id: int, size: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if size == "small":
        polygon = "0.10 0.10 0.12 0.10 0.12 0.12 0.10 0.12"
    elif size == "medium":
        polygon = "0.10 0.10 0.30 0.10 0.30 0.30 0.10 0.30"
    else:
        polygon = "0.10 0.10 0.70 0.10 0.70 0.70 0.10 0.70"
    path.write_text(f"{class_id} {polygon}\n", encoding="utf-8")


def _make_dataset(root: Path) -> tuple[Path, Path, Path]:
    dataset = root / "datasets" / "blade-v2"
    hierarchy = root / "configs" / "class_hierarchy.yaml"
    data = dataset / "data.yaml"
    hierarchy.parent.mkdir(parents=True, exist_ok=True)
    hierarchy.write_text(
        """version: 1
groups:
  surface:
    name_zh: surface
    class_ids: [0, 1]
  damage:
    name_zh: damage
    class_ids: [2]
""",
        encoding="utf-8",
    )
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(
        """path: .
train: train.txt
val: val.txt
names:
  0: a
  1: b
  2: c
""",
        encoding="utf-8",
    )
    lists = {"train": [], "val": []}
    for index in range(60):
        source_split = "train" if index % 5 else "val"
        class_id = index % 3
        size = ("small", "medium", "large")[index % 3]
        capture = 1000 + index
        filename = f"1-{index % 2}-{capture}-DSC{index:05d}.JPG"
        relative = f"./images/{source_split}/{filename}"
        _image(dataset / "images" / source_split / filename, (index, class_id * 50, 120))
        _label(dataset / "labels" / source_split / Path(filename).with_suffix(".txt").name, class_id, size)
        lists[source_split].append(relative)

    # These two images share one sequence and byte-identical content across source splits.
    sequence_samples = [
        ("train", "1-0-9999-DSC00001.JPG"),
        ("val", "1-0-9999-DSC00002.JPG"),
    ]
    for source_split, filename in sequence_samples:
        relative = f"./images/{source_split}/{filename}"
        _image(dataset / "images" / source_split / filename, (255, 255, 255))
        _label(dataset / "labels" / source_split / Path(filename).with_suffix(".txt").name, 0, "large")
        lists[source_split].append(relative)
    for split, rows in lists.items():
        (dataset / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (dataset / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "samples": {split: len(rows) for split, rows in lists.items()},
                "instances": sum(len(rows) for rows in lists.values()),
                "sample_lists_sha256": "source-sample-hash",
                "frozen_labels_sha256": "source-label-hash",
                "generation_command": "test fixture",
            }
        ),
        encoding="utf-8",
    )
    return dataset, data, hierarchy


class GroupedSplitTests(unittest.TestCase):
    def test_grouped_split_is_leakage_free_balanced_and_portable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, data, hierarchy = _make_dataset(root)
            registry_output = root / "results" / "dataset_registry"
            split_output = root / "results" / "split_candidate"
            manifest_output = root / "datasets" / "blade-v3-grouped"
            result = build_grouped_split(
                Namespace(
                    dataset=dataset,
                    data=data,
                    hierarchy=hierarchy,
                    registry_output=registry_output,
                    split_output=split_output,
                    manifest_output=manifest_output,
                    hash_inventory=None,
                    leakage_source=root / "missing.csv",
                    workers=2,
                    seed=42,
                    command="test",
                )
            )

            self.assertEqual(result["sample_count"], 62)
            self.assertTrue(result["portable"]["portable"])
            leakage = validate_from_group_summary(split_output, split_output / "leakage_validation.json")
            dataset_report = validate_dataset_files(dataset, split_output, split_output / "dataset_validation.json")
            self.assertTrue(leakage["valid"])
            self.assertTrue(dataset_report["valid"])

            with (split_output / "sample_group_assignments.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            sequence_rows = [row for row in rows if row["capture_key"] == "1-0-9999"]
            self.assertEqual(len(sequence_rows), 2)
            self.assertEqual(len({row["target_split"] for row in sequence_rows}), 1)
            self.assertEqual(len({row["duplicate_group_id"] for row in sequence_rows}), 1)

            with (split_output / "class_distribution.csv").open(
                encoding="utf-8-sig", newline=""
            ) as handle:
                distribution = list(csv.DictReader(handle))
            fine_total = sum(
                int(row["total_instances"]) for row in distribution if row["level"] == "fine_15"
            )
            coarse_total = sum(
                int(row["total_instances"]) for row in distribution if row["level"] == "coarse_6"
            )
            self.assertEqual(fine_total, coarse_total)

            counts = result["manifest"]["samples"]
            ratios = {split: counts[split] / 62 for split in ("train", "val", "test")}
            self.assertLess(abs(ratios["train"] - 0.70), 0.08)
            self.assertLess(abs(ratios["val"] - 0.15), 0.08)
            self.assertLess(abs(ratios["test"] - 0.15), 0.08)

    def test_same_seed_produces_same_candidate_lists(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, data, hierarchy = _make_dataset(root)
            outputs: list[list[str]] = []
            for run in ("a", "b"):
                split_output = root / run / "split"
                build_grouped_split(
                    Namespace(
                        dataset=dataset,
                        data=data,
                        hierarchy=hierarchy,
                        registry_output=root / run / "registry",
                        split_output=split_output,
                        manifest_output=root / run / "manifest",
                        hash_inventory=None,
                        leakage_source=root / "missing.csv",
                        workers=2,
                        seed=42,
                        command="test",
                    )
                )
                outputs.append(
                    [(split_output / f"{split}.txt").read_text(encoding="utf-8") for split in ("train", "val", "test")]
                )
            self.assertEqual(outputs[0], outputs[1])


if __name__ == "__main__":
    unittest.main()
