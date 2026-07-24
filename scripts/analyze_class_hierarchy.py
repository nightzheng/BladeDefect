"""Aggregate a frozen 15-class YOLO-seg dataset into six coarse classes."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "blade_defect" / "data" / "class_hierarchy.py"
SPEC = importlib.util.spec_from_file_location("blade_class_hierarchy", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
fine_to_coarse = MODULE.fine_to_coarse
load_class_hierarchy = MODULE.load_class_hierarchy


def analyze(dataset: Path, hierarchy_path: Path, output: Path) -> list[dict[str, object]]:
    groups = load_class_hierarchy(hierarchy_path)
    mapping = fine_to_coarse(groups)
    image_counts = {split: Counter() for split in ("train", "val")}
    instance_counts = {split: Counter() for split in ("train", "val")}

    for split in ("train", "val"):
        label_root = dataset / "labels" / split
        for label_path in sorted(label_root.rglob("*.txt")):
            groups_in_image: set[str] = set()
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id = int(float(line.split()[0]))
                coarse = mapping[class_id]
                instance_counts[split][coarse.key] += 1
                groups_in_image.add(coarse.key)
            image_counts[split].update(groups_in_image)

    rows: list[dict[str, object]] = []
    for coarse_id, group in enumerate(groups):
        train_images = image_counts["train"][group.key]
        val_images = image_counts["val"][group.key]
        train_instances = instance_counts["train"][group.key]
        val_instances = instance_counts["val"][group.key]
        rows.append(
            {
                "coarse_id": coarse_id,
                "coarse_key": group.key,
                "coarse_name": group.name_zh,
                "fine_class_ids": ",".join(map(str, group.class_ids)),
                "train_images": train_images,
                "val_images": val_images,
                "total_images": train_images + val_images,
                "train_instances": train_instances,
                "val_instances": val_instances,
                "total_instances": train_instances + val_instances,
            }
        )
    if any(int(row["train_instances"]) == 0 or int(row["val_instances"]) == 0 for row in rows):
        raise ValueError("coarse hierarchy contains an empty class in train or val")
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "coarse_class_distribution.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for row in analyze(args.dataset, args.hierarchy, args.output):
        print(row)
