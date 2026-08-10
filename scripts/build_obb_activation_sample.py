"""Build a deterministic, split-preserving sample index for OBB activation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from blade_defect.data.indexed_splits import load_indexed_splits, membership_hash
from convert_seg_to_obb import convert_polygon_to_obb


SPLITS = ("train", "val", "test")
PRIORITY_CLASSES = {12: 5, 13: 4, 14: 5}
FEATURES = {
    "small",
    "medium",
    "large",
    "elongated",
    "tilted",
    "multi_instance",
    "edge_touching",
    "irregular_polygon",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_hash(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        digest.update(f"{sample_id}\n".encode("utf-8"))
    return digest.hexdigest()


def _polygon_area(points: np.ndarray) -> float:
    return abs(float(cv2.contourArea(points.astype(np.float32))))


def _sample_metadata(sample) -> dict[str, object]:
    lines = [line.split() for line in sample.label_path.read_text(encoding="utf-8-sig").splitlines() if line.split()]
    classes: Counter[int] = Counter()
    features: set[str] = set()
    maximum_points = 0
    with Image.open(sample.image_path) as image:
        image_size = image.size
    for tokens in lines:
        class_id = int(float(tokens[0]))
        polygon = np.asarray([float(value) for value in tokens[1:]], dtype=float).reshape(-1, 2)
        result = convert_polygon_to_obb(polygon, image_size=image_size)
        box = result.corners
        classes[class_id] += 1
        maximum_points = max(maximum_points, len(polygon))
        area = abs(float(cv2.contourArea(box.astype(np.float32))))
        if area < 0.0025:
            features.add("small")
        elif area < 0.05:
            features.add("medium")
        else:
            features.add("large")
        sides = np.linalg.norm(np.roll(box, -1, axis=0) - box, axis=1)
        positive = sides[sides > 1e-12]
        if positive.size and positive.max() / positive.min() >= 4.0:
            features.add("elongated")
        longest = int(np.argmax(sides))
        edge = box[(longest + 1) % 4] - box[longest]
        angle = abs(float(np.degrees(np.arctan2(edge[1], edge[0])))) % 90
        if 10 <= angle <= 80:
            features.add("tilted")
        if np.any((polygon <= 0.02) | (polygon >= 0.98)):
            features.add("edge_touching")
        fill_ratio = _polygon_area(polygon) / area if area > 0 else 1.0
        if len(polygon) >= 8 and fill_ratio < 0.7:
            features.add("irregular_polygon")
    if len(lines) >= 2:
        features.add("multi_instance")
    return {
        "sample": sample,
        "classes": classes,
        "features": features,
        "instances": len(lines),
        "maximum_polygon_points": maximum_points,
    }


def _select(records: list[dict[str, object]], count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    remaining = list(records)
    rng.shuffle(remaining)
    selected: list[dict[str, object]] = []
    class_counts: Counter[int] = Counter()
    covered_features: set[str] = set()

    def score(record: dict[str, object]) -> tuple[float, int, int, str]:
        classes: Counter[int] = record["classes"]  # type: ignore[assignment]
        features: set[str] = record["features"]  # type: ignore[assignment]
        value = 0.0
        for class_id in classes:
            if class_counts[class_id] == 0:
                value += 100.0
            target = PRIORITY_CLASSES.get(class_id, 1)
            if class_counts[class_id] < target:
                value += 35.0 * (target - class_counts[class_id])
        value += 30.0 * len(features - covered_features)
        value += 4.0 * min(int(record["instances"]), 5)
        value += min(int(record["maximum_polygon_points"]), 30) / 10
        sample = record["sample"]
        return value, len(classes), len(features), sample.sample_id

    while remaining and len(selected) < count:
        best_index = max(range(len(remaining)), key=lambda index: score(remaining[index]))
        chosen = remaining.pop(best_index)
        selected.append(chosen)
        class_counts.update(chosen["classes"])  # type: ignore[arg-type]
        covered_features.update(chosen["features"])  # type: ignore[arg-type]
    return selected


def build_sample(parent: Path, output: Path, manifest_csv: Path, count_per_split: int, seed: int) -> dict[str, object]:
    data_yaml = parent / "data.yaml"
    parent_manifest_path = parent / "dataset_manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8-sig"))
    indexed = load_indexed_splits(data_yaml, SPLITS)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    selected_by_split: dict[str, list[dict[str, object]]] = {}
    for split_index, split in enumerate(SPLITS):
        records = [_sample_metadata(sample) for sample in indexed[split]]
        selected = _select(records, min(count_per_split, len(records)), seed + split_index)
        selected_by_split[split] = selected
        entries = [str(record["sample"].image_path) for record in selected]
        (output / f"{split}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
        for rank, record in enumerate(selected, 1):
            sample = record["sample"]
            classes: Counter[int] = record["classes"]  # type: ignore[assignment]
            features: set[str] = record["features"]  # type: ignore[assignment]
            rows.append(
                {
                    "split": split,
                    "selection_rank": rank,
                    "sample_id": sample.sample_id,
                    "parent_index": str(sample.index_file),
                    "parent_index_line": sample.line_number,
                    "parent_image": str(sample.image_path),
                    "parent_label": str(sample.label_path),
                    "class_ids": " ".join(str(value) for value in sorted(classes)),
                    "class_instance_counts": " ".join(f"{key}:{classes[key]}" for key in sorted(classes)),
                    "instances": record["instances"],
                    "size_buckets": " ".join(sorted(features & {"small", "medium", "large"})),
                    "features": " ".join(sorted(features)),
                    "maximum_polygon_points": record["maximum_polygon_points"],
                    "selection_seed": seed,
                }
            )

    names = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))["names"]
    (output / "data.yaml").write_text(
        yaml.safe_dump(
            {"path": ".", "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": names},
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    sampled_indexed = load_indexed_splits(output / "data.yaml", SPLITS)
    split_hashes = {
        split: _identity_hash([sample.sample_id for sample in sampled_indexed[split]]) for split in SPLITS
    }
    sample_manifest = {
        "dataset_id": "blade-v3-grouped-202608-obb-activation-sample",
        "purpose": "split_preserving_obb_activation_sample",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parent_dataset_id": parent_manifest.get("dataset_id"),
        "parent_dataset_path": str(parent),
        "parent_manifest_sha256": _sha256(parent_manifest_path),
        "parent_membership_sha256": parent_manifest.get("membership_sha256"),
        "samples": {split: len(selected_by_split[split]) for split in SPLITS},
        "sample_membership_sha256": membership_hash(sampled_indexed),
        "split_identity_sha256": split_hashes,
        "selection": {
            "method": "deterministic greedy stratification by class, priority classes, size and geometry features",
            "seed": seed,
            "count_per_split": count_per_split,
            "preserves_parent_split": True,
            "reassigns_groups": False,
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(sample_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    fields = list(rows[0])
    with manifest_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return sample_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest-csv", required=True, type=Path)
    parser.add_argument("--count-per-split", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = build_sample(
        args.parent.resolve(), args.output.resolve(), args.manifest_csv.resolve(), args.count_per_split, args.seed
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
