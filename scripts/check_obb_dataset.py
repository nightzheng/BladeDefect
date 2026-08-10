"""校验 YOLO-OBB 数据集的训练集和验证集。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from blade_defect.data import (
    DEFECT_CLASSES,
    check_obb_dataset,
    check_obb_indexed_samples,
    load_indexed_splits,
    membership_hash,
)


SPLITS = ("train", "val", "test")


def _identity_hash(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        digest.update(f"{sample_id}\n".encode("utf-8"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--min-area", type=float, default=1e-8)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    data_yaml = dataset / "data.yaml"
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    indexed_mode = all(str(config.get(split, "")).casefold().endswith(".txt") for split in SPLITS)
    consistency: dict[str, object] = {}
    if indexed_mode:
        indexed = load_indexed_splits(data_yaml, SPLITS)
        split_reports = {
            split: check_obb_indexed_samples(
                indexed[split], num_classes=len(DEFECT_CLASSES), min_area=args.min_area
            ).to_dict()
            for split in SPLITS
        }
        sets = {split: {sample.sample_id for sample in indexed[split]} for split in SPLITS}
        overlaps = {
            f"{left}_{right}": sorted(sets[left] & sets[right])
            for index, left in enumerate(SPLITS)
            for right in SPLITS[index + 1 :]
        }
        current_hashes = {
            split: _identity_hash([sample.sample_id for sample in indexed[split]])
            for split in SPLITS
        }
        manifest_path = dataset / "dataset_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else {}
        parent_hashes = manifest.get("parent_split_identity_sha256", {})
        derived_hashes = manifest.get("derived_split_identity_sha256", {})
        consistency = {
            "split_identity_sha256": current_hashes,
            "parent_split_identity_sha256": parent_hashes,
            "derived_manifest_identity_sha256": derived_hashes,
            "parent_equals_derived": {
                split: bool(parent_hashes.get(split)) and parent_hashes.get(split) == current_hashes[split]
                for split in SPLITS
            },
            "cross_split_overlap_counts": {key: len(value) for key, value in overlaps.items()},
            "membership_sha256": membership_hash(indexed),
            "manifest_membership_sha256": manifest.get("derived_membership_sha256"),
            "test_data_integrity_only": not bool(manifest.get("test_policy", {}).get("allowed_for_training", True)),
        }
        consistency["valid"] = (
            all(consistency["parent_equals_derived"].values())
            and not any(consistency["cross_split_overlap_counts"].values())
            and consistency["membership_sha256"] == consistency["manifest_membership_sha256"]
            and consistency["test_data_integrity_only"]
        )
    else:
        directory_splits = [
            split for split in SPLITS
            if (dataset / "images" / split).is_dir() and (dataset / "labels" / split).is_dir()
        ]
        split_reports = {
            split: check_obb_dataset(
                dataset / "images" / split,
                dataset / "labels" / split,
                num_classes=len(DEFECT_CLASSES),
                min_area=args.min_area,
            ).to_dict()
            for split in directory_splits
        }
    payload = {
        "dataset": str(dataset),
        "mode": "indexed" if indexed_mode else "directory",
        "valid": (
            all(bool(report["valid"]) for report in split_reports.values())
            and (bool(consistency.get("valid")) if indexed_mode else True)
            and (indexed_mode or {"train", "val"}.issubset(split_reports))
        ),
        "splits": split_reports,
        "split_consistency": consistency,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
