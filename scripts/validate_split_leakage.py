"""Validate grouped candidate split lists for exact and sequence leakage."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")


def load_split_entries(split_output: Path) -> dict[str, list[str]]:
    return {
        split: [
            line.strip()
            for line in (split_output / f"{split}.txt").read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        for split in SPLITS
    }


def load_split_lists(split_output: Path) -> dict[str, set[str]]:
    return {split: set(entries) for split, entries in load_split_entries(split_output).items()}


def validate_from_group_summary(
    split_output: Path,
    output: Path,
    hash_inventory: Path | None = None,
) -> dict[str, object]:
    split_entries = load_split_entries(split_output)
    split_lists = load_split_lists(split_output)
    duplicate_members: dict[str, set[str]] = defaultdict(set)
    sequence_members: dict[str, set[str]] = defaultdict(set)
    group_members: dict[str, set[str]] = defaultdict(set)
    assignment_mismatches: list[dict[str, str]] = []
    sample_to_split = {
        sample: split for split, samples in split_lists.items() for sample in samples
    }
    assignment_path = split_output / "sample_group_assignments.csv"
    with assignment_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            relative_image = row["relative_image"]
            split = sample_to_split.get(relative_image, "missing")
            if split != row["target_split"]:
                assignment_mismatches.append(
                    {
                        "relative_image": relative_image,
                        "declared_split": row["target_split"],
                        "actual_split": split,
                    }
                )
            group_members[row["group_id"]].add(split)
            if row["duplicate_group_id"]:
                duplicate_members[row["duplicate_group_id"]].add(split)
            if row["sequence_group"]:
                sequence_members[row["sequence_group"]].add(split)

    duplicate_violations = {key: sorted(value) for key, value in duplicate_members.items() if len(value) > 1}
    sequence_violations = {key: sorted(value) for key, value in sequence_members.items() if len(value) > 1}
    group_violations = {key: sorted(value) for key, value in group_members.items() if len(value) > 1}
    overlap_pairs: list[dict[str, object]] = []
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlap = sorted(split_lists[left] & split_lists[right])
            if overlap:
                overlap_pairs.append({"left": left, "right": right, "count": len(overlap), "examples": overlap[:10]})

    duplicate_entries = {
        split: sorted(sample for sample, count in Counter(entries).items() if count > 1)
        for split, entries in split_entries.items()
    }
    inventory_path = hash_inventory or split_output.parent / "dataset_registry" / "image_sha256.csv"
    hash_splits: dict[str, set[str]] = defaultdict(set)
    missing_hash_entries: list[str] = []
    inventory: dict[str, str] = {}
    if inventory_path.exists():
        with inventory_path.open("r", encoding="utf-8-sig", newline="") as handle:
            inventory = {row["relative_image"]: row["sha256"] for row in csv.DictReader(handle)}
        for split, entries in split_lists.items():
            for entry in entries:
                digest = inventory.get(entry)
                if digest:
                    hash_splits[digest].add(split)
                else:
                    missing_hash_entries.append(entry)
    sha_violations = {digest: sorted(splits) for digest, splits in hash_splits.items() if len(splits) > 1}

    sample_counts = {split: len(values) for split, values in split_lists.items()}
    total = sum(sample_counts.values())
    report = {
        "valid": not any(
            (
                duplicate_violations,
                sequence_violations,
                group_violations,
                overlap_pairs,
                assignment_mismatches,
                sha_violations,
                missing_hash_entries,
                [value for value in duplicate_entries.values() if value],
            )
        ),
        "sample_counts": sample_counts,
        "sample_ratios": {split: sample_counts[split] / total if total else 0 for split in SPLITS},
        "duplicate_group_cross_split": duplicate_violations,
        "sequence_group_cross_split": sequence_violations,
        "group_id_cross_split": group_violations,
        "sha256_cross_split": sha_violations,
        "sample_list_overlaps": overlap_pairs,
        "duplicate_entries_within_split": duplicate_entries,
        "assignment_mismatches": assignment_mismatches[:100],
        "missing_hash_entries": missing_hash_entries[:100],
        "risk_counts": {
            "duplicate_group_cross_split": len(duplicate_violations),
            "sequence_group_cross_split": len(sequence_violations),
            "group_id_cross_split": len(group_violations),
            "sha256_cross_split": len(sha_violations),
            "sample_list_overlap_pairs": len(overlap_pairs),
            "duplicate_entries_within_split": sum(len(value) for value in duplicate_entries.values()),
            "assignment_mismatches": len(assignment_mismatches),
            "missing_hash_entries": len(missing_hash_entries),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_dataset_files(dataset: Path, split_output: Path, output: Path) -> dict[str, object]:
    errors: list[dict[str, str]] = []
    class_counts: Counter[int] = Counter()
    class_counts_by_split: dict[str, Counter[int]] = {split: Counter() for split in SPLITS}
    sample_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()
    split_entries = load_split_entries(split_output)
    source_entries = {
        line.strip()
        for source_split in ("train", "val")
        for line in (dataset / f"{source_split}.txt").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    }
    candidate_counter = Counter(entry for entries in split_entries.values() for entry in entries)
    duplicate_candidates = sorted(entry for entry, count in candidate_counter.items() if count > 1)
    candidate_entries = set(candidate_counter)
    missing_from_candidate = sorted(source_entries - candidate_entries)
    unknown_candidate_entries = sorted(candidate_entries - source_entries)
    for relative_image in duplicate_candidates:
        errors.append({"split": "multiple", "relative_image": relative_image, "reason": "duplicate_candidate_entry"})
    for relative_image in missing_from_candidate:
        errors.append({"split": "missing", "relative_image": relative_image, "reason": "source_sample_missing"})
    for relative_image in unknown_candidate_entries:
        errors.append({"split": "unknown", "relative_image": relative_image, "reason": "unknown_candidate_sample"})
    for split in SPLITS:
        for relative_image in sorted(set(split_entries[split])):
            sample_counts[split] += 1
            image_path = (dataset / relative_image).resolve()
            label_path = dataset / "labels" / Path(relative_image).parts[-2] / Path(relative_image).with_suffix(".txt").name
            if not image_path.exists():
                errors.append({"split": split, "relative_image": relative_image, "reason": "missing_image"})
                continue
            if not label_path.exists():
                errors.append({"split": split, "relative_image": relative_image, "reason": "missing_label"})
                continue
            for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8-sig").splitlines(), start=1):
                line = raw_line.strip()
                if not line:
                    continue
                tokens = line.split()
                try:
                    class_value = float(tokens[0])
                    coordinates = [float(value) for value in tokens[1:]]
                except ValueError:
                    errors.append({"split": split, "relative_image": relative_image, "reason": f"non_numeric:{line_number}"})
                    continue
                if not class_value.is_integer() or not 0 <= int(class_value) <= 14:
                    errors.append({"split": split, "relative_image": relative_image, "reason": f"invalid_class:{line_number}"})
                if len(coordinates) < 6 or len(coordinates) % 2:
                    errors.append({"split": split, "relative_image": relative_image, "reason": f"invalid_polygon:{line_number}"})
                if any(value < 0.0 or value > 1.0 for value in coordinates):
                    errors.append({"split": split, "relative_image": relative_image, "reason": f"out_of_range:{line_number}"})
                class_counts[int(class_value)] += 1
                class_counts_by_split[split][int(class_value)] += 1
                instance_counts[split] += 1
    expected_class_ids = sorted(class_counts)
    missing_classes_by_split = {
        split: [class_id for class_id in expected_class_ids if class_counts_by_split[split][class_id] == 0]
        for split in SPLITS
    }
    for split, class_ids in missing_classes_by_split.items():
        for class_id in class_ids:
            errors.append({"split": split, "relative_image": "", "reason": f"missing_class:{class_id}"})
    report = {
        "valid": not errors,
        "samples": dict(sample_counts),
        "source_sample_count": len(source_entries),
        "candidate_unique_sample_count": len(candidate_entries),
        "source_coverage_complete": not missing_from_candidate and not unknown_candidate_entries,
        "instances": sum(instance_counts.values()),
        "instances_by_split": dict(instance_counts),
        "expected_class_ids": expected_class_ids,
        "class_instances": {str(class_id): class_counts[class_id] for class_id in expected_class_ids},
        "class_instances_by_split": {
            split: {str(class_id): class_counts_by_split[split][class_id] for class_id in expected_class_ids}
            for split in SPLITS
        },
        "missing_classes_by_split": missing_classes_by_split,
        "all_expected_classes_present_in_each_split": not any(missing_classes_by_split.values()),
        "all_15_classes_present_in_each_split": len(expected_class_ids) == 15 and not any(missing_classes_by_split.values()),
        "duplicate_candidate_entries": duplicate_candidates[:100],
        "missing_source_samples": missing_from_candidate[:100],
        "unknown_candidate_samples": unknown_candidate_entries[:100],
        "errors": errors[:100],
        "error_count": len(errors),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/blade-v2"))
    parser.add_argument("--split-output", type=Path, default=Path("results/split_candidate"))
    parser.add_argument("--hash-inventory", type=Path, default=Path("results/dataset_registry/image_sha256.csv"))
    parser.add_argument("--leakage-output", type=Path, default=Path("results/split_candidate/leakage_validation.json"))
    parser.add_argument("--dataset-output", type=Path, default=Path("results/split_candidate/dataset_validation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "leakage": validate_from_group_summary(args.split_output, args.leakage_output, args.hash_inventory),
        "dataset": validate_dataset_files(args.dataset, args.split_output, args.dataset_output),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
