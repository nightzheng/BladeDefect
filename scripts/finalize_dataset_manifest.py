"""Finalize blade-v2 hashes, manifest, and before/after audit tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_filter_decisions(path: Path) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_images = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line == "images:":
            in_images = True
            continue
        if not in_images or not line or line.startswith("#"):
            continue
        if line.startswith("- filename:"):
            if current:
                entries.append(current)
            current = {"filename": line.split(":", 1)[1].strip().strip('"\'')}
        elif current is not None and ":" in line:
            key, value = line.split(":", 1)
            current[key.strip()] = value.strip().strip('"\'')
    if current:
        entries.append(current)
    return entries


def aggregate_label_hash(dataset: Path, workers: int) -> str:
    paths = sorted((dataset / "labels").rglob("*.txt"))

    def hash_one(path: Path) -> tuple[str, str]:
        return path.relative_to(dataset).as_posix(), sha256_file(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(hash_one, paths, chunksize=64))
    digest = hashlib.sha256()
    for relative, file_hash in results:
        digest.update(relative.encode("utf-8"))
        digest.update(file_hash.encode("ascii"))
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def finalize(args: argparse.Namespace) -> dict[str, object]:
    dataset = args.dataset.resolve()
    validation = json.loads(args.validation.read_text(encoding="utf-8"))
    if validation.get("valid") is not True:
        raise ValueError("cannot finalize a dataset whose strict validation did not pass")
    samples = validation["samples"]
    if sum(samples.values()) != 48291 or validation["instances"] != 49471:
        raise ValueError(f"unexpected final totals: samples={samples}, instances={validation['instances']}")
    if validation["unreadable_images"] or validation["out_of_range_coordinates"]:
        raise ValueError("strict validation still contains unreadable images or bad coordinates")

    list_digest = hashlib.sha256()
    for name in ("train.txt", "val.txt"):
        list_digest.update((dataset / name).read_bytes())
    label_hash = aggregate_label_hash(dataset, args.workers)
    decisions = load_filter_decisions(args.filter_config)
    exclude_entries = [entry for entry in decisions if entry.get("action") == "exclude"]
    negative_entries = [entry for entry in decisions if entry.get("action") == "keep_negative"]
    if len(exclude_entries) != 25 or len(negative_entries) != 5:
        raise ValueError(
            f"unexpected filter totals: exclude={len(exclude_entries)}, negative={len(negative_entries)}"
        )

    manifest = {
        "dataset_id": "blade-v2",
        "generated_at": validation["generated_at"],
        "source": {"images": str(args.images.resolve()), "labels": str(args.labels.resolve())},
        "filter_config": {"path": str(args.filter_config.resolve()), "sha256": sha256_file(args.filter_config)},
        "hard_error_review": {"path": str(args.hard_review.resolve()), "sha256": sha256_file(args.hard_review)},
        "random_seed": args.seed,
        "split_policy": "preserve_source_train_val; no test split created",
        "image_storage": "NTFS directory junctions to read-only source image directories",
        "sample_lists_sha256": list_digest.hexdigest(),
        "frozen_labels_sha256": label_hash,
        "samples": samples,
        "instances": validation["instances"],
        "filter_summary": {
            "configured_images": len(decisions),
            "excluded_images": len(exclude_entries),
            "negative_images": len(negative_entries),
            "corrupted_images_excluded": sum(entry.get("issue") == "corrupted_file" for entry in exclude_entries),
        },
        "repair_summary": {
            "soft_coordinates_reset": validation["soft_coordinates_reset"],
            "hard_coordinates_repaired_after_review": validation["hard_coordinates_repaired"],
        },
        "validation": {"valid": True, "report": str(args.validation.resolve())},
        "generation_command": args.command,
    }
    (dataset / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    change_rows = [
        {"metric": "images", "before": 48316, "after": 48291, "difference": -25},
        {"metric": "label_files", "before": 48302, "after": 48291, "difference": -11},
        {"metric": "instances_after_first_week_filter", "before": 49477, "after": 49471, "difference": -6},
        {"metric": "out_of_range_annotation_lines", "before": 729, "after": 0, "difference": -729},
        {"metric": "out_of_range_coordinates", "before": 949, "after": 0, "difference": -949},
        {"metric": "full_decode_failures_in_selected_set", "before": 6, "after": 0, "difference": -6},
        {"metric": "virtual_empty_negative_labels", "before": 5, "after": 5, "difference": 0},
    ]
    write_csv(args.review_output / "dataset_change_summary.csv", change_rows, ["metric", "before", "after", "difference"])

    image_locations = {
        path.name.casefold(): (split, path)
        for split in ("train", "val")
        for path in (args.images / split).iterdir()
        if path.is_file()
    }
    filtered_rows: list[dict[str, object]] = []
    for entry in exclude_entries:
        split, image_path = image_locations[entry["filename"].casefold()]
        filtered_rows.append(
            {
                "split": split,
                "image_path": str(image_path),
                "action": "exclude",
                "issue": entry.get("issue", ""),
                "reason": entry.get("reason", ""),
                "status": entry.get("status", ""),
                "source": "dataset_filter",
            }
        )
    write_csv(
        args.review_output / "filtered_source_images.csv",
        filtered_rows,
        ["split", "image_path", "action", "issue", "reason", "status", "source"],
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--filter-config", type=Path, required=True)
    parser.add_argument("--hard-review", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--review-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--command", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    parsed = parse_args()
    print(json.dumps(finalize(parsed), ensure_ascii=False, indent=2))
