"""Build exact SHA-256 duplicate groups for a frozen YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_sample_list(dataset: Path, split: str) -> list[tuple[str, Path]]:
    entries: list[tuple[str, Path]] = []
    list_path = dataset / f"{split}.txt"
    for raw_line in list_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        entries.append((line, (dataset / line).resolve()))
    return entries


def build_hash_inventory(dataset: Path, workers: int = 8) -> list[dict[str, object]]:
    samples = [
        (split, relative, path)
        for split in ("train", "val")
        for relative, path in resolve_sample_list(dataset, split)
    ]

    def hash_sample(sample: tuple[str, str, Path]) -> tuple[str, str, Path, str]:
        split, relative, path = sample
        return split, relative, path, sha256_file(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        hashed = list(executor.map(hash_sample, samples, chunksize=64))

    return [
        {
            "source_split": split,
            "relative_image": relative,
            "sha256": digest,
            "file_size": path.stat().st_size,
        }
        for split, relative, path, digest in sorted(hashed, key=lambda item: (item[0], item[1]))
    ]


def duplicate_rows_from_inventory(inventory: list[dict[str, object]]) -> list[dict[str, object]]:
    by_hash: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in inventory:
        by_hash[str(row["sha256"])].append(row)

    rows: list[dict[str, object]] = []
    group_index = 0
    for digest, group in sorted(by_hash.items(), key=lambda item: (item[0], len(item[1]))):
        if len(group) < 2:
            continue
        group_index += 1
        group_id = f"sha256-{digest[:16]}"
        splits = sorted({str(row["source_split"]) for row in group})
        for row in sorted(group, key=lambda item: (str(item["source_split"]), str(item["relative_image"]))):
            rows.append(
                {
                    "duplicate_group_id": group_id,
                    "sha256": digest,
                    "group_size": len(group),
                    "split": row["source_split"],
                    "relative_image": row["relative_image"],
                    "file_size": row["file_size"],
                    "cross_split": len(splits) > 1,
                    "group_index": group_index,
                }
            )
    return rows


def build_exact_duplicate_rows(dataset: Path, workers: int = 8) -> list[dict[str, object]]:
    return duplicate_rows_from_inventory(build_hash_inventory(dataset, workers=workers))


def write_hash_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["source_split", "relative_image", "sha256", "file_size"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_duplicate_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "duplicate_group_id",
        "sha256",
        "group_size",
        "split",
        "relative_image",
        "file_size",
        "cross_split",
        "group_index",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inventory = build_hash_inventory(args.dataset, workers=args.workers)
    rows = duplicate_rows_from_inventory(inventory)
    write_duplicate_csv(args.output, rows)
    if args.inventory_output:
        write_hash_inventory(args.inventory_output, inventory)
    print(f"images hashed: {len(inventory)}; duplicate rows: {len(rows)}")


if __name__ == "__main__":
    main()
