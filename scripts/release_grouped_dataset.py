"""Release an audited grouped split as an immutable indexed dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from blade_defect.data.indexed_splits import load_indexed_splits, membership_hash

SPLITS = ("train", "val", "test")
EXPECTED_COUNTS = {"train": 33804, "val": 7244, "test": 7243}
GROUPING_RULE_VERSION = "grouped-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def aggregate_hash(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_immutable(path: Path, content: str | bytes) -> None:
    payload = content.encode("utf-8") if isinstance(content, str) else content
    if path.exists():
        if path.read_bytes() != payload:
            raise FileExistsError(f"immutable release file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _load_candidate(candidate: Path) -> dict[str, Any]:
    manifest_path = candidate / "dataset_manifest.json" if candidate.is_dir() else candidate
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _candidate_lists(candidate_manifest: dict[str, Any], split_source: Path) -> dict[str, Path]:
    outputs = candidate_manifest.get("outputs", {})
    result: dict[str, Path] = {}
    for split in SPLITS:
        configured = outputs.get(split)
        path = Path(configured) if configured else split_source / f"{split}.txt"
        if not path.is_absolute() and not path.is_file():
            path = split_source / path.name
        if not path.is_file():
            raise FileNotFoundError(f"candidate split list not found: {path}")
        result[split] = path.resolve()
    return result


def _normalize_candidate_entry(entry: str, source_dataset: Path, output: Path) -> str:
    path = Path(entry.strip())
    if path.is_absolute():
        parts = path.parts
        lowered = [part.casefold() for part in parts]
        if "images" not in lowered:
            raise ValueError(f"candidate entry has no images component: {entry}")
        position = lowered.index("images")
        source_path = source_dataset.joinpath(*parts[position:])
    else:
        clean = entry.replace("\\", "/")
        while clean.startswith("./"):
            clean = clean[2:]
        source_path = source_dataset / clean
    relative = Path(os.path.relpath(source_path, output)).as_posix()
    return relative if relative.startswith(".") else f"./{relative}"


def _release_lists(
    candidate_lists: dict[str, Path], source_dataset: Path, output: Path
) -> tuple[dict[str, list[str]], dict[str, str]]:
    released: dict[str, list[str]] = {}
    hashes: dict[str, str] = {}
    for split, candidate_list in candidate_lists.items():
        entries = [line.strip() for line in candidate_list.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        released[split] = [_normalize_candidate_entry(entry, source_dataset, output) for entry in entries]
        content = "\n".join(released[split]) + "\n"
        target = output / f"{split}.txt"
        _write_immutable(target, content)
        hashes[split] = sha256_file(target)
    return released, hashes


def _load_names(source_dataset: Path) -> Any:
    data = yaml.safe_load((source_dataset / "data.yaml").read_text(encoding="utf-8-sig"))
    return data["names"]


def _migration_rows(assignments: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with assignments.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            exact = bool(row.get("duplicate_group_id"))
            sequence = bool(row.get("sequence_group"))
            reasons: list[str] = []
            if exact:
                reasons.append("SHA组")
            if sequence:
                reasons.append("序列组")
            reasons.append("组级均衡")
            rows.append(
                {
                    "sample_id": row["relative_image"].replace("\\", "/").removeprefix("./").casefold(),
                    "relative_image": row["relative_image"],
                    "old_split": row["source_split"],
                    "new_split": row["target_split"],
                    "changed": str(row["source_split"] != row["target_split"]).lower(),
                    "group_id": row["group_id"],
                    "migration_reason": "+".join(reasons),
                    "field_status": "inferred_from_filename",
                    "is_negative": str(not row.get("class_ids", "").strip()).lower(),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _distribution_rows(source: Path, migration: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if source.is_file():
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                for split in SPLITS:
                    rows.append(
                        {
                            "dimension": row["level"],
                            "category": row["class_key"],
                            "category_name": row["class_name"],
                            "split": split,
                            "images": row[f"{split}_images"],
                            "instances": row[f"{split}_instances"],
                        }
                    )
    changed = Counter(row["new_split"] for row in migration if row["changed"] == "true")
    for split in SPLITS:
        negative_count = sum(
            row["new_split"] == split and row["is_negative"] == "true" for row in migration
        )
        rows.append(
            {
                "dimension": "negative_sample",
                "category": "empty_label",
                "category_name": "空标签负样本",
                "split": split,
                "images": str(negative_count),
                "instances": "0",
            }
        )
        rows.append(
            {
                "dimension": "migration",
                "category": "changed_from_old_split",
                "category_name": "旧划分发生变化",
                "split": split,
                "images": str(changed[split]),
                "instances": "",
            }
        )
    return rows


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def release_dataset(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    source_dataset = args.source_dataset.resolve()
    candidate_manifest_path = (args.candidate / "dataset_manifest.json") if args.candidate.is_dir() else args.candidate
    candidate_manifest = _load_candidate(args.candidate)
    audit_summary = json.loads(args.audit_summary.read_text(encoding="utf-8"))
    if not audit_summary.get("valid"):
        raise ValueError(f"grouping audit has not passed: {args.audit_summary}")
    candidate_lists = _candidate_lists(candidate_manifest, args.split_source)

    output.mkdir(parents=True, exist_ok=True)
    released, split_hashes = _release_lists(candidate_lists, source_dataset, output)
    counts = {split: len(entries) for split, entries in released.items()}
    expected = EXPECTED_COUNTS if not args.allow_nonstandard_counts else counts
    if counts != expected:
        raise ValueError(f"unexpected split counts: {counts}; expected {expected}")

    names = _load_names(source_dataset)
    data_yaml = yaml.safe_dump(
        {"path": ".", "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": names},
        allow_unicode=True,
        sort_keys=False,
    )
    _write_immutable(output / "data.yaml", data_yaml)
    indexed = load_indexed_splits(output / "data.yaml", SPLITS)
    all_ids = [sample.sample_id for samples in indexed.values() for sample in samples]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("formal split lists overlap")
    missing_images = [str(sample.image_path) for samples in indexed.values() for sample in samples if not sample.image_path.is_file()]
    missing_labels = [str(sample.label_path) for samples in indexed.values() for sample in samples if not sample.label_path.is_file()]
    if missing_images or missing_labels:
        raise FileNotFoundError(
            f"release references missing files: images={len(missing_images)}, labels={len(missing_labels)}"
        )

    migration = _migration_rows(args.assignments)
    if len(migration) != sum(counts.values()):
        raise ValueError(f"migration row count mismatch: {len(migration)}")
    migration_path = args.results / "split_migration.csv"
    _write_csv(migration_path, migration, list(migration[0]))

    distribution = _distribution_rows(args.distribution_source, migration)
    distribution_path = args.results / "split_class_distribution.csv"
    _write_csv(
        distribution_path,
        distribution,
        ["dimension", "category", "category_name", "split", "images", "instances"],
    )

    candidate_hashes = {split: sha256_file(path) for split, path in candidate_lists.items()}
    source_manifest = json.loads((source_dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
    instance_count = int(candidate_manifest.get("instances", 0))
    label_hash = candidate_manifest.get("frozen_labels_sha256", source_manifest.get("frozen_labels_sha256"))
    existing_manifest_path = output / "dataset_manifest.json"
    existing_manifest = (
        json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        if existing_manifest_path.is_file()
        else {}
    )
    generated_at = (
        args.generated_at
        or existing_manifest.get("generated_at")
        or datetime.now().astimezone().isoformat()
    )
    manifest = {
        "dataset_id": args.dataset_id,
        "status": "released",
        "generated_at": generated_at,
        "parent_dataset_id": candidate_manifest.get("source_dataset_id", source_manifest.get("dataset_id")),
        "candidate_dataset_id": candidate_manifest.get("dataset_id"),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "generation_commit": _git_commit(),
        "random_seed": candidate_manifest.get("random_seed", 42),
        "grouping_rule_version": GROUPING_RULE_VERSION,
        "split_policy": candidate_manifest.get("split_policy"),
        "split_ratio_target": candidate_manifest.get("split_ratio_target"),
        "samples": counts,
        "labels": counts,
        "instances": instance_count,
        "candidate_split_sha256": candidate_hashes,
        "release_split_sha256": split_hashes,
        "candidate_lists_sha256": aggregate_hash(candidate_lists[split] for split in SPLITS),
        "release_lists_sha256": aggregate_hash(output / f"{split}.txt" for split in SPLITS),
        "membership_sha256": membership_hash(indexed),
        "frozen_labels_sha256": label_hash,
        "field_inference": {
            field: "inferred_from_filename_not_owner_confirmed"
            for field in ("blade_id", "flight_batch", "capture_key", "sequence_group")
        },
        "test_lock": {
            "policy": "test is readable only for owner-approved final evaluation",
            "automatic_unlock": False,
            "locked_split_sha256": split_hashes["test"],
        },
        "storage": {
            "type": "relative_txt_indexes_to_parent_frozen_dataset",
            "source_dataset": Path(os.path.relpath(source_dataset, output)).as_posix(),
            "copies_images": False,
            "ntfs_junction_is_dataset_identity": False,
        },
        "outputs": {
            "split_migration": str(migration_path.as_posix()),
            "split_class_distribution": str(distribution_path.as_posix()),
            "grouping_audit_summary": str(args.audit_summary.as_posix()),
        },
    }
    manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    _write_immutable(output / "dataset_manifest.json", manifest_text)

    validation = {
        "dataset_id": args.dataset_id,
        "valid": True,
        "counts": counts,
        "total_samples": len(all_ids),
        "unique_samples": len(set(all_ids)),
        "missing_images": len(missing_images),
        "missing_labels": len(missing_labels),
        "split_overlap": 0,
        "all_samples_exactly_once": True,
        "all_fine_and_coarse_classes_present": candidate_manifest.get("balance_summary", {}).get(
            "all_fine_and_coarse_classes_present", False
        ),
        "manifest_sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
    }
    args.results.mkdir(parents=True, exist_ok=True)
    validation_path = args.results / "release_validation.json"
    _write_immutable(validation_path, json.dumps(validation, ensure_ascii=False, indent=2) + "\n")
    return validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--split-source", required=True, type=Path)
    parser.add_argument("--source-dataset", type=Path, default=Path("datasets/blade-v2"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dataset-id", default="blade-v3-grouped-202608")
    parser.add_argument("--assignments", type=Path, default=Path("results/split_candidate/sample_group_assignments.csv"))
    parser.add_argument("--distribution-source", type=Path, default=Path("results/split_candidate/class_distribution.csv"))
    parser.add_argument("--results", type=Path, default=Path("results/dataset_release"))
    parser.add_argument("--audit-summary", type=Path, default=Path("results/dataset_release/grouping_audit_summary.json"))
    parser.add_argument("--generated-at", help="fixed timestamp for deterministic tests")
    parser.add_argument("--allow-nonstandard-counts", action="store_true", help=argparse.SUPPRESS)
    return parser


def main() -> None:
    result = release_dataset(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
