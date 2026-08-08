"""Audit immutable test membership and reject test-assisted model selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

import yaml

from blade_defect.data.indexed_splits import load_indexed_splits

SPLITS = ("train", "val", "test")
FORBIDDEN_KEY_FRAGMENTS = (
    "split",
    "threshold",
    "best_epoch",
    "hyperparameter",
    "tuning",
    "selection",
)
CONFIG_SUFFIXES = {".yaml", ".yml", ".json"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _walk(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, item
            yield from _walk(item, path)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{prefix}[{index}]")


def _load_structured(path: Path) -> Any:
    text = path.read_text(encoding="utf-8-sig")
    return json.loads(text) if path.suffix.casefold() == ".json" else yaml.safe_load(text)


def _is_training_config(value: Any) -> bool:
    return isinstance(value, dict) and "model" in value and "data" in value


def _test_violation(key_path: str, value: Any) -> bool:
    key = key_path.rsplit(".", 1)[-1].casefold()
    key_relevant = any(fragment in key for fragment in FORBIDDEN_KEY_FRAGMENTS)
    if key == "test":
        key_relevant = True
    if not key_relevant:
        return False
    if isinstance(value, str):
        normalized = value.replace("\\", "/").casefold().strip()
        return normalized == "test" or normalized.endswith("/test.txt") or normalized.endswith("/images/test")
    return value is True and key == "test"


def _scan_root(root: Path, *, run_root: bool) -> tuple[list[str], list[dict[str, str]]]:
    checked: list[str] = []
    violations: list[dict[str, str]] = []
    if not root.exists():
        return checked, violations
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.casefold() in CONFIG_SUFFIXES):
        try:
            value = _load_structured(path)
        except (OSError, UnicodeError, json.JSONDecodeError, yaml.YAMLError):
            continue
        if not run_root and not _is_training_config(value):
            continue
        checked.append(str(path))
        for key_path, item in _walk(value):
            if _test_violation(key_path, item):
                violations.append(
                    {"file": str(path), "key": key_path, "value": str(item), "reason": "test_used_for_training_or_selection"}
                )
    return checked, violations


def audit_test_lock(dataset: Path, config_root: Path, run_root: Path) -> dict[str, Any]:
    dataset = dataset.resolve()
    manifest_path = dataset / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_yaml = dataset / "data.yaml"
    indexed = load_indexed_splits(data_yaml, SPLITS)
    ids = {split: [sample.sample_id for sample in indexed[split]] for split in SPLITS}
    id_sets = {split: set(values) for split, values in ids.items()}
    overlaps = {
        "train_val": len(id_sets["train"] & id_sets["val"]),
        "train_test": len(id_sets["train"] & id_sets["test"]),
        "val_test": len(id_sets["val"] & id_sets["test"]),
    }
    all_ids = [item for split in SPLITS for item in ids[split]]
    current_test_hash = sha256_file(dataset / "test.txt")
    locked_test_hash = manifest.get("test_lock", {}).get("locked_split_sha256")
    manifest_counts = manifest.get("samples", {})

    config_checked, config_violations = _scan_root(config_root.resolve(), run_root=False)
    run_checked, run_violations = _scan_root(run_root.resolve(), run_root=True)
    integrity_violations: list[dict[str, str]] = []
    if current_test_hash != locked_test_hash:
        integrity_violations.append({"reason": "test_list_hash_changed", "expected": str(locked_test_hash), "actual": current_test_hash})
    if any(overlaps.values()):
        integrity_violations.append({"reason": "split_overlap", "details": json.dumps(overlaps)})
    if len(all_ids) != len(set(all_ids)):
        integrity_violations.append({"reason": "sample_not_exactly_once", "details": f"rows={len(all_ids)}, unique={len(set(all_ids))}"})
    actual_counts = {split: len(values) for split, values in ids.items()}
    if actual_counts != manifest_counts:
        integrity_violations.append({"reason": "manifest_count_mismatch", "details": f"actual={actual_counts}, manifest={manifest_counts}"})

    violations = integrity_violations + config_violations + run_violations
    return {
        "dataset_id": manifest.get("dataset_id"),
        "checked": {
            "dataset_manifest": str(manifest_path),
            "test_index": str(dataset / "test.txt"),
            "training_configs": config_checked,
            "run_manifests": run_checked,
        },
        "test_hash": {"expected": locked_test_hash, "actual": current_test_hash, "matches": current_test_hash == locked_test_hash},
        "counts": actual_counts,
        "overlaps": overlaps,
        "all_samples_exactly_once": len(all_ids) == len(set(all_ids)),
        "violations": violations,
        "valid": not violations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config-root", type=Path, default=Path("configs"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = audit_test_lock(args.dataset, args.config_root, args.run_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
