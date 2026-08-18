"""数据门禁全量图像解码的可审计缓存（成员指纹 + 文件元信息 + 校验版本）。

背景：正式 v3 / OBB 派生数据的全量门禁对 48,291 张图片逐张 OpenCV 解码，
单次约 40—50 分钟纯 CPU，smoke / 校准 / baseline 每次启动都会重复触发。
本脚本在不降低校验强度的前提下固定以下门禁规则：

1. 正式数据首次构建，或成员 / 文件状态发生变化时，执行全量解码；
2. 成员指纹、逐文件元信息（大小 + mtime_ns）与校验版本三者全部一致时，
   复用已通过的解码结果（缓存命中），不再逐张解码；
3. 仅个别文件被修改时，只对变化样本触发解码复检，其余样本复用缓存；
4. 无论命中与否，标签格式与路径存在性每次都全量检查（不进入缓存）；
5. CI / smoke 可使用固定种子的抽样解码（--decode-sample + --decode-seed），
   抽样模式不读也不写正式缓存，报告中单独标记。

缓存键（三者缺一不可，禁止只用旧时间戳或目录名判断数据未变化）：
- 成员指纹：membership_sha256 + 各 split identity_sha256（来自索引 txt）；
- 文件元信息：逐样本图片 / 标签的 (size_bytes, mtime_ns) 聚合 SHA-256；
- 校验版本：CACHE_SCHEMA_VERSION + 校验参数（num_classes、min_area）。

用法：
    python scripts/cache_dataset_validation.py --dataset datasets/blade-v3-grouped-obb
    python scripts/cache_dataset_validation.py --dataset ... --report results/data_gate_cache/cold_run_report.json
    python scripts/cache_dataset_validation.py --dataset ... --decode-sample 512 --decode-seed 42  # CI/smoke 抽样
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml

from blade_defect.data import (
    DEFECT_CLASSES,
    IndexedSample,
    check_obb_label_text,
    is_image_decodable,
    load_indexed_splits,
    membership_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE = PROJECT_ROOT / "results" / "data_gate_cache" / "validation_cache.json"

CACHE_SCHEMA_VERSION = 1
VALIDATOR_ID = "check_obb_indexed_samples"
SPLITS = ("train", "val", "test")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _identity_hash(sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(sample_ids):
        digest.update(f"{sample_id}\n".encode("utf-8"))
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            timeout=15,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _file_meta(path: Path) -> dict[str, Any]:
    """单文件元信息：存在性 + 大小 + 纳秒 mtime（不是单独的旧时间戳判断）。"""
    try:
        stat = path.stat()
    except OSError:
        return {"exists": False, "size": None, "mtime_ns": None}
    return {"exists": True, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def build_file_state(
    indexed: dict[str, list[IndexedSample]],
) -> tuple[dict[str, dict[str, Any]], str]:
    """逐样本文件状态表 + 聚合指纹；返回 (states, file_state_sha256)。

    states 的键为 ``{split}\\t{sample_id}``，值为 image/label 两组元信息。
    """
    states: dict[str, dict[str, Any]] = {}
    digest = hashlib.sha256()
    for split in sorted(indexed):
        for sample in indexed[split]:
            key = f"{split}\t{sample.sample_id}"
            entry = {
                "image": _file_meta(sample.image_path),
                "label": _file_meta(sample.label_path),
            }
            states[key] = entry
            image, label = entry["image"], entry["label"]
            digest.update(
                f"{key}\timg:{image['exists']}:{image['size']}:{image['mtime_ns']}"
                f"\tlbl:{label['exists']}:{label['size']}:{label['mtime_ns']}\n".encode("utf-8")
            )
    return states, digest.hexdigest()


def load_cache(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        return json.loads(cache_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def decide_cache(
    cache: dict[str, Any] | None,
    *,
    membership: str,
    file_state_hash: str,
    num_classes: int,
    min_area: float,
    states: dict[str, dict[str, Any]],
) -> tuple[str, list[str], list[str]]:
    """决定 cold / hit / partial，返回 (decision, reasons, recheck_keys)。"""
    if cache is None:
        return "cold", ["cache_missing"], []
    gate = cache.get("gate") or {}
    if (
        cache.get("cache_schema_version") != CACHE_SCHEMA_VERSION
        or gate.get("validator") != VALIDATOR_ID
        or gate.get("num_classes") != num_classes
        or gate.get("min_area") != min_area
    ):
        return "cold", ["validation_version_changed"], []
    if cache.get("membership_sha256") != membership:
        return "cold", ["membership_changed"], []
    if cache.get("file_state_sha256") == file_state_hash:
        return "hit", ["membership_file_state_and_version_all_match"], []
    cached_states = cache.get("file_states") or {}
    recheck = sorted(
        key
        for key, entry in states.items()
        if cached_states.get(key, {}).get("image") != entry["image"]
    )
    return "partial", ["file_state_changed"], recheck


def _decode_outcomes(
    keys: Iterable[str],
    indexed: dict[str, list[IndexedSample]],
) -> dict[str, bool]:
    outcomes: dict[str, bool] = {}
    index = {
        f"{split}\t{sample.sample_id}": sample
        for split, samples in indexed.items()
        for sample in samples
    }
    for key in keys:
        sample = index[key]
        outcomes[key] = is_image_decodable(sample.image_path)
    return outcomes


def run_cached_validation(
    dataset: Path,
    *,
    cache_path: Path | None = None,
    use_cache: bool = True,
    decode_sample: int | None = None,
    decode_seed: int = 42,
    num_classes: int | None = None,
    min_area: float = 1e-8,
    splits: tuple[str, ...] = SPLITS,
) -> dict[str, Any]:
    """执行一次带缓存的门禁校验，返回完整负载（含 cache 决策块）。"""
    started = time.perf_counter()
    dataset = dataset.resolve()
    data_yaml = dataset / "data.yaml"
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8-sig"))
    if num_classes is None:
        names = config.get("names") or {}
        num_classes = len(names) if names else len(DEFECT_CLASSES)

    index_started = time.perf_counter()
    indexed = load_indexed_splits(data_yaml, splits)
    index_seconds = time.perf_counter() - index_started

    membership = membership_hash(indexed)
    identity_hashes = {
        split: _identity_hash([sample.sample_id for sample in indexed[split]])
        for split in indexed
    }

    stat_started = time.perf_counter()
    states, file_state_hash = build_file_state(indexed)
    stat_seconds = time.perf_counter() - stat_started

    sampled = decode_sample is not None
    cache_file = cache_path if cache_path is not None else DEFAULT_CACHE
    cache = load_cache(cache_file) if (use_cache and not sampled) else None
    if sampled:
        decision, reasons, recheck_keys = "sampled", ["ci_smoke_fixed_seed_sampling"], []
    else:
        decision, reasons, recheck_keys = decide_cache(
            cache,
            membership=membership,
            file_state_hash=file_state_hash,
            num_classes=num_classes,
            min_area=min_area,
            states=states,
        )

    all_keys = sorted(states)
    existing_image_keys = [key for key in all_keys if states[key]["image"]["exists"]]
    if sampled:
        rng = random.Random(decode_seed)
        decode_keys = sorted(
            rng.sample(existing_image_keys, min(decode_sample, len(existing_image_keys)))
        )
    elif decision == "cold":
        decode_keys = existing_image_keys
    elif decision == "partial":
        decode_keys = [key for key in recheck_keys if states[key]["image"]["exists"]]
    else:
        decode_keys = []

    decode_started = time.perf_counter()
    fresh_outcomes = _decode_outcomes(decode_keys, indexed)
    decode_seconds = time.perf_counter() - decode_started

    cached_outcomes: dict[str, bool] = {}
    if cache is not None and decision in {"hit", "partial"}:
        cached_outcomes = {
            key: bool(value)
            for key, value in (cache.get("decode_results") or {}).items()
            if key in states and key not in fresh_outcomes
        }
    decode_results = {**cached_outcomes, **fresh_outcomes}

    label_started = time.perf_counter()
    split_reports: dict[str, dict[str, Any]] = {}
    for split in splits:
        samples = indexed[split]
        report: dict[str, Any] = {
            "valid": True,
            "images": len(samples),
            "labels": 0,
            "valid_instances": 0,
            "negative_images": 0,
            "missing_labels": [],
            "orphan_labels": [],
            "corrupt_images": [],
            "error_type_counts": {},
            "issues": [],
        }
        seen: set[str] = set()
        for sample in samples:
            identity = sample.sample_id
            key = f"{split}\t{identity}"
            if identity in seen:
                report["issues"].append(
                    {
                        "file": identity,
                        "error_type": "duplicate_index_entry",
                        "message": "sample appears more than once in split index",
                        "line": None,
                    }
                )
                continue
            seen.add(identity)
            image_meta = states[key]["image"]
            label_meta = states[key]["label"]
            if not image_meta["exists"]:
                report["corrupt_images"].append(identity)
                report["issues"].append(
                    {
                        "file": identity,
                        "error_type": "missing_image",
                        "message": "indexed image does not exist",
                        "line": None,
                    }
                )
                continue
            decodable = decode_results.get(key)
            if decodable is None:
                # 缓存缺少该键（例如旧缓存被截断）：立即解码补齐，不静默放过。
                decodable = is_image_decodable(sample.image_path)
                decode_results[key] = decodable
                if decision == "hit":
                    decision = "partial"
                    reasons.append("cache_entry_missing")
                if key not in recheck_keys:
                    recheck_keys.append(key)
            if not decodable:
                report["corrupt_images"].append(identity)
                report["issues"].append(
                    {
                        "file": identity,
                        "error_type": "corrupt_image",
                        "message": "OpenCV could not decode image",
                        "line": None,
                    }
                )
            if not label_meta["exists"]:
                report["missing_labels"].append(identity)
                report["issues"].append(
                    {
                        "file": identity,
                        "error_type": "missing_label",
                        "message": "indexed image has no matching label",
                        "line": None,
                    }
                )
                continue
            report["labels"] += 1
            text = sample.label_path.read_text(encoding="utf-8-sig")
            valid_count, negative, issues = check_obb_label_text(
                text, num_classes=num_classes, min_area=min_area
            )
            if negative:
                report["negative_images"] += 1
                continue
            report["valid_instances"] += valid_count
            for error_type, message, line_number in issues:
                report["issues"].append(
                    {
                        "file": identity,
                        "error_type": error_type,
                        "message": message,
                        "line": line_number,
                    }
                )
        report["valid"] = not report["issues"]
        counts: dict[str, int] = {}
        for issue in report["issues"]:
            counts[issue["error_type"]] = counts.get(issue["error_type"], 0) + 1
        report["error_type_counts"] = dict(sorted(counts.items()))
        split_reports[split] = report
    label_seconds = time.perf_counter() - label_started

    sets = {split: {sample.sample_id for sample in indexed[split]} for split in splits}
    overlaps = {
        f"{left}_{right}": sorted(sets[left] & sets[right])
        for index, left in enumerate(splits)
        for right in splits[index + 1 :]
    }
    manifest_path = dataset / "dataset_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest_path.is_file()
        else {}
    )
    parent_hashes = manifest.get("parent_split_identity_sha256", {})
    derived_hashes = manifest.get("derived_split_identity_sha256", {})
    consistency: dict[str, Any] = {
        "split_identity_sha256": identity_hashes,
        "parent_split_identity_sha256": parent_hashes,
        "derived_manifest_identity_sha256": derived_hashes,
        "parent_equals_derived": {
            split: bool(parent_hashes.get(split)) and parent_hashes.get(split) == identity_hashes[split]
            for split in splits
        },
        "cross_split_overlap_counts": {key: len(value) for key, value in overlaps.items()},
        "membership_sha256": membership,
        "manifest_membership_sha256": manifest.get("derived_membership_sha256"),
        "test_data_integrity_only": not bool(
            manifest.get("test_policy", {}).get("allowed_for_training", True)
        ),
    }
    consistency["valid"] = (
        all(consistency["parent_equals_derived"].values())
        and not any(consistency["cross_split_overlap_counts"].values())
        and consistency["membership_sha256"] == consistency["manifest_membership_sha256"]
        and consistency["test_data_integrity_only"]
    )

    total_seconds = time.perf_counter() - started
    cache_block = {
        "decision": decision,
        "decision_reasons": reasons,
        "cache_path": str(cache_file) if not sampled else None,
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "membership_sha256": membership,
        "file_state_sha256": file_state_hash,
        "decoded_images": len(fresh_outcomes),
        "reused_decode_results": len(cached_outcomes),
        "rechecked_sample_ids": sorted(recheck_keys) if decision == "partial" else [],
        "timing_seconds": {
            "total": round(total_seconds, 3),
            "index_load": round(index_seconds, 3),
            "file_stat": round(stat_seconds, 3),
            "image_decode": round(decode_seconds, 3),
            "label_check": round(label_seconds, 3),
        },
    }
    if sampled:
        cache_block["sampled_decode"] = {
            "count": len(fresh_outcomes),
            "seed": decode_seed,
            "scope": "ci_smoke_only_not_written_to_formal_cache",
        }

    payload: dict[str, Any] = {
        "dataset": str(dataset),
        "mode": "indexed_cached",
        "generated_at": _now(),
        "code_commit": _git_commit(),
        "valid": all(report["valid"] for report in split_reports.values())
        and consistency["valid"],
        "splits": split_reports,
        "split_consistency": consistency,
        "cache": cache_block,
    }

    if not sampled and use_cache:
        new_cache = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "validator": VALIDATOR_ID,
            "gate": {
                "validator": VALIDATOR_ID,
                "num_classes": num_classes,
                "min_area": min_area,
            },
            "dataset": str(dataset),
            "membership_sha256": membership,
            "split_identity_sha256": identity_hashes,
            "file_state_sha256": file_state_hash,
            "file_states": states,
            "decode_results": decode_results,
            "last_run_at": payload["generated_at"],
            "last_run_valid": payload["valid"],
            "last_decision": decision,
            "code_commit": payload["code_commit"],
        }
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(new_cache, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        cache_block["cache_written"] = True
    else:
        cache_block["cache_written"] = False
    return payload


def write_run_report(payload: dict[str, Any], report_path: Path) -> None:
    """落盘单次运行的耗时与决策报告（cold / hit / partial 均适用）。"""
    cache = payload["cache"]
    report = {
        "generated_at": payload["generated_at"],
        "dataset": payload["dataset"],
        "code_commit": payload["code_commit"],
        "decision": cache["decision"],
        "decision_reasons": cache["decision_reasons"],
        "valid": payload["valid"],
        "decoded_images": cache["decoded_images"],
        "reused_decode_results": cache["reused_decode_results"],
        "rechecked_sample_ids": cache["rechecked_sample_ids"],
        "timing_seconds": cache["timing_seconds"],
        "membership_sha256": cache["membership_sha256"],
        "file_state_sha256": cache["file_state_sha256"],
        "cache_path": cache["cache_path"],
        "cache_written": cache["cache_written"],
        "splits": {
            split: {
                "images": report_["images"],
                "labels": report_["labels"],
                "valid_instances": report_["valid_instances"],
                "negative_images": report_["negative_images"],
                "corrupt_images": len(report_["corrupt_images"]),
                "missing_labels": len(report_["missing_labels"]),
                "issues": len(report_["issues"]),
                "valid": report_["valid"],
            }
            for split, report_ in payload["splits"].items()
        },
        "split_consistency_valid": payload["split_consistency"]["valid"],
    }
    if "sampled_decode" in cache:
        report["sampled_decode"] = cache["sampled_decode"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-cache", action="store_true", help="忽略已有缓存，强制全量解码（仍写缓存）")
    parser.add_argument("--min-area", type=float, default=1e-8)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--report", type=Path, help="单次运行报告 JSON 输出路径")
    parser.add_argument("--decode-sample", type=int, help="CI/smoke 固定种子抽样解码张数（不读写正式缓存）")
    parser.add_argument("--decode-seed", type=int, default=42)
    args = parser.parse_args()

    payload = run_cached_validation(
        args.dataset,
        cache_path=args.cache,
        use_cache=not args.no_cache,
        decode_sample=args.decode_sample,
        decode_seed=args.decode_seed,
        num_classes=args.num_classes,
        min_area=args.min_area,
    )
    if args.report:
        write_run_report(payload, args.report.resolve())
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if not payload["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
