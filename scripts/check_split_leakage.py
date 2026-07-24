"""Review train/val leakage risk using sequence metadata and perceptual hashes."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image


DSC_PATTERN = re.compile(r"^(?P<prefix>.*?)-?DSC(?P<frame>\d{5})", re.IGNORECASE)
BLADE_PATTERN = re.compile(r"^(?P<blade>-?\d+-\d+)")


def _resolve_list(dataset: Path, split: str) -> list[Path]:
    result: list[Path] = []
    for line in (dataset / f"{split}.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append((dataset / line.strip()).resolve())
    return result


def _dhash(path: Path) -> int:
    with Image.open(path) as image:
        image.draft("L", (64, 64))
        reduced = image.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = list(reduced.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            value = (value << 1) | (pixels[row * 9 + column] > pixels[row * 9 + column + 1])
    return value


def _metadata(path: Path, split: str) -> dict[str, object]:
    match = DSC_PATTERN.search(path.stem)
    blade = BLADE_PATTERN.search(path.stem)
    try:
        dhash: int | None = _dhash(path)
        decode_error = ""
    except OSError as error:
        dhash = None
        decode_error = str(error)
    return {
        "split": split,
        "path": path,
        "name": path.name,
        "capture_key": match.group("prefix").rstrip("-") if match else "",
        "frame": int(match.group("frame")) if match else None,
        "blade_key": blade.group("blade") if blade else "unparsed",
        "dhash": dhash,
        "decode_error": decode_error,
    }


def _row(**values: object) -> dict[str, object]:
    fields = {
        "risk_type": "", "group_key": "", "train_image": "", "val_image": "",
        "frame_delta": "", "dhash_distance": "", "train_group_count": "",
        "val_group_count": "", "note": "",
    }
    fields.update(values)
    return fields


def analyze(dataset: Path, output: Path, workers: int, near_threshold: int) -> dict[str, object]:
    split_paths = {split: _resolve_list(dataset, split) for split in ("train", "val")}
    jobs = [(path, split) for split, paths in split_paths.items() for path in paths]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(lambda item: _metadata(item[0], item[1]), jobs, chunksize=32))
    by_split = {split: [row for row in records if row["split"] == split] for split in ("train", "val")}
    rows: list[dict[str, object]] = []
    decode_failures = [row for row in records if row["dhash"] is None]
    for record in decode_failures:
        rows.append(
            _row(
                risk_type="full_decode_error",
                group_key=record["split"],
                train_image=record["path"] if record["split"] == "train" else "",
                val_image=record["path"] if record["split"] == "val" else "",
                note=f"完整像素解码失败：{record['decode_error']}",
            )
        )
    records = [row for row in records if row["dhash"] is not None]
    by_split = {split: [row for row in records if row["split"] == split] for split in ("train", "val")}

    # Same blade/camera-side groups appearing in both partitions are a grouping-risk signal.
    blade_counts = {
        split: Counter(str(row["blade_key"]) for row in by_split[split])
        for split in ("train", "val")
    }
    for key in sorted(set(blade_counts["train"]) & set(blade_counts["val"])):
        rows.append(
            _row(
                risk_type="same_blade_group_split",
                group_key=key,
                train_group_count=blade_counts["train"][key],
                val_group_count=blade_counts["val"][key],
                note="同一文件名前缀组跨train/val；需结合业务确认该前缀是否对应同一叶片或相机侧面",
            )
        )

    # Adjacent frames in the same capture key are high-priority leakage candidates.
    capture_train: dict[str, list[dict[str, object]]] = defaultdict(list)
    capture_val: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in by_split["train"]:
        if row["capture_key"] and row["frame"] is not None:
            capture_train[str(row["capture_key"])].append(row)
    for row in by_split["val"]:
        if row["capture_key"] and row["frame"] is not None:
            capture_val[str(row["capture_key"])].append(row)
    seen_pairs: set[tuple[str, str]] = set()
    for key in sorted(set(capture_train) & set(capture_val)):
        for train in capture_train[key]:
            for val in capture_val[key]:
                delta = abs(int(train["frame"]) - int(val["frame"]))
                if delta <= 3:
                    pair = (str(train["path"]), str(val["path"]))
                    seen_pairs.add(pair)
                    distance = (int(train["dhash"]) ^ int(val["dhash"])).bit_count()
                    rows.append(
                        _row(
                            risk_type="adjacent_sequence_cross_split",
                            group_key=key,
                            train_image=train["path"],
                            val_image=val["path"],
                            frame_delta=delta,
                            dhash_distance=distance,
                            note="同一捕获键且DSC帧号相邻；建议按序列/叶片重新分组划分",
                        )
                    )

    # Exact perceptual-hash groups are recorded before the wider LSH search.
    hashes: dict[int, dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {"train": [], "val": []}
    )
    for record in records:
        hashes[int(record["dhash"])][str(record["split"])].append(record)
    for value, group in hashes.items():
        if not group["train"] or not group["val"]:
            continue
        for train in group["train"][:10]:
            for val in group["val"][:10]:
                pair = (str(train["path"]), str(val["path"]))
                if pair in seen_pairs:
                    continue
                rows.append(
                    _row(
                        risk_type="identical_perceptual_hash",
                        group_key=f"{value:016x}",
                        train_image=train["path"],
                        val_image=val["path"],
                        dhash_distance=0,
                        train_group_count=len(group["train"]),
                        val_group_count=len(group["val"]),
                        note="感知哈希完全一致；每个大组最多列出100对，需人工确认是否近重复",
                    )
                )

    # LSH: sharing at least one identical 16-bit chunk yields near-duplicate candidates.
    buckets: dict[tuple[int, int], dict[str, list[dict[str, object]]]] = defaultdict(
        lambda: {"train": [], "val": []}
    )
    for row in records:
        value = int(row["dhash"])
        for chunk_index in range(4):
            chunk = (value >> (chunk_index * 16)) & 0xFFFF
            buckets[(chunk_index, chunk)][str(row["split"])].append(row)
    near_pairs: dict[tuple[str, str], int] = {}
    near_truncated = False
    for bucket in buckets.values():
        if not bucket["train"] or not bucket["val"]:
            continue
        if len(bucket["train"]) * len(bucket["val"]) > 50_000:
            continue
        for train in bucket["train"]:
            for val in bucket["val"]:
                pair = (str(train["path"]), str(val["path"]))
                if pair in near_pairs:
                    continue
                distance = (int(train["dhash"]) ^ int(val["dhash"])).bit_count()
                if distance <= near_threshold:
                    near_pairs[pair] = distance
                    if len(near_pairs) >= 50_000:
                        near_truncated = True
                        break
            if near_truncated:
                break
        if near_truncated:
            break
    for (train_path, val_path), distance in sorted(near_pairs.items(), key=lambda item: (item[1], item[0])):
        if (train_path, val_path) in seen_pairs:
            continue
        rows.append(
            _row(
                risk_type="perceptual_near_duplicate",
                group_key=f"dhash_distance<={near_threshold}",
                train_image=train_path,
                val_image=val_path,
                dhash_distance=distance,
                note="感知哈希高度相似；这是候选风险，最终需人工查看原图确认",
            )
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(_row())
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(str(row["risk_type"]) for row in rows)
    summary = {
        "dataset": str(dataset.resolve()),
        "train_images": len(split_paths["train"]),
        "val_images": len(split_paths["val"]),
        "near_duplicate_threshold": near_threshold,
        "full_decode_failures": len(decode_failures),
        "risk_counts": dict(counts),
        "near_pair_search_truncated": near_truncated,
        "interpretation": "风险清单是待复核候选，不等同于已确认数据泄漏；本周不据此随机构造test集。",
    }
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def finalize_after_filter(dataset: Path, output: Path, near_threshold: int) -> dict[str, object]:
    """Drop decode-error rows after those images have been filtered from blade-v2."""
    with output.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["risk_type"] != "full_decode_error"]
    fields = list(_row())
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["risk_type"] for row in rows)
    summary = {
        "dataset": str(dataset.resolve()),
        "train_images": len(_resolve_list(dataset, "train")),
        "val_images": len(_resolve_list(dataset, "val")),
        "near_duplicate_threshold": near_threshold,
        "full_decode_failures": 0,
        "risk_counts": dict(counts),
        "near_pair_search_truncated": False,
        "reuse_note": "复用全量感知哈希扫描；6张完整解码失败图片未进入哈希配对，加入filter后仅移除其错误行。",
        "interpretation": "风险清单是待复核候选，不等同于已确认数据泄漏；本周不据此随机构造test集。",
    }
    output.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--near-threshold", type=int, default=4)
    parser.add_argument("--finalize-after-filter", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = (
        finalize_after_filter(args.dataset, args.output, args.near_threshold)
        if args.finalize_after_filter
        else analyze(args.dataset, args.output, args.workers, args.near_threshold)
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
