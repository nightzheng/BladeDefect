"""将 15 类 YOLO-seg 标签映射为 6 个粗粒度类别。

映射来自 configs/class_hierarchy.yaml：
  表面腐蚀(0) ← 0,1,2,3
  表面裂纹(1) ← 4,5
  表面缺陷(2) ← 6,7,8,9
  维修痕迹(3) ← 10
  叶片损伤(4) ← 11,12,13
  附件脱落(5) ← 14

两种转换模式：
- 目录模式（--source/--output）：按 images/{split}/ 目录成员关系转换 train/val。
- 清单模式（--source-data/--output）：按 txt 清单的 train/val/test 成员关系转换，
  图片硬链接进扁平 images/{split}/，标签重映射写入 labels/{split}/，
  生成含 test 条目与 6 类名的目录式 data.yaml 和 dataset_manifest.json，
  写后执行 strict 校验。适用于 blade-v3-grouped 这类 split 成员关系与
  源目录不一致的分组数据集（目录模式会静默错配回源目录成员关系）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

from blade_defect.data.class_hierarchy import fine_to_coarse, load_class_hierarchy
from blade_defect.data import check_dataset, DEFECT_CLASSES
from blade_defect.data.indexed_splits import (
    load_index_config,
    load_indexed_splits,
    membership_hash,
)
from blade_defect.utils import resolve_path

HIERARCHY_PATH = Path(__file__).resolve().parent.parent / "configs" / "class_hierarchy.yaml"
COARSE_NAMES: dict[int, str] = {
    0: "表面腐蚀",
    1: "表面裂纹",
    2: "表面缺陷",
    3: "维修痕迹",
    4: "叶片损伤",
    5: "附件脱落",
}
SPLITS = ("train", "val")
INDEX_SPLITS = ("train", "val", "test")

_fine_to_coarse_cache: dict[int, int] | None = None


def _get_mapping() -> dict[int, int]:
    global _fine_to_coarse_cache
    if _fine_to_coarse_cache is not None:
        return _fine_to_coarse_cache
    groups = load_class_hierarchy(HIERARCHY_PATH)
    f2c = fine_to_coarse(groups)
    mapping: dict[int, int] = {}
    for cid in range(15):
        mapping[cid] = f2c[cid].class_ids[0]
    coarse_keys = sorted({g.class_ids[0] for g in groups})
    reindex = {old: new for new, old in enumerate(coarse_keys)}
    _fine_to_coarse_cache = {old: reindex[coarse] for old, coarse in mapping.items()}
    return _fine_to_coarse_cache


def _remap_label_lines(text: str, mapping: dict[int, int]) -> tuple[str, int, Counter[int]]:
    """重映射标签文本，返回新文本、实例数和按新类别的计数。"""
    new_lines: list[str] = []
    instances = 0
    class_counts: Counter[int] = Counter()
    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        parts = raw_line.split()
        old_cls = int(parts[0])
        new_cls = mapping[old_cls]
        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
        instances += 1
        class_counts[new_cls] += 1
    output = "\n".join(new_lines)
    return (output + "\n" if output else ""), instances, class_counts


def _remap_label_file(src: Path, dst: Path, mapping: dict[int, int]) -> None:
    text, _, _ = _remap_label_lines(src.read_text(encoding="utf-8-sig"), mapping)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")


def _link_or_copy(source: Path, target: Path) -> str:
    """同卷硬链接（零额外空间），失败时回退 copy2（跨卷等情况）。"""
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dry_run_index_check(source_data: str | Path) -> dict[str, object]:
    """Verify that every indexed label is readable and maps from 15 to 6 classes."""
    indexed = load_indexed_splits(source_data, ("train", "val", "test"))
    mapping = _get_mapping()
    errors: list[str] = []
    instances = 0
    for samples in indexed.values():
        for sample in samples:
            if not sample.image_path.is_file():
                errors.append(f"missing image: {sample.sample_id}")
                continue
            if not sample.label_path.is_file():
                errors.append(f"missing label: {sample.sample_id}")
                continue
            for line_number, raw_line in enumerate(sample.label_path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if not raw_line.strip():
                    continue
                try:
                    class_id = int(raw_line.split()[0])
                    mapping[class_id]
                except (ValueError, KeyError, IndexError):
                    errors.append(f"invalid class: {sample.sample_id}:{line_number}")
                instances += 1
    return {
        "tool": "remap_to_6class",
        "splits": {split: len(samples) for split, samples in indexed.items()},
        "total_samples": sum(len(samples) for samples in indexed.values()),
        "instances": instances,
        "membership_sha256": membership_hash(indexed),
        "errors": errors,
        "valid": not errors,
    }


def convert_from_index(
    source_data: str | Path,
    output: str | Path,
    *,
    generation_command: str | None = None,
) -> dict[str, object]:
    """按 txt 清单成员关系生成 6 类数据集。

    图片按清单逐张硬链接进扁平 images/{split}/，标签重映射写入 labels/{split}/，
    不受源目录成员关系影响。生成目录式 data.yaml（含 test 条目、6 类名）和
    dataset_manifest.json（label_level=coarse、derived_from、membership_sha256、
    instances 等），写后执行 strict 校验与扁平成员关系复核。
    """
    source_data = Path(source_data)
    output = Path(output)
    indexed = load_indexed_splits(source_data, INDEX_SPLITS)
    _, config = load_index_config(source_data)
    source_root = Path(config["path"])
    source_manifest: dict[str, object] = {}
    source_manifest_path = source_root / "dataset_manifest.json"
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8-sig"))

    membership = membership_hash(indexed)
    source_membership = source_manifest.get("membership_sha256")
    if isinstance(source_membership, str) and source_membership != membership:
        raise ValueError(
            "清单成员关系与源 manifest 不一致: "
            f"{membership} != {source_membership}（{source_manifest_path}）"
        )

    mapping = _get_mapping()
    print("15 → 6 映射：")
    for old, new in sorted(mapping.items()):
        print(f"  {old:2d} {DEFECT_CLASSES[old]} → {new} {COARSE_NAMES[new]}")

    # 第一遍：校验 split 内文件名零冲突、图片/标签存在、类别可映射，并统计实例数。
    split_stats: dict[str, dict[str, int]] = {}
    class_counts: Counter[int] = Counter()
    total_instances = 0
    for split in INDEX_SPLITS:
        samples = indexed[split]
        seen: dict[str, Path] = {}
        for sample in samples:
            key = sample.image_path.name.casefold()
            if key in seen:
                raise ValueError(
                    f"{split} 内文件名冲突，无法安全扁平化: {seen[key]} 与 {sample.image_path}"
                )
            seen[key] = sample.image_path
        split_instances = 0
        for sample in samples:
            if not sample.image_path.is_file():
                raise FileNotFoundError(f"缺失图片: {sample.image_path}")
            if not sample.label_path.is_file():
                raise FileNotFoundError(f"缺失标签: {sample.label_path}")
            text = sample.label_path.read_text(encoding="utf-8-sig")
            _, instances, counts = _remap_label_lines(text, mapping)
            split_instances += instances
            class_counts.update(counts)
        split_stats[split] = {"samples": len(samples), "instances": split_instances}
        total_instances += split_instances

    expected_instances = source_manifest.get("instances")
    if isinstance(expected_instances, int) and expected_instances != total_instances:
        raise ValueError(
            f"实例数与源 manifest 不一致: {total_instances} != {expected_instances}"
        )
    expected_samples = source_manifest.get("samples")
    if isinstance(expected_samples, dict):
        for split, expected in expected_samples.items():
            actual = split_stats.get(split, {}).get("samples")
            if actual != expected:
                raise ValueError(f"{split} 样本数与源 manifest 不一致: {actual} != {expected}")

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录不为空: {output}")
    output.mkdir(parents=True, exist_ok=True)

    try:
        # 第二遍：硬链接图片、写入重映射标签。
        storage_counts = {"hardlink": 0, "copy": 0}
        for split in INDEX_SPLITS:
            dst_img_dir = output / "images" / split
            dst_lbl_dir = output / "labels" / split
            dst_img_dir.mkdir(parents=True, exist_ok=True)
            dst_lbl_dir.mkdir(parents=True, exist_ok=True)
            for sample in indexed[split]:
                kind = _link_or_copy(sample.image_path, dst_img_dir / sample.image_path.name)
                storage_counts[kind] += 1
                text = sample.label_path.read_text(encoding="utf-8-sig")
                remapped, _, _ = _remap_label_lines(text, mapping)
                (dst_lbl_dir / sample.label_path.name).write_text(remapped, encoding="utf-8")
            print(f"  {split}: {split_stats[split]['samples']} 图片, "
                  f"{split_stats[split]['instances']} 实例")

        data_yaml = output / "data.yaml"
        with data_yaml.open("w", encoding="utf-8", newline="\n") as f:
            yaml.safe_dump({
                "path": ".",
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": COARSE_NAMES,
            }, f, allow_unicode=True, sort_keys=False)

        print("\nstrict 校验:")
        all_ok = True
        validation: dict[str, dict[str, object]] = {}
        for split in INDEX_SPLITS:
            report = check_dataset(
                output / "images" / split,
                output / "labels" / split,
                num_classes=6,
                polygon_mode="strict",
                dry_run=True,
            )
            written_images = {
                p.name.casefold()
                for p in (output / "images" / split).iterdir()
                if p.is_file()
            }
            expected_images = {s.image_path.name.casefold() for s in indexed[split]}
            membership_ok = written_images == expected_images
            ok = report.valid and membership_ok
            if not ok:
                all_ok = False
            validation[split] = {
                "valid": report.valid,
                "membership_ok": membership_ok,
                "images": report.images,
                "labels": report.labels,
                "issues": len(report.issues),
            }
            ok_flag = "OK" if ok else "FAIL"
            print(f"  {split}: {ok_flag}  images={report.images}  labels={report.labels}  "
                  f"issues={len(report.issues)}  membership_ok={membership_ok}")
        if not all_ok:
            raise RuntimeError(
                "strict 校验失败: " + json.dumps(validation, ensure_ascii=False)
            )

        # 校验通过后才落盘 manifest，避免半成品携带 status=derived 的身份声明。
        source_dataset_id = source_manifest.get("dataset_id") or source_root.name
        manifest: dict[str, object] = {
            "dataset_id": output.resolve().name,
            "status": "derived",
            "label_level": "coarse",
            "derived_from": source_dataset_id,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "generation_command": generation_command
            or f"python scripts/remap_to_6class.py --source-data {source_data} --output {output}",
            "mapping": {
                "config": "configs/class_hierarchy.yaml",
                "fine_to_coarse": {str(old): new for old, new in sorted(mapping.items())},
            },
            "source_dataset": str(source_root),
            "source_dataset_manifest_sha256": (
                _sha256_file(source_manifest_path) if source_manifest_path.is_file() else None
            ),
            "source_membership_sha256": source_membership,
            "membership_sha256": membership,
            "samples": {split: stats["samples"] for split, stats in split_stats.items()},
            "instances": total_instances,
            "class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
            "storage": {
                "type": "flat_hardlinked_images_directory_layout",
                "copies_images": False,
                "hardlink_fallback": "copy2",
                "images_hardlinked": storage_counts["hardlink"],
                "images_copied": storage_counts["copy"],
            },
        }
        source_test_lock = source_manifest.get("test_lock")
        if isinstance(source_test_lock, dict):
            inherited = {k: v for k, v in source_test_lock.items() if k != "locked_split_sha256"}
            inherited["inherited_from"] = source_dataset_id
            manifest["test_lock"] = inherited
        manifest_path = output / "dataset_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except BaseException:
        # 输出目录由本次调用创建且只含硬链接与生成物，失败时整体清理以便直接重试。
        shutil.rmtree(output, ignore_errors=True)
        raise

    return {
        "tool": "remap_to_6class",
        "mode": "index",
        "source_data": str(source_data),
        "output": str(output),
        "splits": split_stats,
        "instances": total_instances,
        "class_counts": manifest["class_counts"],
        "membership_sha256": membership,
        "storage": storage_counts,
        "validation": validation,
        "manifest": str(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=resolve_path,
                        help="原始 15 类数据集根目录（含 images/ 和 labels/），目录模式")
    parser.add_argument("--output", type=resolve_path,
                        help="6 类输出目录（必须为空）")
    parser.add_argument("--source-data", type=Path,
                        help="清单模式：包含 train/val/test txt 清单的 data.yaml；"
                             "与 --output 连用，按清单成员关系生成 6 类数据集")
    parser.add_argument("--dry-run-index-check", action="store_true", help="仅检查全量清单兼容性，不写文件")
    parser.add_argument("--copy-images", action="store_true",
                        help="复制图片（仅目录模式）；默认不复制，请手动在 images/ 下创建 junction")
    args = parser.parse_args()

    if args.dry_run_index_check:
        if args.source_data is None:
            parser.error("--dry-run-index-check requires --source-data")
        report = dry_run_index_check(args.source_data)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["valid"]:
            raise SystemExit(1)
        return
    if args.source_data is not None:
        if args.source is not None:
            parser.error("--source-data 清单模式与 --source 目录模式互斥")
        if args.output is None:
            parser.error("清单模式需要 --output")
        summary = convert_from_index(
            args.source_data,
            args.output,
            generation_command="python " + " ".join(sys.argv),
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.source is None or args.output is None:
        parser.error("normal conversion requires --source and --output")

    source: Path = args.source
    output: Path = args.output

    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录不为空: {output}")
    output.mkdir(parents=True, exist_ok=True)

    mapping = _get_mapping()
    print("15 → 6 映射：")
    for old, new in sorted(mapping.items()):
        print(f"  {old:2d} {DEFECT_CLASSES[old]} → {new} {COARSE_NAMES[new]}")

    stats: dict[str, int] = {}
    for split in SPLITS:
        src_img = source / "images" / split
        src_lbl = source / "labels" / split
        if not src_img.is_dir():
            print(f"跳过 {split}: {src_img} 不存在")
            continue

        dst_img = output / "images" / split
        dst_lbl = output / "labels" / split

        if args.copy_images:
            print(f"复制 {split} 图片...")
            for img in src_img.rglob("*"):
                if not img.is_file():
                    continue
                rel = img.relative_to(src_img)
                (dst_img / rel.parent).mkdir(parents=True, exist_ok=True)
                shutil.copy2(img, dst_img / rel)
        else:
            dst_img.mkdir(parents=True, exist_ok=True)
            print(f"跳过图片复制（当前模式），images/{split} 目录已创建。"
                  " 请手动创建 junction 指向 {src_img}。")

        print(f"重映射 {split} 标签...")
        count = 0
        for lbl in src_lbl.rglob("*.txt"):
            rel = lbl.relative_to(src_lbl)
            _remap_label_file(lbl, dst_lbl / rel, mapping)
            count += 1
        stats[split] = count
        print(f"  {split}: {count} 标签已写入")

    data_yaml = output / "data.yaml"
    with data_yaml.open("w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump({
            "path": ".",
            "train": "images/train",
            "val": "images/val",
            "names": COARSE_NAMES,
        }, f, allow_unicode=True, sort_keys=False)
    print(f"data.yaml: {data_yaml}")

    print("\nstrict 校验:")
    all_ok = True
    for split in SPLITS:
        img_dir = output / "images" / split
        lbl_dir = output / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        report = check_dataset(img_dir, lbl_dir, num_classes=6, polygon_mode="strict", dry_run=True)
        ok_flag = "OK" if report.valid else "FAIL"
        if not report.valid:
            all_ok = False
        print(f"  {split}: {ok_flag}  images={report.images}  labels={report.labels}  issues={len(report.issues)}")

    if all_ok:
        print("\n全部校验通过")
    else:
        print("\n存在校验问题，请检查输出")


if __name__ == "__main__":
    main()
