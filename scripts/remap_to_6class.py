"""将 15 类 YOLO-seg 标签映射为 6 个粗粒度类别。

映射来自 configs/class_hierarchy.yaml：
  表面腐蚀(0) ← 0,1,2,3
  表面裂纹(1) ← 4,5
  表面缺陷(2) ← 6,7,8,9
  维修痕迹(3) ← 10
  叶片损伤(4) ← 11,12,13
  附件脱落(5) ← 14
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from blade_defect.data.class_hierarchy import fine_to_coarse, load_class_hierarchy
from blade_defect.data import check_dataset, DEFECT_CLASSES
from blade_defect.data.indexed_splits import load_indexed_splits, membership_hash
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


def _remap_label_file(src: Path, dst: Path, mapping: dict[int, int]) -> None:
    lines = src.read_text(encoding="utf-8").strip().splitlines()
    new_lines = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        old_cls = int(parts[0])
        new_cls = mapping[old_cls]
        new_lines.append(f"{new_cls} " + " ".join(parts[1:]))
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=resolve_path,
                        help="原始 15 类数据集根目录（含 images/ 和 labels/）")
    parser.add_argument("--output", type=resolve_path,
                        help="6 类输出目录（必须为空）")
    parser.add_argument("--source-data", type=Path, help="包含 train/val/test txt 清单的 data.yaml")
    parser.add_argument("--dry-run-index-check", action="store_true", help="仅检查全量清单兼容性，不写文件")
    parser.add_argument("--copy-images", action="store_true",
                        help="复制图片；默认不复制，请手动在 images/ 下创建 junction")
    args = parser.parse_args()

    if args.dry_run_index_check:
        if args.source_data is None:
            parser.error("--dry-run-index-check requires --source-data")
        report = dry_run_index_check(args.source_data)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["valid"]:
            raise SystemExit(1)
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
