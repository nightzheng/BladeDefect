#!/usr/bin/env python
"""删除同类重复标注（同一处缺陷被标成两个几乎重合的多边形）。

背景：Round 3 标注审计在类 13（结构损伤）发现 3 对重复——同一处缺陷被标了两次。
类 13 在 val 只有 25 个实例，6 个重复占 24%，足以让单类 AP 波动 ±0.05。

安全设计（这是改数据集原件，必须留退路）：
  1. 默认 **dry-run**，只打印将删除哪些实例，不写任何文件；
  2. `--apply` 才真正修改，且修改前把原标签复制到 `--backup-dir`（保留相对路径）；
  3. 每次修改记录 sha256（改前/改后）、被删/保留实例的面积与 IoU，写入 manifest；
  4. 只删多边形，不新增、不改类别、不动图片；
  5. 同一文件内若有多对重叠，按"保留面积较大的、删较小的"逐对处理，处理前重算 IoU。

用法：
  # 1) 先看会删什么
  python scripts/fix_duplicate_labels.py --data-yaml <data.yaml> --classes 13 --dry-run
  # 2) 确认后执行（自动备份）
  python scripts/fix_duplicate_labels.py --data-yaml <data.yaml> --classes 13 --apply \
      --backup-dir /data/backup/round3/label_backup_20261022 \
      --manifest /data/backup/round3/duplicate_fix_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.data.class_copy_paste import (
    LabelInstance,
    format_label_line,
    polygon_area,
    polygon_iou,
    read_label_file,
)
from blade_defect.data.indexed_splits import load_indexed_split
from blade_defect.utils.files import save_json


@dataclass
class FixRecord:
    label_path: str
    split: str
    kept_index: int
    removed_index: int
    iou: float
    kept_area: float
    removed_area: float
    sha256_before: str
    sha256_after: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plan_file(path: Path, targets: set[int], iou_threshold: float) -> tuple[list[tuple[int, int, float, float, float]], list[LabelInstance]]:
    """返回 (删除计划, 处理后的实例列表)。计划项 = (保留idx, 删除idx, iou, 保留面积, 删除面积)。"""
    instances = read_label_file(path)
    drop: set[int] = set()
    plan: list[tuple[int, int, float, float, float]] = []

    for left in range(len(instances)):
        if left in drop or instances[left].class_id not in targets:
            continue
        for right in range(left + 1, len(instances)):
            if right in drop or instances[right].class_id != instances[left].class_id:
                continue
            iou = polygon_iou(instances[left].polygon, instances[right].polygon)
            if iou < iou_threshold:
                continue
            area_left = polygon_area(instances[left].polygon)
            area_right = polygon_area(instances[right].polygon)
            # 保留面积较大的（通常框得更完整），删除较小的
            if area_left >= area_right:
                keep, removed, area_keep, area_removed = left, right, area_left, area_right
            else:
                keep, removed, area_keep, area_removed = right, left, area_right, area_left
            if removed in drop:
                continue
            drop.add(removed)
            plan.append((keep, removed, round(iou, 4), round(area_keep, 6), round(area_removed, 6)))

    kept_instances = [inst for index, inst in enumerate(instances) if index not in drop]
    return plan, kept_instances


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--classes", required=True, help="目标类别 ID，逗号分隔，例如 13")
    parser.add_argument("--splits", default="train,val")
    parser.add_argument("--iou-threshold", type=float, default=0.85)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="只打印计划（默认）")
    mode.add_argument("--apply", action="store_true", help="真正修改标签文件")
    parser.add_argument("--backup-dir", default=None, help="--apply 时必填，原标签备份目录")
    parser.add_argument("--manifest", default=None, help="--apply 时必填，改动清单输出路径")
    args = parser.parse_args()

    if args.apply and (not args.backup_dir or not args.manifest):
        raise SystemExit("--apply 必须同时提供 --backup-dir 与 --manifest")

    targets = {int(item.strip()) for item in args.classes.split(",") if item.strip()}
    backup_root = Path(args.backup_dir) if args.backup_dir else None
    records: list[FixRecord] = []
    planned_files: dict[str, int] = {}   # split -> 该 split 下待改文件数
    planned_drops: dict[str, int] = {}   # split -> 该 split 下待删多边形数

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        samples = load_indexed_split(args.data_yaml, split)
        for sample in samples:
            if not sample.label_path.is_file():
                continue
            plan, kept = _plan_file(sample.label_path, targets, args.iou_threshold)
            if not plan:
                continue
            planned_files[split] = planned_files.get(split, 0) + 1
            planned_drops[split] = planned_drops.get(split, 0) + len(plan)
            print(f"[{split}] {sample.label_path}")
            for keep, removed, iou, area_keep, area_removed in plan:
                print(f"  keep #{keep} (area {area_keep:.6f})  drop #{removed} (area {area_removed:.6f})  iou={iou}")
            if not args.apply:
                continue

            sha_before = _sha256(sample.label_path)
            relative = sample.label_path.relative_to(Path(sample.label_path.anchor or "/"))
            target_backup = backup_root / split / relative
            target_backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sample.label_path, target_backup)

            lines = [format_label_line(instance) for instance in kept]
            sample.label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            sha_after = _sha256(sample.label_path)

            for keep, removed, iou, area_keep, area_removed in plan:
                records.append(
                    FixRecord(
                        label_path=str(sample.label_path),
                        split=split,
                        kept_index=keep,
                        removed_index=removed,
                        iou=iou,
                        kept_area=area_keep,
                        removed_area=area_removed,
                        sha256_before=sha_before,
                        sha256_after=sha_after,
                    )
                )

    if args.apply:
        changed_files = len({record.label_path for record in records})
        save_json(
            {
                "kind": "duplicate_label_fix",
                "data_yaml": args.data_yaml,
                "classes": sorted(targets),
                "iou_threshold": args.iou_threshold,
                "files_modified": changed_files,
                "files_changed": changed_files,
                "instances_removed": len(records),
                "per_split_removed": dict(planned_drops),
                "backup_dir": args.backup_dir,
                "records": [record.__dict__ for record in records],
            },
            Path(args.manifest),
        )
        print(f"\n改动 {changed_files} 个文件，删除 {len(records)} 个多边形（{', '.join(f'{k}={v}' for k, v in planned_drops.items())}）")
        print(f"备份: {args.backup_dir}")
        print(f"清单: {args.manifest}")
    else:
        total_files = sum(planned_files.values())
        total_drops = sum(planned_drops.values())
        print(
            f"\n[dry-run] 待改 {total_files} 个文件，删除 {total_drops} 个多边形"
            f"（{', '.join(f'{split}={planned_files.get(split, 0)}文件/{planned_drops.get(split, 0)}删' for split in planned_files) or '无'}）；加 --apply 才会执行"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
