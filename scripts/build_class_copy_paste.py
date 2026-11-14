#!/usr/bin/env python
"""按类定向 copy-paste 的离线合成入口（方案 B 的前置步骤）。

用法：
  # 先干跑，看会合成多少张、每类多少条
  python scripts/build_class_copy_paste.py --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
      --out-dir /data/datasets/derived/blade-v3-grouped-202608-cp10_13_14 --classes 10,13,14 --copies 3 --dry-run

  # 确认后真正生成
  python scripts/build_class_copy_paste.py --data-yaml ... --out-dir ... --classes 10,13,14 --copies 3

产物（全部落在 --out-dir）：
  images/train/*.jpg        合成图
  labels/train/*.txt        对应的 YOLO-seg 标签（目标图原有标注 + 贴入的实例）
  train.txt                 原 train 索引 + 合成样本
  data.yaml                 可直接喂给训练入口的派生数据配置（val 指回原 val，不泄漏）
  copy_paste_manifest.json  逐条审计记录（seed/源图/目标图/类别/缩放/落点）

红线：只读原数据集，绝不修改原目录；合成样本只进 train，val/test 不动。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from blade_defect.data.class_copy_paste import SynthesizeConfig, synthesize


def _parse_classes(raw: str) -> list[int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--classes 不能为空，例如 10,13,14")
    return [int(item) for item in values]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-yaml", required=True, help="原数据集 data.yaml")
    parser.add_argument("--out-dir", required=True, help="派生数据集输出目录")
    parser.add_argument("--classes", type=_parse_classes, required=True, help="目标类别 ID，逗号分隔")
    parser.add_argument("--copies", type=int, default=3, help="每个实例复制次数（默认 3）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scale-min", type=float, default=0.8)
    parser.add_argument("--scale-max", type=float, default=1.2)
    parser.add_argument("--max-synthetic", type=int, default=None, help="硬上限，保护磁盘")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dry-run", action="store_true", help="只统计源实例数，不写文件")
    args = parser.parse_args()

    if args.dry_run:
        from blade_defect.data.class_copy_paste import _target_instances  # noqa: PLC0415
        from blade_defect.data.indexed_splits import load_indexed_split

        samples = load_indexed_split(args.data_yaml, args.split)
        picked = _target_instances(samples, set(args.classes))
        per_class: dict[int, int] = {}
        for _, instance in picked:
            per_class[instance.class_id] = per_class.get(instance.class_id, 0) + 1
        print(json.dumps(
            {
                "split": args.split,
                "train_samples": len(samples),
                "source_instances": len(picked),
                "per_class_source": {str(k): v for k, v in sorted(per_class.items())},
                "planned_synthetic": len(picked) * args.copies,
                "out_dir": args.out_dir,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return 0

    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"{out_dir} 已存在且非空，请换目录或先删除，避免混入上一轮合成样本")

    manifest = synthesize(
        SynthesizeConfig(
            data_yaml=args.data_yaml,
            out_dir=out_dir,
            target_classes=args.classes,
            copies_per_instance=args.copies,
            seed=args.seed,
            scale_range=(args.scale_min, args.scale_max),
            max_synthetic=args.max_synthetic,
            split=args.split,
        )
    )
    summary = {
        key: manifest[key]
        for key in (
            "source_instances",
            "synthetic_images",
            "per_class_synthetic",
            "skipped_total",
            "derived_data_yaml",
            "derived_train_index",
            "seed",
            "target_classes",
        )
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"manifest: {out_dir / 'copy_paste_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
