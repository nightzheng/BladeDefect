"""为 6 类采样数据集生成 train.txt/val.txt 索引并冻结 manifest。

6 类数据集的标签来自 blade-v2-sampled-4987 采样版的 remap（3,991/996），
但 images 目录是指向全量 blade-v2 的 junction，直接按目录训练会混入
39k 无 6 类标签的图片（上周 0.316 无效结果的根因）。本脚本按标签清单
回写 txt 索引，使 data.yaml 精确指向有 6 类标签的样本子集。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _sha256_paths(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_index(dataset_root: Path, manifest_path: Path | None = None) -> dict:
    dataset_root = dataset_root.resolve()
    names = yaml.safe_load(dataset_root.joinpath("data.yaml").read_text(encoding="utf-8"))["names"]
    num_classes = len(names)
    summary: dict = {"dataset_root": str(dataset_root), "splits": {}}
    all_labels: list[Path] = []
    for split in ("train", "val"):
        labels_dir = dataset_root / "labels" / split
        images_dir = dataset_root / "images" / split
        labels = sorted(labels_dir.glob("*.txt"))
        image_index = {
            path.stem: path
            for path in images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        lines: list[str] = []
        missing: list[str] = []
        for label_path in labels:
            image = image_index.get(label_path.stem)
            if image is None:
                missing.append(label_path.name)
                continue
            relative = image.relative_to(dataset_root).as_posix()
            lines.append(f"./{relative}")
        if missing:
            raise SystemExit(f"{split}: {len(missing)} 个标签找不到对应图片：{missing[:5]}")
        index_path = dataset_root / f"{split}.txt"
        index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        summary["splits"][split] = {"samples": len(lines), "index": str(index_path)}
        all_labels.extend(labels)

    data_yaml = dataset_root / "data.yaml"
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    payload["train"] = "train.txt"
    payload["val"] = "val.txt"
    data_yaml.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    instances = 0
    class_counts: dict[int, int] = {}
    for label_path in all_labels:
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id = int(float(line.split()[0]))
            if class_id >= num_classes:
                raise SystemExit(f"标签 {label_path.name} 含越界 class_id={class_id}")
            instances += 1
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

    manifest = {
        "dataset_id": "blade-v2-sampled-4987-6class",
        "parent_dataset_id": "blade-v2-sampled-4987",
        "purpose": "hier_coarse_smoke",
        "label_level": "coarse6",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source": {
            "labels": "labels remapped from blade-v2-sampled-4987 via configs/class_hierarchy.yaml",
            "images": "junction subsets indexed by train.txt/val.txt",
        },
        "random_seed": 42,
        "samples": {split: info["samples"] for split, info in summary["splits"].items()},
        "instances": instances,
        "class_counts": {str(key): class_counts[key] for key in sorted(class_counts)},
        "sample_lists_sha256": _sha256_paths(
            [dataset_root / "train.txt", dataset_root / "val.txt"]
        ),
        "labels_sha256": _sha256_files(all_labels),
        "generation_command": "python scripts/build_coarse6_index.py --dataset datasets/blade-v2-6class",
    }
    manifest_output = manifest_path or dataset_root / "dataset_manifest.json"
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary["instances"] = instances
    summary["class_counts"] = manifest["class_counts"]
    summary["manifest"] = str(manifest_output)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="datasets/blade-v2-6class", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_index(args.dataset), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
