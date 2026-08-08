"""train/val/test split 隔离校验：确保 test 不参与训练、验证或阈值选择。

无泄漏实验（如 blade-v3-grouped）要求三个 split 两两不相交。本模块在图片
路径与文件名 stem 两个层面检查重叠：
- 路径级重叠：同一文件被多个 split 引用（索引重复引用）；
- stem 级重叠：不同路径但同名文件，通常是同一样本被复制进多个 split。

校验只读取索引/目录清单，不加载图片内容，适用于大型数据集启动前检查。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from blade_defect.data.validation import (
    DatasetGateError,
    _abspath,
    _resolve_index_entry,
    load_dataset_config_portable,
)
from blade_defect.utils.files import IMAGE_SUFFIXES
SPLIT_FIELDS = ("train", "val", "test")


@dataclass
class SplitIsolationReport:
    """split 隔离校验结果。"""

    dataset_id: str
    split_counts: dict[str, int] = field(default_factory=dict)
    overlaps: dict[str, list[str]] = field(default_factory=dict)

    @property
    def isolated(self) -> bool:
        return not any(self.overlaps.values())


def _split_images(dataset_root: Path, split_value: Any) -> list[Path]:
    """展开 split 字段（txt 索引或目录）为图片路径列表。"""
    if split_value is None:
        return []
    entry = Path(str(split_value))
    if entry.is_file() and entry.suffix.lower() == ".txt":
        images: list[Path] = []
        for raw_line in entry.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                images.append(_resolve_index_entry(dataset_root, line, entry))
        return images
    if entry.is_dir():
        return sorted(
            path
            for path in entry.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    return []


def validate_test_split_isolation(data_yaml: str | Path) -> SplitIsolationReport:
    """校验 data.yaml 中 train/val/test 三个 split 两两不相交。

    返回 SplitIsolationReport；dataset 没有 test 字段时同样校验 train/val
    是否相交（旧全量 split 的已知泄漏形态之一）。
    """
    config = load_dataset_config_portable(data_yaml)
    dataset_root = _abspath(config["path"])
    manifest_path = dataset_root / "dataset_manifest.json"
    dataset_id = dataset_root.name
    if manifest_path.is_file():
        import json

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            if isinstance(manifest.get("dataset_id"), str):
                dataset_id = manifest["dataset_id"]
        except (OSError, json.JSONDecodeError):
            pass

    split_images: dict[str, list[Path]] = {
        split: _split_images(dataset_root, config.get(split)) for split in SPLIT_FIELDS
    }
    report = SplitIsolationReport(
        dataset_id=dataset_id,
        split_counts={split: len(images) for split, images in split_images.items()},
    )
    splits = [split for split in SPLIT_FIELDS if split_images[split]]
    for position, left in enumerate(splits):
        for right in splits[position + 1:]:
            left_paths = {str(path).casefold() for path in split_images[left]}
            right_paths = {str(path).casefold() for path in split_images[right]}
            path_overlap = sorted(left_paths & right_paths)
            left_stems = {path.stem.casefold() for path in split_images[left]}
            right_stems = {path.stem.casefold() for path in split_images[right]}
            stem_overlap = sorted(left_stems & right_stems)
            key = f"{left}_vs_{right}"
            report.overlaps[key] = []
            if path_overlap:
                report.overlaps[key].extend(f"path:{item}" for item in path_overlap)
            if stem_overlap:
                report.overlaps[key].extend(f"stem:{item}" for item in stem_overlap)
    return report


def assert_test_split_isolation(data_yaml: str | Path) -> SplitIsolationReport:
    """校验并在发现任何 split 重叠时抛出 DatasetGateError。"""
    report = validate_test_split_isolation(data_yaml)
    if not report.isolated:
        details = "; ".join(
            f"{pair}: {len(items)} 处重叠（如 {items[0]}）"
            for pair, items in report.overlaps.items()
            if items
        )
        raise DatasetGateError(
            f"数据集 {report.dataset_id} 存在 split 泄漏：{details}。"
            "test 不得参与训练、验证、best epoch 或阈值选择。"
        )
    return report


__all__ = [
    "SplitIsolationReport",
    "assert_test_split_isolation",
    "validate_test_split_isolation",
]
