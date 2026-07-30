"""训练前数据集门禁：同时支持目录型与 train.txt/val.txt 清单型数据集。

门禁在创建模型前运行，覆盖：
- txt 索引（train.txt/val.txt）解析，兼容相对（含 ``./`` 前缀）与绝对图片路径；
- 索引中缺失图片、重复索引；
- 图片—标签配对（缺失标签、孤立标签）；
- 空标签负样本（合法，仅计数）；
- polygon 标注 strict 校验与类别分布统计。

门禁失败时抛出 :class:`DatasetGateError`，错误信息包含数据版本
（dataset_id）、索引文件路径和具体错误原因。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from blade_defect.utils.files import IMAGE_SUFFIXES, load_yaml
from .label_check import LabelIssue, _inspect_seg_line

MAX_REPORTED_SAMPLES = 20


class DatasetGateError(RuntimeError):
    """数据集未通过训练前 strict 门禁时抛出。"""


@dataclass
class SplitGateReport:
    """单个 split 的门禁结果。"""

    split: str
    source: str
    index_file: str | None = None
    images: int = 0
    labels: int = 0
    instances: int = 0
    negative_samples: int = 0
    duplicates: list[str] = field(default_factory=list)
    missing_images: list[str] = field(default_factory=list)
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    issues: list[LabelIssue] = field(default_factory=list)
    class_counts: dict[int, int] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not (
            self.duplicates
            or self.missing_images
            or self.missing_labels
            or self.orphan_labels
            or self.issues
        )

    def failure_summary(self) -> list[str]:
        """生成 ``missing_images=3 (line 12: x.jpg, ...)`` 形式的失败摘要。"""
        summary: list[str] = []
        prefix = f"{self.split}"
        if self.index_file is not None:
            prefix += f"(index={self.index_file})"
        if self.missing_images:
            summary.append(f"{prefix}: missing_images={len(self.missing_images)} ({_sample(self.missing_images)})")
        if self.missing_labels:
            summary.append(f"{prefix}: missing_labels={len(self.missing_labels)} ({_sample(self.missing_labels)})")
        if self.orphan_labels:
            summary.append(f"{prefix}: orphan_labels={len(self.orphan_labels)} ({_sample(self.orphan_labels)})")
        if self.duplicates:
            summary.append(f"{prefix}: duplicate_index={len(self.duplicates)} ({_sample(self.duplicates)})")
        if self.issues:
            first = self.issues[0]
            summary.append(
                f"{prefix}: hard_error={len(self.issues)} "
                f"(first: {first.file} line {first.line} {first.error_type}: {first.message})"
            )
        return summary

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["valid"] = self.valid
        result["class_counts"] = {str(key): value for key, value in sorted(self.class_counts.items())}
        return result


@dataclass
class DatasetGateReport:
    """整个数据集的门禁结果。"""

    dataset_id: str
    data_yaml: str
    dataset_root: str
    num_classes: int | None
    splits: dict[str, SplitGateReport]

    @property
    def valid(self) -> bool:
        return all(report.valid for report in self.splits.values())

    def failure_summary(self) -> list[str]:
        lines: list[str] = []
        for report in self.splits.values():
            lines.extend(report.failure_summary())
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "data_yaml": self.data_yaml,
            "dataset_root": self.dataset_root,
            "num_classes": self.num_classes,
            "valid": self.valid,
            "splits": {split: report.to_dict() for split, report in self.splits.items()},
        }


def _abspath(path: str | Path, base: str | Path | None = None) -> Path:
    """绝对化并规范化路径，但不解析符号链接/NTFS junction。

    junction 指向的目标目录可能属于另一个数据版本（例如 6 类数据集的
    ``images/train`` 指向 15 类全量数据），链接解析会让图片跑到数据集根之外，
    导致标签错配；门禁必须按 data.yaml 书写的路径结构工作。
    """
    candidate = Path(os.path.expandvars(str(path).strip()))
    if not candidate.is_absolute():
        root = Path(os.path.expandvars(str(base))) if base is not None else Path.cwd()
        candidate = root / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def load_dataset_config_portable(data_yaml: str | Path) -> dict[str, Any]:
    """加载 data.yaml 并以不解析链接的方式绝对化路径字段。"""
    config_path = _abspath(data_yaml)
    config = load_yaml(config_path)
    dataset_root = _abspath(config.get("path", "."), config_path.parent)
    result: dict[str, Any] = dict(config)
    result["path"] = dataset_root
    for field_name in ("train", "val", "test"):
        if result.get(field_name) is not None:
            result[field_name] = _abspath(result[field_name], dataset_root)
    return result


def _iter_images(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _iter_labels(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".txt")


def _sample(items: list[str], limit: int = 5) -> str:
    shown = items[:limit]
    suffix = f", ... (+{len(items) - limit})" if len(items) > limit else ""
    return "; ".join(shown) + suffix


def _dataset_id(dataset_root: Path) -> str:
    """优先使用 dataset_manifest.json 中的 dataset_id，否则回退到目录名。"""
    manifest_path = dataset_root / "dataset_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}
        dataset_id = manifest.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id:
            return dataset_id
    return dataset_root.name


def _resolve_index_entry(dataset_root: Path, raw_line: str) -> Path:
    """解析 txt 索引行，兼容 ``./`` 前缀相对路径与绝对路径。"""
    candidate = Path(raw_line)
    if candidate.is_absolute():
        return Path(os.path.normpath(os.fspath(candidate)))
    return _abspath(candidate, dataset_root)


def _label_path_for(dataset_root: Path, image_path: Path, split: str) -> Path:
    """按 images→labels 目录约定推导图片对应的标签路径。"""
    try:
        relative = image_path.relative_to(dataset_root)
    except ValueError:
        relative = None
    if relative is not None:
        parts = relative.parts
        if parts and parts[0].lower() == "images":
            return dataset_root / "labels" / Path(*parts[1:]).with_suffix(".txt")
        return dataset_root / "labels" / split / relative.with_suffix(".txt").name
    parts = image_path.parts
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        position = lowered.index("images")
        return Path(*parts[:position], "labels", *parts[position + 1:]).with_suffix(".txt")
    return dataset_root / "labels" / split / f"{image_path.stem}.txt"


def _check_label_file(
    label_path: Path,
    display_name: str,
    num_classes: int | None,
    report: SplitGateReport,
) -> None:
    """strict 校验单个标签文件，空文件按负样本计数，合法实例计入类别分布。"""
    report.labels += 1
    content = label_path.read_text(encoding="utf-8")
    if not content.strip():
        report.negative_samples += 1
        return
    for line_number, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        result = _inspect_seg_line(line, num_classes, mode="strict")
        if result.error:
            report.issues.append(
                LabelIssue(display_name, line_number, "HARD_ERROR", result.error.error_type, result.error.message)
            )
            continue
        report.instances += 1
        class_id = int(float(line.split()[0]))
        report.class_counts[class_id] = report.class_counts.get(class_id, 0) + 1


def _scan_orphan_labels(labels_dir: Path, expected: set[str], report: SplitGateReport) -> None:
    """找出有标签但不在图片集合内的孤立标签。"""
    if not labels_dir.is_dir():
        return
    for label_path in _iter_labels(labels_dir):
        key = label_path.relative_to(labels_dir).with_suffix("").as_posix().casefold()
        if key not in expected:
            report.orphan_labels.append(
                label_path.relative_to(labels_dir).as_posix()
            )


def _validate_directory_split(
    dataset_root: Path,
    images_dir: Path,
    split: str,
    num_classes: int | None,
) -> SplitGateReport:
    report = SplitGateReport(split=split, source="directory")
    labels_dir = dataset_root / "labels" / split
    if not labels_dir.is_dir():
        report.missing_labels.append(f"labels directory not found: {labels_dir}")
        return report
    images = _iter_images(images_dir)
    report.images = len(images)
    expected_keys: set[str] = set()
    for image_path in images:
        label_path = _label_path_for(dataset_root, image_path, split)
        try:
            key = label_path.relative_to(labels_dir).with_suffix("").as_posix().casefold()
            expected_keys.add(key)
        except ValueError:
            pass
        if not label_path.is_file():
            report.missing_labels.append(image_path.relative_to(images_dir).as_posix())
            continue
        _check_label_file(label_path, label_path.name, num_classes, report)
    _scan_orphan_labels(labels_dir, expected_keys, report)
    return report


def _validate_txt_index_split(
    dataset_root: Path,
    index_file: Path,
    split: str,
    num_classes: int | None,
) -> SplitGateReport:
    report = SplitGateReport(split=split, source="txt_index", index_file=str(index_file))
    labels_dir = dataset_root / "labels" / split
    raw_lines = index_file.read_text(encoding="utf-8").splitlines()
    seen: dict[str, int] = {}
    expected_keys: set[str] = set()
    for line_number, raw_line in enumerate(raw_lines, 1):
        entry = raw_line.strip()
        if not entry:
            continue
        image_path = _resolve_index_entry(dataset_root, entry)
        identity = str(image_path).casefold()
        if identity in seen:
            report.duplicates.append(f"line {line_number}: {entry} (first at line {seen[identity]})")
            continue
        seen[identity] = line_number
        report.images += 1
        if image_path.suffix.lower() not in IMAGE_SUFFIXES or not image_path.is_file():
            report.missing_images.append(f"line {line_number}: {entry}")
            continue
        label_path = _label_path_for(dataset_root, image_path, split)
        try:
            key = label_path.relative_to(labels_dir).with_suffix("").as_posix().casefold()
            expected_keys.add(key)
        except ValueError:
            pass
        if not label_path.is_file():
            report.missing_labels.append(f"line {line_number}: {entry}")
            continue
        _check_label_file(label_path, label_path.name, num_classes, report)
    _scan_orphan_labels(labels_dir, expected_keys, report)
    return report


def validate_dataset_gate(
    data_yaml: str | Path,
    num_classes: int | None = None,
) -> DatasetGateReport:
    """对 data.yaml 指向的数据集执行训练前 strict 门禁。

    ``train``/``val`` 字段既可以是图片目录，也可以是 ``train.txt``/``val.txt``
    清单文件；清单行支持相对数据集根（含 ``./`` 前缀）或绝对图片路径。
    空标签文件视为合法负样本。门禁失败抛出 :class:`DatasetGateError`，
    错误信息包含 dataset_id、索引文件与具体原因。
    """
    config = load_dataset_config_portable(data_yaml)
    dataset_root = Path(config["path"])
    names = config.get("names", {})
    if num_classes is None and isinstance(names, (dict, list)):
        num_classes = len(names)
    dataset_id = _dataset_id(dataset_root)

    splits: dict[str, SplitGateReport] = {}
    for split in ("train", "val"):
        entry = config.get(split)
        entry_path = Path(entry) if entry is not None else None
        if entry_path is not None and entry_path.is_file() and entry_path.suffix.lower() == ".txt":
            splits[split] = _validate_txt_index_split(dataset_root, entry_path, split, num_classes)
        elif entry_path is not None and entry_path.is_dir():
            splits[split] = _validate_directory_split(dataset_root, entry_path, split, num_classes)
        else:
            report = SplitGateReport(split=split, source="unknown")
            report.missing_images.append(f"{split} entry not found or unsupported: {entry}")
            splits[split] = report

    gate_report = DatasetGateReport(
        dataset_id=dataset_id,
        data_yaml=str(data_yaml),
        dataset_root=str(dataset_root),
        num_classes=num_classes,
        splits=splits,
    )
    if not gate_report.valid:
        raise DatasetGateError(
            f"Dataset validation failed [dataset_id={dataset_id}, data_yaml={data_yaml}]; "
            "run-all stopped before training: " + "; ".join(gate_report.failure_summary())
        )
    return gate_report


__all__ = [
    "DatasetGateError",
    "DatasetGateReport",
    "SplitGateReport",
    "load_dataset_config_portable",
    "validate_dataset_gate",
]
