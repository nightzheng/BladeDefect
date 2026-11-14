"""小样本类标注审计的测试（纯 numpy + Pillow）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from blade_defect.data.class_copy_paste import LabelInstance, format_label_line, write_label_file
from blade_defect.data.label_audit import audit_split, summarize
from blade_defect.data.indexed_splits import load_indexed_split


def _write_image(path: Path, size: tuple[int, int]) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def _square(cx: float, cy: float, half: float) -> np.ndarray:
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )


def _build_dataset(root: Path) -> Path:
    images = root / "images" / "train"
    labels = root / "labels" / "train"

    # img0：正常实例 + 极小实例 + 越界实例 + 退化实例
    # 注意：越界/退化标注必须直接写原始文本——write_label_file 会把坐标裁剪到 [0,1]，
    # 走它写出来的标签就永远测不到 OUT_OF_BOUNDS。
    _write_image(images / "img0.jpg", (100, 100))
    raw_lines = [
        format_label_line(LabelInstance(10, _square(0.5, 0.5, 0.2))),   # 正常
        format_label_line(LabelInstance(10, _square(0.5, 0.5, 0.01))),  # TINY（约 2x2 px）
        "10 0.88 0.88 1.08 0.88 1.08 1.08 0.88 1.08",                    # OUT_OF_BOUNDS
        "10 0.1 0.1 0.2 0.2 0.3 0.3",                                    # ZERO_AREA
    ]
    (labels).mkdir(parents=True, exist_ok=True)
    (labels / "img0.txt").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    # img1：同类重复实例 + 易混类共现（类 10 与类 7）
    _write_image(images / "img1.jpg", (100, 100))
    write_label_file(
        labels / "img1.txt",
        [
            LabelInstance(10, _square(0.5, 0.5, 0.2)),
            LabelInstance(10, _square(0.5, 0.5, 0.2)),  # DUPLICATE
            LabelInstance(7, _square(0.2, 0.2, 0.1)),   # CONFUSABLE
        ],
    )

    # img2：只有别类，不该出现在审计里
    _write_image(images / "img2.jpg", (100, 100))
    write_label_file(labels / "img2.txt", [LabelInstance(0, _square(0.5, 0.5, 0.2))])

    # img3：有图无标签
    _write_image(images / "img3.jpg", (100, 100))

    index = root / "train.txt"
    index.write_text(
        "\n".join(str(images / f"img{index}.jpg") for index in range(4)) + "\n",
        encoding="utf-8",
    )
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(["path: " + root.as_posix(), "train: train.txt", "val: train.txt", "names:", "  0: 其他", "  7: 表面油污", "  10: 维修痕迹"])
        + "\n",
        encoding="utf-8",
    )
    return data_yaml


def test_audit_flags_each_defect_class(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data_yaml = _build_dataset(root)
    samples = load_indexed_split(data_yaml, "train")
    rows = audit_split(samples, [10], {10: "维修痕迹", 7: "表面油污"}, split="train")

    flags = [set(row.flags) for row in rows]
    assert any("TINY" in item for item in flags)
    assert any("OUT_OF_BOUNDS" in item for item in flags)
    assert any("ZERO_AREA" in item for item in flags)
    assert any("DUPLICATE" in item for item in flags)
    assert any("CONFUSABLE" in item for item in flags)
    assert any("NO_LABEL_FILE" in item for item in flags)


def test_audit_only_returns_target_classes(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data_yaml = _build_dataset(root)
    samples = load_indexed_split(data_yaml, "train")
    rows = audit_split(samples, [10], {10: "维修痕迹"}, split="train")

    instance_rows = [row for row in rows if row.class_id != -1]
    assert instance_rows
    assert {row.class_id for row in instance_rows} == {10}


def test_audit_sorted_by_priority_desc(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data_yaml = _build_dataset(root)
    samples = load_indexed_split(data_yaml, "train")
    rows = audit_split(samples, [10], {10: "维修痕迹"}, split="train")

    priorities = [row.priority for row in rows]
    assert priorities == sorted(priorities, reverse=True)
    # 最严重的（零面积/越界）应排在前面
    assert rows[0].priority >= 4.0


def test_summary_counts_by_class(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data_yaml = _build_dataset(root)
    samples = load_indexed_split(data_yaml, "train")
    rows = audit_split(samples, [10], {10: "维修痕迹"}, split="train")
    summary = summarize(rows)

    assert summary["total_rows"] == len(rows)
    assert summary["flagged_rows"] == sum(1 for row in rows if row.flags)
    bucket = summary["by_class"]["10"]
    assert bucket["instances"] == 6  # img0 四个 + img1 两个
    assert bucket["flagged"] >= 4


def test_review_columns_are_empty_for_human(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data_yaml = _build_dataset(root)
    samples = load_indexed_split(data_yaml, "train")
    rows = audit_split(samples, [10], {10: "维修痕迹"}, split="train")
    assert all(row.review_verdict == "" and row.reviewer_note == "" for row in rows)
