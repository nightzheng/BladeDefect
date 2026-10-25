"""按类定向 copy-paste 合成的测试（纯 numpy + Pillow，不依赖 torch/ultralytics）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blade_defect.data.class_copy_paste import (
    LabelInstance,
    SynthesizeConfig,
    format_label_line,
    parse_label_line,
    polygon_area,
    polygon_bbox,
    read_label_file,
    synthesize,
    write_label_file,
)


def _write_image(path: Path, size: tuple[int, int], seed: int) -> None:
    from PIL import Image

    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path, quality=95)


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
    """构造最小数据集：4 张 train（2 张含类 1，2 张只含类 0）+ 2 张 val。"""
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    val_images = root / "images" / "val"
    val_labels = root / "labels" / "val"
    train_lines: list[str] = []
    for index, class_id in enumerate([1, 1, 0, 0]):
        image_path = images / f"img{index}.jpg"
        _write_image(image_path, (200, 160), seed=index)
        write_label_file(labels / f"img{index}.txt", [LabelInstance(class_id, _square(0.4, 0.5, 0.12))])
        train_lines.append(str(image_path))
    val_lines: list[str] = []
    for index in range(2):
        image_path = val_images / f"val{index}.jpg"
        _write_image(image_path, (200, 160), seed=100 + index)
        write_label_file(val_labels / f"val{index}.txt", [LabelInstance(1, _square(0.5, 0.5, 0.1))])
        val_lines.append(str(image_path))

    train_index = root / "train.txt"
    train_index.write_text("\n".join(train_lines) + "\n", encoding="utf-8")
    val_index = root / "val.txt"
    val_index.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root.as_posix()}",
                "train: train.txt",
                "val: val.txt",
                "names:",
                "  0: 其他类",
                "  1: 目标类",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return data_yaml


def test_parse_and_format_roundtrip() -> None:
    instance = parse_label_line("3 0.1 0.2 0.3 0.4 0.5 0.6")
    assert instance is not None
    assert instance.class_id == 3
    assert instance.polygon.shape == (3, 2)
    line = format_label_line(instance)
    assert line.startswith("3 ")
    assert parse_label_line(line).polygon.shape == (3, 2)


def test_parse_rejects_odd_coordinates() -> None:
    with pytest.raises(ValueError):
        parse_label_line("1 0.1 0.2 0.3")


def test_parse_skips_blank_line() -> None:
    assert parse_label_line("   ") is None


def test_polygon_geometry() -> None:
    square = _square(0.5, 0.5, 0.25)
    assert polygon_area(square) == pytest.approx(0.25, rel=1e-9)
    assert polygon_bbox(square) == (pytest.approx(0.25), pytest.approx(0.25), pytest.approx(0.75), pytest.approx(0.75))


def test_synthesize_writes_auditable_outputs(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data_yaml = _build_dataset(root)
    out_dir = tmp_path / "derived"

    manifest = synthesize(
        SynthesizeConfig(
            data_yaml=data_yaml,
            out_dir=out_dir,
            target_classes=[1],
            copies_per_instance=2,
            seed=7,
            max_synthetic=10,
        )
    )

    # 2 个目标实例 × 2 次 = 最多 4 张（目的地与源相同会被跳过，所以用 >= 断言）
    assert 2 <= manifest["synthetic_images"] <= 4
    assert manifest["target_classes"] == [1]
    assert manifest["seed"] == 7
    assert sum(manifest["per_class_synthetic"].values()) == manifest["synthetic_images"]

    assert (out_dir / "data.yaml").is_file()
    assert (out_dir / "train.txt").is_file()
    assert (out_dir / "copy_paste_manifest.json").is_file()
    saved = json.loads((out_dir / "copy_paste_manifest.json").read_text(encoding="utf-8"))
    assert saved["synthetic_images"] == manifest["synthetic_images"]

    # 每张合成图都有图、有标签、标签里至少一个目标类实例，且坐标合法
    records = manifest["records"]
    assert records
    for record in records:
        image_path = Path(record["out_image"])
        assert image_path.is_file()
        label_path = out_dir / "labels" / "train" / f"{image_path.stem}.txt"
        instances = read_label_file(label_path)
        assert any(instance.class_id == 1 for instance in instances)
        for instance in instances:
            assert instance.polygon.min() >= 0.0
            assert instance.polygon.max() <= 1.0
        assert record["class_id"] == 1

    # train 索引 = 原始 4 条 + 合成条数
    lines = [line for line in (out_dir / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 4 + manifest["synthetic_images"]


def test_synthesize_does_not_touch_original_dataset(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data_yaml = _build_dataset(root)
    original_train = (root / "train.txt").read_text(encoding="utf-8")
    original_label = read_label_file(root / "labels" / "train" / "img0.txt")

    synthesize(
        SynthesizeConfig(
            data_yaml=data_yaml,
            out_dir=tmp_path / "derived",
            target_classes=[1],
            copies_per_instance=1,
            seed=3,
            max_synthetic=2,
        )
    )

    assert (root / "train.txt").read_text(encoding="utf-8") == original_train
    assert len(read_label_file(root / "labels" / "train" / "img0.txt")) == len(original_label)


def test_derived_yaml_keeps_val_on_original_dataset(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data_yaml = _build_dataset(root)
    out_dir = tmp_path / "derived"
    synthesize(
        SynthesizeConfig(
            data_yaml=data_yaml,
            out_dir=out_dir,
            target_classes=[1],
            copies_per_instance=1,
            seed=5,
            max_synthetic=2,
        )
    )
    text = (out_dir / "data.yaml").read_text(encoding="utf-8")
    val_line = next(line for line in text.splitlines() if line.startswith("val:"))
    val_path = Path(val_line.split(":", 1)[1].strip())
    # val 必须指回原数据集，绝不能指向派生目录（否则验证集被合成样本污染）
    assert val_path == (root / "val.txt")
    assert out_dir.resolve() not in val_path.resolve().parents


def test_missing_target_class_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    data_yaml = _build_dataset(root)
    with pytest.raises(ValueError):
        synthesize(
            SynthesizeConfig(
                data_yaml=data_yaml,
                out_dir=tmp_path / "derived",
                target_classes=[],
                copies_per_instance=1,
            )
        )
