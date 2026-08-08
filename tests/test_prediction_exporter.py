"""prediction_exporter 测试：txt 索引支持、data.yaml 类名与逐实例原始字段。"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from blade_defect.experiment import prediction_exporter as exporter_module
from blade_defect.experiment.prediction_exporter import export_validation_predictions

VALID_LINE = "0 0.1 0.1 0.2 0.1 0.2 0.2\n"


class _FakeTensor:
    def __init__(self, value: object) -> None:
        self._value = value

    def item(self) -> object:
        return self._value

    def tolist(self) -> object:
        return self._value


class _FakeBox:
    def __init__(self, cls_id: int, confidence: float, bbox: list[float]) -> None:
        self.cls = _FakeTensor(cls_id)
        self.conf = _FakeTensor(confidence)
        self.xyxy = _FakeTensor([bbox])


def _fake_result(image_path: Path) -> SimpleNamespace:
    boxes = [_FakeBox(1, 0.8765, [10.0, 20.0, 30.0, 40.0])]
    masks = SimpleNamespace(xy=[[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    return SimpleNamespace(
        path=str(image_path),
        boxes=boxes,
        masks=masks,
        orig_shape=(480, 640),
    )


class _FakePredictor:
    def __init__(self, model_path: object) -> None:
        self.model = SimpleNamespace(predict=self._predict)
        self._sources: list[object] = []

    def _predict(self, **kwargs: object):
        source = kwargs["source"]
        if isinstance(source, (list, tuple)):
            paths = [Path(item) for item in source]
        else:
            source_path = Path(source)
            paths = [source_path] if source_path.is_file() else sorted(source_path.glob("*.jpg"))
        for path in paths:
            yield _fake_result(path)


def _build_dataset(tmp_path: Path, *, txt_index: bool) -> Path:
    dataset = tmp_path / "dataset"
    image = dataset / "images" / "val" / "val_0.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8\xff")
    labels_dir = dataset / "labels" / "val"
    labels_dir.mkdir(parents=True)
    labels_dir.joinpath("val_0.txt").write_text(VALID_LINE, encoding="utf-8")
    data = tmp_path / "data.yaml"
    names = "{0: 表面腐蚀, 1: 表面裂纹}"
    if txt_index:
        dataset.joinpath("val.txt").write_text("./images/val/val_0.jpg\n", encoding="utf-8")
        data.write_text(
            f"path: {dataset.as_posix()}\ntrain: images/val\nval: val.txt\nnames: {names}\n",
            encoding="utf-8",
        )
    else:
        data.write_text(
            f"path: {dataset.as_posix()}\ntrain: images/val\nval: images/val\nnames: {names}\n",
            encoding="utf-8",
        )
    return data


@pytest.mark.parametrize("txt_index", [True, False])
def test_export_predictions_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, txt_index: bool) -> None:
    data = _build_dataset(tmp_path, txt_index=txt_index)
    monkeypatch.setattr(exporter_module, "SegmentationPredictor", _FakePredictor)
    output = export_validation_predictions(
        model_path="model.pt",
        data_yaml=data,
        output_path=tmp_path / "out" / "validation_predictions.json",
        experiment_id="exp_test",
        imgsz=960,
        device="cpu",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["experiment_id"] == "exp_test"
    assert payload["num_samples"] == 1
    sample = payload["samples"][0]
    assert sample["split"] == "val"
    assert sample["image_width"] == 640
    assert sample["image_height"] == 480
    assert sample["true_classes"] == [0]
    assert sample["true_class_names"] == ["表面腐蚀"]
    assert sample["ground_truth"][0]["polygon"] == [0.1, 0.1, 0.2, 0.1, 0.2, 0.2]
    assert sample["predicted_classes"] == [1]
    assert sample["predicted_class_names"] == ["表面裂纹"]
    prediction = sample["predictions"][0]
    assert prediction["class_name"] == "表面裂纹"
    assert prediction["confidence"] == 0.8765  # round(0.8765, 4)
    assert prediction["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert prediction["mask_polygon"] == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]


def test_export_rejects_missing_val(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data.yaml"
    data.write_text(
        f"path: {tmp_path.as_posix()}\ntrain: images/train\nnames: [defect]\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(exporter_module, "SegmentationPredictor", _FakePredictor)
    with pytest.raises(FileNotFoundError):
        export_validation_predictions(
            model_path="model.pt",
            data_yaml=data,
            output_path=tmp_path / "out.json",
            experiment_id="exp_test",
        )


def test_export_txt_index_outside_dataset_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pool = tmp_path / "blade-v2"
    image = pool / "images" / "val" / "val_0.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"\xff\xd8\xff")
    pool_label = pool / "labels" / "val"
    pool_label.mkdir(parents=True)
    pool_label.joinpath("val_0.txt").write_text(VALID_LINE, encoding="utf-8")
    release = tmp_path / "release"
    release.mkdir()
    release.joinpath("val.txt").write_text("../blade-v2/images/val/val_0.jpg\n", encoding="utf-8")
    data = release / "data.yaml"
    data.write_text(
        "path: .\ntrain: train.txt\nval: val.txt\nnames: {0: 表面腐蚀, 1: 表面裂纹}\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(exporter_module, "SegmentationPredictor", _FakePredictor)
    output = export_validation_predictions(
        model_path="model.pt",
        data_yaml=data,
        output_path=tmp_path / "out" / "validation_predictions.json",
        experiment_id="exp_test",
        imgsz=960,
        device="cpu",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["num_samples"] == 1
    sample = payload["samples"][0]
    assert sample["true_classes"] == [0]
    assert sample["ground_truth"][0]["polygon"] == [0.1, 0.1, 0.2, 0.1, 0.2, 0.2]
