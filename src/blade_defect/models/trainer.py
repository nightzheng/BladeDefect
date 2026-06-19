"""Ultralytics YOLO segmentation training wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blade_defect.utils.files import load_yaml


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("请先安装 ultralytics：pip install -r requirements.txt") from exc
    return YOLO


class SegmentationTrainer:
    def __init__(self, model: str = "yolo11n-seg.pt") -> None:
        self.model_source = model
        self.model = _load_yolo()(model)

    @classmethod
    def from_config(cls, config_path: str | Path) -> tuple["SegmentationTrainer", dict[str, Any]]:
        config = load_yaml(config_path)
        model = config.pop("model", "yolo11n-seg.pt")
        return cls(model), config

    def train(self, **kwargs: Any) -> Any:
        """Train a segmentation model. Keyword arguments map to YOLO.train."""
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        return self.model.train(task="segment", **kwargs)

    def validate(self, data: str | Path, device: str | int | None = 0, **kwargs: Any) -> Any:
        if device is not None:
            kwargs["device"] = device
        return self.model.val(data=str(data), task="segment", **kwargs)

    def export(self, format: str = "onnx", **kwargs: Any) -> Any:
        return self.model.export(format=format, **kwargs)
