"""Ultralytics YOLO segmentation training wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blade_defect.utils.device import resolve_device
from blade_defect.utils.files import load_project_config, resolved_data_yaml
from blade_defect.utils.paths import resolve_model_reference, resolve_path


def _load_yolo():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("请先安装 ultralytics：python -m pip install -r requirements.txt") from exc
    return YOLO


class SegmentationTrainer:
    def __init__(self, model: str | Path = "yolo11n-seg.pt") -> None:
        self.model_source = resolve_model_reference(model)
        self.model = _load_yolo()(str(self.model_source))

    @classmethod
    def from_config(cls, config_path: str | Path) -> tuple["SegmentationTrainer", dict[str, Any]]:
        config, project_root = load_project_config(config_path)
        model = config.pop("model", "yolo11n-seg.pt")
        return cls(resolve_model_reference(model, project_root)), config

    def train(self, **kwargs: Any) -> Any:
        """Train a segmentation model. Keyword arguments map to YOLO.train."""
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        kwargs["device"] = resolve_device(kwargs.get("device", "auto"))
        if isinstance(kwargs.get("project"), Path):
            kwargs["project"] = str(kwargs["project"])
        data = kwargs.pop("data", None)
        if data is None:
            return self.model.train(task="segment", **kwargs)
        with resolved_data_yaml(data) as normalized_data:
            return self.model.train(data=normalized_data, task="segment", **kwargs)

    def validate(self, data: str | Path, device: str | int | None = "auto", **kwargs: Any) -> Any:
        kwargs["device"] = resolve_device(device)
        with resolved_data_yaml(resolve_path(data)) as normalized_data:
            return self.model.val(data=normalized_data, task="segment", **kwargs)

    def export(self, format: str = "onnx", **kwargs: Any) -> Any:
        return self.model.export(format=format, **kwargs)
