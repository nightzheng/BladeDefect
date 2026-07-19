"""Ultralytics YOLO segmentation inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from blade_defect.utils.device import resolve_device
from blade_defect.utils.paths import resolve_model_reference

from .trainer import _load_yolo


class SegmentationPredictor:
    def __init__(self, weights: str | Path) -> None:
        self.weights = str(resolve_model_reference(weights))
        self.model = _load_yolo()(self.weights)

    def predict(
        self,
        source: Any,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | int | None = "auto",
        save: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        if isinstance(source, Path):
            source = str(source)
        options = dict(source=source, conf=conf, iou=iou, imgsz=imgsz, save=save, task="segment", **kwargs)
        options["device"] = resolve_device(device)
        return self.model.predict(**options)
