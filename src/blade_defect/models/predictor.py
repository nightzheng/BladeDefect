"""Ultralytics YOLO segmentation inference wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .trainer import _load_yolo


class SegmentationPredictor:
    def __init__(self, weights: str | Path) -> None:
        self.weights = str(weights)
        self.model = _load_yolo()(self.weights)

    def predict(
        self,
        source: Any,
        conf: float = 0.25,
        iou: float = 0.7,
        imgsz: int = 640,
        device: str | int | None = 0,
        save: bool = False,
        **kwargs: Any,
    ) -> list[Any]:
        options = dict(source=source, conf=conf, iou=iou, imgsz=imgsz, save=save, task="segment", **kwargs)
        if device is not None:
            options["device"] = device
        return self.model.predict(**options)
