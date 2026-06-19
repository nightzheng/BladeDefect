"""Ultralytics model wrappers."""

from .predictor import SegmentationPredictor
from .trainer import SegmentationTrainer

__all__ = ["SegmentationPredictor", "SegmentationTrainer"]
