"""Abstract RGB-T registration and fusion interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FusionResult:
    image: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


class ImageRegistration(ABC):
    @abstractmethod
    def register(self, rgb: np.ndarray, thermal: np.ndarray | None = None) -> FusionResult:
        """Align thermal data to RGB coordinates."""


class ImageFusion(ABC):
    @abstractmethod
    def fuse(self, rgb: np.ndarray, thermal: np.ndarray | None = None) -> FusionResult:
        """Produce a model-ready image or tensor representation."""
