"""No-op implementations that keep RGB-only workflows independent of thermal data."""

from __future__ import annotations

import numpy as np

from .base import FusionResult, ImageFusion, ImageRegistration


class IdentityRegistration(ImageRegistration):
    def register(self, rgb: np.ndarray, thermal: np.ndarray | None = None) -> FusionResult:
        return FusionResult(rgb.copy(), {"method": "identity", "thermal_used": False})


class IdentityFusion(ImageFusion):
    def fuse(self, rgb: np.ndarray, thermal: np.ndarray | None = None) -> FusionResult:
        return FusionResult(rgb.copy(), {"method": "rgb_only", "thermal_used": False})
