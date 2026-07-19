"""Visualization helpers for segmentation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def result_to_rgb(result: Any) -> np.ndarray:
    """Convert one Ultralytics result to an RGB image with masks and boxes."""
    plotted_bgr = result.plot()
    return cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)


def save_result(result: Any, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), result.plot())
    return output
