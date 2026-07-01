"""Experiment configuration types and defaults."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Immutable parameters for one reproducible segmentation baseline."""
    name: str
    model: str
    imgsz: int = 640
    epochs: int = 50
    batch: int = 8
    seed: int = 42

    def training_kwargs(self, data: str | Path, runs_dir: str | Path,
                        device: str | int | None = "auto") -> dict[str, Any]:
        """Translate the experiment definition to existing trainer arguments."""
        return {"data": data, "imgsz": self.imgsz, "epochs": self.epochs,
                "batch": self.batch, "seed": self.seed, "project": runs_dir,
                "name": self.name, "device": device, "exist_ok": True}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable experiment description."""
        return asdict(self)
