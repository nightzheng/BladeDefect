"""实验配置类型与默认值。"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """单个可复现分割 baseline 的不可变参数。"""
    name: str
    model: str
    imgsz: int = 640
    epochs: int = 50
    batch: int = 8
    seed: int = 42

    def training_kwargs(self, data: str | Path, runs_dir: str | Path,
                        device: str | int | None = "auto") -> dict[str, Any]:
        """将实验定义转换为现有训练器参数。"""
        return {"data": data, "imgsz": self.imgsz, "epochs": self.epochs,
                "batch": self.batch, "seed": self.seed, "project": runs_dir,
                "name": self.name, "device": device, "exist_ok": True}

    def to_dict(self) -> dict[str, Any]:
        """返回可写入 JSON 的实验描述。"""
        return asdict(self)
