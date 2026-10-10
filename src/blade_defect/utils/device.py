"""Shared compute-device selection."""

from __future__ import annotations

from typing import TypeAlias

DeviceInput: TypeAlias = str | int | None
ResolvedDevice: TypeAlias = str | int


def _cuda_is_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def resolve_device(device: DeviceInput = "auto") -> ResolvedDevice:
    """Resolve auto to the first CUDA device or CPU; validate explicit values.

    接受任意非负 GPU 序号（如 0、1 或 cuda:1），以支持多卡实例；
    逗号分隔的多卡列表仍不支持（单任务单卡为当前设计）。
    """
    normalized = "auto" if device is None else str(device).strip().lower()
    if normalized == "auto":
        return 0 if _cuda_is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized in {"0", "cuda", "cuda:0"}:
        return 0
    if normalized.isdigit():
        return int(normalized)
    if normalized.startswith("cuda:") and normalized[5:].isdigit():
        return int(normalized[5:])
    raise ValueError("device 仅支持 auto、cpu 或非负 GPU 序号（如 0、1）")
