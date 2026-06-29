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
    """Resolve auto to the first CUDA device or CPU; validate explicit values."""
    normalized = "auto" if device is None else str(device).strip().lower()
    if normalized == "auto":
        return 0 if _cuda_is_available() else "cpu"
    if normalized == "cpu":
        return "cpu"
    if normalized in {"0", "cuda", "cuda:0"}:
        return 0
    raise ValueError("device 仅支持 auto、0 或 cpu")
