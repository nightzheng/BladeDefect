"""Cross-platform path and configuration helpers."""

from __future__ import annotations

import os
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Mapping


def user_path(value: str | Path) -> Path:
    """Create a native Path while accepting either slash style for relative paths."""
    raw = os.path.expandvars(str(value).strip())
    if os.name == "nt":
        raw = raw.replace("/", "\\")
    else:
        raw = raw.replace("\\", "/")
    return Path(raw).expanduser()


def resolve_path(value: str | Path, base: str | Path | None = None) -> Path:
    """Resolve a user path against *base* (or the current working directory)."""
    raw = str(value).strip()
    path = user_path(raw)
    if os.name != "nt" and PureWindowsPath(raw).is_absolute():
        # A foreign absolute path cannot be resolved on POSIX; preserve it using
        # forward slashes so it can still be serialized or validated clearly.
        return Path(PureWindowsPath(raw).as_posix())
    if not path.is_absolute():
        root = user_path(base) if base is not None else Path.cwd()
        path = root / path
    return path.resolve(strict=False)


def resolve_model_reference(value: str | Path, base: str | Path | None = None) -> str | Path:
    """Resolve model file paths without changing Ultralytics model identifiers."""
    if isinstance(value, Path):
        return resolve_path(value, base)
    raw = str(value).strip()
    candidate = user_path(raw)
    looks_like_path = candidate.is_absolute() or "/" in raw or "\\" in raw
    if looks_like_path or (user_path(base) / candidate if base else candidate).exists():
        return resolve_path(raw, base)
    return raw


def resolve_config_paths(
    config: Mapping[str, Any],
    config_path: str | Path,
    path_fields: Iterable[str],
) -> tuple[dict[str, Any], Path]:
    """Resolve selected config fields against project_root or the config directory."""
    config_file = resolve_path(config_path)
    result = dict(config)
    project_root_value = result.pop("project_root", None)
    project_root = (
        resolve_path(project_root_value, config_file.parent)
        if project_root_value is not None
        else config_file.parent
    )
    for field in path_fields:
        value = result.get(field)
        if value is not None:
            result[field] = resolve_path(value, project_root)
    return result, project_root


def posix_path(path: str | Path) -> str:
    """Return a slash-normalized path suitable for YAML and Ultralytics."""
    return user_path(path).as_posix()
