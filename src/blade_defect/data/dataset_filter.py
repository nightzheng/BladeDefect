"""Load filename-based dataset filtering decisions from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


SUPPORTED_ACTIONS = ("exclude", "review", "keep_negative")


@dataclass(frozen=True)
class ImageFilterDecision:
    """A confirmed or pending decision for one source image."""

    filename: str
    action: str
    issue: str | None
    reason: str
    status: str


@dataclass(frozen=True)
class DatasetFilter:
    """Validated filter configuration indexed by case-insensitive filename."""

    source: Path
    issue_types: tuple[str, ...]
    actions: tuple[str, ...]
    decisions: dict[str, ImageFilterDecision]

    def decision_for(self, image: str | Path) -> ImageFilterDecision | None:
        return self.decisions.get(Path(image).name.casefold())


def _string_list(payload: dict[str, Any], field: str) -> tuple[str, ...]:
    values = payload.get(field)
    if (
        not isinstance(values, list)
        or not values
        or not all(isinstance(value, str) for value in values)
    ):
        raise ValueError(f"dataset filter 的 {field} 必须是非空字符串列表")
    if len({value.casefold() for value in values}) != len(values):
        raise ValueError(f"dataset filter 的 {field} 不能包含重复值")
    return tuple(values)


def load_dataset_filter(config_path: str | Path) -> DatasetFilter:
    """Read and validate a filename-based dataset filter configuration."""

    source = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dataset filter 顶层必须是 YAML 映射")
    if payload.get("version") != 1:
        raise ValueError("dataset filter 目前只支持 version: 1")

    issue_types = _string_list(payload, "issue_types")
    actions = _string_list(payload, "actions")
    unsupported = sorted(set(actions) - set(SUPPORTED_ACTIONS))
    if unsupported:
        raise ValueError(f"dataset filter 包含未实现的 action：{', '.join(unsupported)}")

    image_entries = payload.get("images")
    if not isinstance(image_entries, list):
        raise ValueError("dataset filter 的 images 必须是列表")

    decisions: dict[str, ImageFilterDecision] = {}
    for index, entry in enumerate(image_entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"dataset filter 的第 {index} 个 images 项必须是映射")
        filename = entry.get("filename")
        action = entry.get("action")
        issue = entry.get("issue")
        reason = entry.get("reason")
        status = entry.get("status")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError(f"dataset filter 的第 {index} 项缺少 filename")
        if Path(filename).name != filename:
            raise ValueError(f"dataset filter 只按文件名匹配，不能包含目录：{filename}")
        if action not in actions:
            raise ValueError(f"{filename} 使用了未声明的 action：{action}")
        if issue is not None and issue not in issue_types:
            raise ValueError(f"{filename} 使用了未声明的 issue：{issue}")
        if action in {"exclude", "review"} and issue is None:
            raise ValueError(f"{filename} 的 action={action} 时必须填写 issue")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{filename} 缺少 reason")
        if not isinstance(status, str) or not status.strip():
            raise ValueError(f"{filename} 缺少 status")

        key = filename.casefold()
        if key in decisions:
            raise ValueError(f"dataset filter 包含重复文件名：{filename}")
        decisions[key] = ImageFilterDecision(
            filename=filename,
            action=action,
            issue=issue,
            reason=reason,
            status=status,
        )

    return DatasetFilter(
        source=source,
        issue_types=issue_types,
        actions=actions,
        decisions=decisions,
    )
