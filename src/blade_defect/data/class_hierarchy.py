"""Validated mapping from 15 fine-grained classes to six coarse groups."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


EXPECTED_FINE_CLASSES = set(range(15))


@dataclass(frozen=True)
class CoarseClass:
    key: str
    name_zh: str
    class_ids: tuple[int, ...]


def load_class_hierarchy(path: str | Path) -> tuple[CoarseClass, ...]:
    """Load the deliberately small hierarchy YAML without a runtime dependency."""
    source = Path(path)
    groups: list[CoarseClass] = []
    current_key: str | None = None
    current_name = ""
    current_ids: tuple[int, ...] | None = None

    def flush() -> None:
        nonlocal current_key, current_name, current_ids
        if current_key is None:
            return
        if not current_name or current_ids is None:
            raise ValueError(f"hierarchy group {current_key} is incomplete")
        groups.append(CoarseClass(current_key, current_name, current_ids))
        current_key, current_name, current_ids = None, "", None

    for raw_line in source.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line in {"groups:", "version: 1"}:
            continue
        group_match = re.fullmatch(r"([a-z][a-z0-9_]*):", line)
        if group_match:
            flush()
            current_key = group_match.group(1)
            continue
        if current_key is None:
            raise ValueError(f"unexpected hierarchy line: {line}")
        if line.startswith("name_zh:"):
            current_name = line.split(":", 1)[1].strip()
        elif line.startswith("class_ids:"):
            values = line.split(":", 1)[1].strip()
            if not (values.startswith("[") and values.endswith("]")):
                raise ValueError(f"class_ids must use an inline integer list: {line}")
            body = values[1:-1].strip()
            current_ids = tuple(int(value.strip()) for value in body.split(",") if value.strip())
        else:
            raise ValueError(f"unexpected hierarchy field: {line}")
    flush()

    if len(groups) != 6:
        raise ValueError(f"expected 6 coarse groups, found {len(groups)}")
    assigned = [class_id for group in groups for class_id in group.class_ids]
    duplicates = sorted({class_id for class_id in assigned if assigned.count(class_id) > 1})
    missing = sorted(EXPECTED_FINE_CLASSES - set(assigned))
    extras = sorted(set(assigned) - EXPECTED_FINE_CLASSES)
    if duplicates or missing or extras:
        raise ValueError(
            f"invalid fine-class coverage: duplicates={duplicates}, missing={missing}, extras={extras}"
        )
    return tuple(groups)


def fine_to_coarse(groups: tuple[CoarseClass, ...]) -> dict[int, CoarseClass]:
    return {class_id: group for group in groups for class_id in group.class_ids}
