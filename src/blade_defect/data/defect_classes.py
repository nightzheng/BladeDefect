"""Wind turbine blade defect class names and category groups."""

from __future__ import annotations


DEFECT_CLASSES: dict[int, str] = {
    0: "表面腐蚀--保护膜损伤",
    1: "表面腐蚀--玻纤腐蚀",
    2: "表面腐蚀--合模缝裸漏",
    3: "表面腐蚀--胶衣腐蚀",
    4: "表面裂纹--玻纤裂纹",
    5: "表面裂纹--后缘弦向裂纹",
    6: "表面缺陷--表面掉漆",
    7: "表面缺陷--表面油污",
    8: "表面缺陷--胶衣脱落",
    9: "表面缺陷--胶衣裂纹",
    10: "维修痕迹",
    11: "叶片损伤--玻纤损伤",
    12: "叶片损伤--叶片开裂",
    13: "叶片损伤--结构损伤",
    14: "附件脱落--接闪器脱落",
}

DEFECT_GROUPS: dict[str, list[int]] = {
    "表面腐蚀": [0, 1, 2, 3],
    "表面裂纹": [4, 5],
    "表面缺陷": [6, 7, 8, 9],
    "维修痕迹": [10],
    "叶片损伤": [11, 12, 13],
    "附件脱落": [14],
}


def _validate_class_id(class_id: int) -> None:
    if not isinstance(class_id, int) or isinstance(class_id, bool) or class_id not in DEFECT_CLASSES:
        raise ValueError(f"非法缺陷类别 ID：{class_id!r}；有效范围为 0-14")


def get_class_name(class_id: int) -> str:
    """Return the defect class name for a numeric class ID."""
    _validate_class_id(class_id)
    return DEFECT_CLASSES[class_id]


def get_group_name(class_id: int) -> str:
    """Return the defect group containing a numeric class ID."""
    _validate_class_id(class_id)
    for group_name, class_ids in DEFECT_GROUPS.items():
        if class_id in class_ids:
            return group_name
    raise RuntimeError(f"缺陷类别 ID {class_id} 未配置类别分组")


def get_group_classes(group_name: str) -> list[int]:
    """Return a copy of the class IDs assigned to a defect group."""
    try:
        return list(DEFECT_GROUPS[group_name])
    except KeyError as error:
        available = "、".join(DEFECT_GROUPS)
        raise ValueError(f"未知缺陷分组：{group_name!r}；可用分组为：{available}") from error
