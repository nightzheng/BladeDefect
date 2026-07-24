import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "blade_defect" / "data" / "class_hierarchy.py"
SPEC = importlib.util.spec_from_file_location("blade_class_hierarchy_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
fine_to_coarse = MODULE.fine_to_coarse
load_class_hierarchy = MODULE.load_class_hierarchy


class ClassHierarchyTests(unittest.TestCase):
    def test_hierarchy_covers_each_fine_class_once(self) -> None:
        groups = load_class_hierarchy(PROJECT_ROOT / "configs" / "class_hierarchy.yaml")
        mapping = fine_to_coarse(groups)
        self.assertEqual(len(groups), 6)
        self.assertEqual(set(mapping), set(range(15)))
        self.assertEqual(len(mapping), 15)

    def test_expected_group_membership(self) -> None:
        groups = load_class_hierarchy(PROJECT_ROOT / "configs" / "class_hierarchy.yaml")
        by_key = {group.key: group.class_ids for group in groups}
        self.assertEqual(
            by_key,
            {
                "surface_corrosion": (0, 1, 2, 3),
                "surface_crack": (4, 5),
                "surface_defect": (6, 7, 8, 9),
                "repair_trace": (10,),
                "blade_damage": (11, 12, 13),
                "attachment_loss": (14,),
            },
        )


if __name__ == "__main__":
    unittest.main()
