from __future__ import annotations

import unittest
from pathlib import Path


class DatasetRegistryTests(unittest.TestCase):
    def test_registry_contains_required_versions_and_split_policy(self) -> None:
        registry = Path("configs/dataset_registry.yaml").read_text(encoding="utf-8")
        self.assertIn("blade-v2-sampled-4987:", registry)
        self.assertIn("blade-v2-full-frozen-48291:", registry)
        self.assertIn("blade-v3-grouped:", registry)
        self.assertIn("purpose: full_frozen_source", registry)
        self.assertIn("train: 0.70", registry)
        self.assertIn("val: 0.15", registry)
        self.assertIn("test: 0.15", registry)
        self.assertIn("no_sample_level_random_split", registry)


if __name__ == "__main__":
    unittest.main()
