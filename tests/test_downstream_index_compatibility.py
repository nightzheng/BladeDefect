from __future__ import annotations

from pathlib import Path

import yaml

from blade_defect.data.indexed_splits import load_indexed_splits, membership_hash
from scripts.convert_seg_to_obb import dry_run_index_check as obb_check
from scripts.remap_to_6class import dry_run_index_check as coarse_check


def test_seg_coarse_and_obb_share_exact_split_membership(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        image = dataset / "images" / "train" / f"{split}.jpg"
        label = dataset / "labels" / "train" / f"{split}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"not-decoded-in-dry-run")
        label.write_text("0 0.1 0.1 0.3 0.1 0.3 0.3 0.1 0.3\n", encoding="utf-8")
        (dataset / f"{split}.txt").write_text(f"./images/train/{split}.jpg\n", encoding="utf-8")
    data = dataset / "data.yaml"
    data.write_text(
        yaml.safe_dump({"path": ".", "train": "train.txt", "val": "val.txt", "test": "test.txt", "names": {0: "defect"}}),
        encoding="utf-8",
    )
    indexed = load_indexed_splits(data, ("train", "val", "test"))
    expected = membership_hash(indexed)
    coarse = coarse_check(data)
    obb = obb_check(data)
    assert coarse["valid"] and obb["valid"]
    assert coarse["membership_sha256"] == obb["membership_sha256"] == expected
    assert coarse["splits"] == obb["splits"] == {"train": 1, "val": 1, "test": 1}
