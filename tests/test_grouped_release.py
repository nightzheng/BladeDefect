from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

from scripts.release_grouped_dataset import release_dataset


def _sample(root: Path, split: str, name: str, class_id: int | None = 0) -> str:
    image = root / "images" / split / f"{name}.jpg"
    label = root / "labels" / split / f"{name}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image-" + name.encode())
    label.write_text("" if class_id is None else f"{class_id} 0.1 0.1 0.2 0.1 0.2 0.2\n", encoding="utf-8")
    return f"./images/{split}/{name}.jpg"


def test_release_is_portable_complete_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "blade-v2"
    rows = [
        (_sample(source, "train", "a"), "train", "train", "0"),
        (_sample(source, "train", "b"), "train", "val", "0"),
        (_sample(source, "val", "c", None), "val", "test", ""),
    ]
    (source / "data.yaml").write_text(yaml.safe_dump({"path": ".", "names": {0: "defect"}}), encoding="utf-8")
    (source / "dataset_manifest.json").write_text(
        json.dumps({"dataset_id": "source", "frozen_labels_sha256": "labels"}), encoding="utf-8"
    )
    split_source = tmp_path / "candidate-lists"
    split_source.mkdir()
    for split in ("train", "val", "test"):
        selected = [row[0] for row in rows if row[2] == split]
        (split_source / f"{split}.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    manifest = {
        "dataset_id": "candidate",
        "source_dataset_id": "source",
        "random_seed": 42,
        "instances": 2,
        "frozen_labels_sha256": "labels",
        "balance_summary": {"all_fine_and_coarse_classes_present": True},
        "outputs": {split: str(split_source / f"{split}.txt") for split in ("train", "val", "test")},
    }
    (candidate / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assignments = tmp_path / "assignments.csv"
    with assignments.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["relative_image", "source_split", "target_split", "group_id", "duplicate_group_id", "sequence_group", "class_ids"],
        )
        writer.writeheader()
        for index, (image, old, new, classes) in enumerate(rows):
            writer.writerow({"relative_image": image, "source_split": old, "target_split": new, "group_id": f"g{index}", "duplicate_group_id": "", "sequence_group": f"s{index}", "class_ids": classes})
    distribution = tmp_path / "distribution.csv"
    distribution.write_text(
        "level,class_key,class_name,train_images,train_instances,val_images,val_instances,test_images,test_instances\n"
        "fine_15,0,defect,1,1,1,1,0,0\n",
        encoding="utf-8",
    )
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"valid": True}), encoding="utf-8")
    args = argparse.Namespace(
        candidate=candidate,
        split_source=split_source,
        source_dataset=source,
        output=tmp_path / "release",
        dataset_id="release",
        assignments=assignments,
        distribution_source=distribution,
        results=tmp_path / "results",
        audit_summary=audit,
        generated_at="2026-08-04T00:00:00+08:00",
        allow_nonstandard_counts=True,
    )
    first = release_dataset(args)
    second = release_dataset(args)
    assert first == second
    assert first["valid"]
    assert first["counts"] == {"train": 1, "val": 1, "test": 1}
    released_manifest = json.loads((args.output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert released_manifest["status"] == "released"
    assert released_manifest["test_lock"]["automatic_unlock"] is False
    assert "../blade-v2/images/" in (args.output / "train.txt").read_text(encoding="utf-8")
