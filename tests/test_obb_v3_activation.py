import json
from pathlib import Path

import yaml
from PIL import Image

from blade_defect.data import (
    DEFECT_CLASSES,
    check_obb_indexed_samples,
    load_indexed_splits,
    membership_hash,
)
from scripts.convert_seg_to_obb import build_parser, convert_indexed_dataset
from scripts.run_obb_smoke import validate_smoke_dataset
from scripts.visualize_obb_labels import generate_previews


SEG_LABEL = "12 0.1 0.1 0.8 0.2 0.7 0.8 0.2 0.7\n"


def _build_indexed_parent(tmp_path: Path) -> Path:
    frozen = tmp_path / "frozen"
    release = tmp_path / "blade-v3-grouped"
    release.mkdir()
    samples = {
        "train": frozen / "images" / "legacy_train" / "a.png",
        "val": frozen / "images" / "legacy_train" / "b.png",
        "test": frozen / "images" / "legacy_val" / "c.png",
    }
    for image_path in samples.values():
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (20, 12), "white").save(image_path)
        relative = image_path.relative_to(frozen / "images")
        label_path = frozen / "labels" / relative.with_suffix(".txt")
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text(SEG_LABEL, encoding="utf-8")

    (release / "train.txt").write_text("../frozen/images/legacy_train/a.png\n", encoding="utf-8")
    (release / "val.txt").write_text(str(samples["val"]) + "\n", encoding="utf-8")
    (release / "test.txt").write_text("../frozen/images/legacy_val/c.png\n", encoding="utf-8")
    (release / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": ".", "train": "train.txt", "val": "val.txt", "test": "test.txt",
                "names": DEFECT_CLASSES,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (release / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "blade-v3-grouped",
                "samples": {"train": 1, "val": 1, "test": 1},
                "instances": 3,
                "groups": {"group-a": ["images/legacy_train/a.png", "images/legacy_train/b.png"]},
            }
        ),
        encoding="utf-8",
    )
    return release / "data.yaml"


def test_indexed_conversion_accepts_relative_and_absolute_entries(tmp_path: Path) -> None:
    parent_data = _build_indexed_parent(tmp_path)
    output = tmp_path / "blade-v3-grouped-obb"
    results = tmp_path / "results"

    summary = convert_indexed_dataset(
        parent_data, output, results, image_mode="copy", parent_dataset_id="blade-v3-grouped"
    )

    assert summary["valid"] is True
    assert summary["totals"]["images"] == 3
    assert summary["totals"]["source_instances"] == 3
    assert summary["totals"]["converted_instances"] == 3
    assert set(summary["splits"]) == {"train", "val", "test"}
    assert (output / "images" / "legacy_train" / "a.png").is_file()
    assert len((output / "labels" / "legacy_train" / "a.txt").read_text().split()) == 9
    assert (results / "conversion_summary.json").is_file()
    assert (results / "invalid_obb_labels.csv").is_file()


def test_parent_and_derived_split_identity_and_manifest_linkage(tmp_path: Path) -> None:
    parent_data = _build_indexed_parent(tmp_path)
    output = tmp_path / "blade-v3-grouped-obb"
    convert_indexed_dataset(parent_data, output, tmp_path / "results", image_mode="copy")

    parent = load_indexed_splits(parent_data, ("train", "val", "test"))
    derived = load_indexed_splits(output / "data.yaml", ("train", "val", "test"))
    for split in ("train", "val", "test"):
        assert {sample.sample_id for sample in parent[split]} == {
            sample.sample_id for sample in derived[split]
        }
        assert check_obb_indexed_samples(derived[split]).valid
    assert membership_hash(parent) == membership_hash(derived)

    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["parent_dataset_id"] == "blade-v3-grouped"
    assert manifest["derived_dataset_id"] == "blade-v3-grouped-obb"
    assert manifest["conversion_type"] == "seg_polygon_to_min_area_obb"
    assert manifest["parent_manifest_sha256"]
    assert manifest["split_inheritance"]["no_cross_split_movement"] is True
    assert manifest["split_inheritance"]["group_identity_preserved"] is True
    assert manifest["parent_split_identity_sha256"] == manifest["derived_split_identity_sha256"]


def test_test_split_exists_but_smoke_gate_reads_train_and_val_only(tmp_path: Path) -> None:
    parent_data = _build_indexed_parent(tmp_path)
    output = tmp_path / "blade-v3-grouped-obb"
    convert_indexed_dataset(parent_data, output, tmp_path / "results", image_mode="copy")
    (output / "labels" / "legacy_val" / "c.txt").write_text("invalid test label\n", encoding="utf-8")

    validation, val_images = validate_smoke_dataset(output / "data.yaml")

    assert validation["validated_splits"] == ["train", "val"]
    assert validation["test_used"] is False
    assert len(val_images) == 1


def test_indexed_preview_covers_all_three_splits(tmp_path: Path) -> None:
    parent_data = _build_indexed_parent(tmp_path)
    output = tmp_path / "blade-v3-grouped-obb"
    convert_indexed_dataset(parent_data, output, tmp_path / "results", image_mode="copy")

    summary = generate_previews(
        parent_data.parent,
        output,
        tmp_path / "activation" / "preview",
        count_per_split=1,
        acceptance=tmp_path / "activation" / "preview_acceptance.csv",
    )

    assert summary["written"] == 3
    assert {split: item["written"] for split, item in summary["splits"].items()} == {
        "train": 1, "val": 1, "test": 1,
    }
    rows = (tmp_path / "activation" / "preview_acceptance.csv").read_text(
        encoding="utf-8-sig"
    ).splitlines()
    assert len(rows) == 4


def test_cli_selects_indexed_write_mode() -> None:
    args = build_parser().parse_args(
        ["--source-data", "parent/data.yaml", "--output", "derived"]
    )
    assert args.source is None
    assert args.source_data == Path("parent/data.yaml")
    assert args.output == Path("derived")
