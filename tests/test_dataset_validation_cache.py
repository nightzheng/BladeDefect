import json
from pathlib import Path

from PIL import Image

import scripts.cache_dataset_validation as cache_gate
from blade_defect.data import identity_hash
from scripts.cache_dataset_validation import (
    CACHE_SCHEMA_VERSION,
    build_file_state,
    decide_cache,
    run_cached_validation,
)

OBB_LINE = "3 0.1 0.1 0.8 0.2 0.7 0.8 0.2 0.7\n"


def _build_dataset(root: Path, per_split: int = 4) -> Path:
    dataset = root / "blade-obb-mini"
    for split in ("train", "val", "test"):
        (dataset / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset / "labels" / split).mkdir(parents=True, exist_ok=True)
        entries = []
        for index in range(per_split):
            image_path = dataset / "images" / split / f"{split}_{index}.png"
            Image.new("RGB", (24, 16), (index * 20, 10, 200)).save(image_path)
            (dataset / "labels" / split / f"{split}_{index}.txt").write_text(
                OBB_LINE, encoding="utf-8"
            )
            entries.append(str(image_path))
        (dataset / f"{split}.txt").write_text("\n".join(entries) + "\n", encoding="utf-8")
    (dataset / "data.yaml").write_text(
        "path: .\ntrain: train.txt\nval: val.txt\ntest: test.txt\n"
        "names: {0: a, 1: b, 2: c, 3: d}\n",
        encoding="utf-8",
    )
    from blade_defect.data import load_indexed_splits, membership_hash

    indexed = load_indexed_splits(dataset / "data.yaml", ("train", "val", "test"))
    identity = {
        split: identity_hash(sample.sample_id for sample in indexed[split])
        for split in indexed
    }
    (dataset / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "parent_split_identity_sha256": dict(identity),
                "derived_split_identity_sha256": dict(identity),
                "derived_membership_sha256": membership_hash(indexed),
                "test_policy": {"allowed_for_training": False},
            }
        ),
        encoding="utf-8",
    )
    return dataset


def test_cold_run_decodes_every_indexed_image(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    payload = run_cached_validation(dataset, cache_path=cache)
    assert payload["cache"]["decision"] == "cold"
    assert payload["cache"]["decoded_images"] == 12
    assert payload["cache"]["reused_decode_results"] == 0
    assert payload["valid"] is True
    assert payload["splits"]["train"]["valid_instances"] == 4
    assert cache.is_file()
    stored = json.loads(cache.read_text(encoding="utf-8-sig"))
    assert stored["cache_schema_version"] == CACHE_SCHEMA_VERSION
    assert len(stored["decode_results"]) == 12
    assert stored["membership_sha256"] == payload["cache"]["membership_sha256"]


def test_cache_hit_skips_decode_and_reproduces_report(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    cold = run_cached_validation(dataset, cache_path=cache)
    hit = run_cached_validation(dataset, cache_path=cache)
    assert hit["cache"]["decision"] == "hit"
    assert hit["cache"]["decoded_images"] == 0
    assert hit["cache"]["reused_decode_results"] == 12
    for split in ("train", "val", "test"):
        assert hit["splits"][split]["images"] == cold["splits"][split]["images"]
        assert hit["splits"][split]["valid_instances"] == cold["splits"][split]["valid_instances"]
        assert hit["splits"][split]["valid"] == cold["splits"][split]["valid"]
    assert hit["valid"] == cold["valid"]


def test_bootstrap_from_unchanged_full_report_skips_first_decode(
    tmp_path: Path, monkeypatch
) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    initial = run_cached_validation(dataset, cache_path=tmp_path / "initial-cache.json")
    report = tmp_path / "legacy-validation-report.json"
    report.write_text(
        json.dumps(
            {
                "mode": "indexed",
                "valid": True,
                "splits": initial["splits"],
                "split_consistency": initial["split_consistency"],
            }
        ),
        encoding="utf-8",
    )

    calls = []
    monkeypatch.setattr(
        cache_gate,
        "is_image_decodable",
        lambda path: calls.append(path) or True,
    )
    payload = run_cached_validation(
        dataset, cache_path=cache, bootstrap_report=report
    )

    assert payload["cache"]["decision"] == "bootstrap"
    assert payload["cache"]["decoded_images"] == 0
    assert payload["cache"]["bootstrapped_decode_results"] == 12
    assert payload["cache"]["reused_decode_results"] == 12
    assert calls == []
    assert cache.is_file()


def test_bootstrap_rejected_when_an_image_is_newer_than_report(
    tmp_path: Path,
) -> None:
    dataset = _build_dataset(tmp_path)
    initial = run_cached_validation(dataset, cache_path=tmp_path / "initial-cache.json")
    report = tmp_path / "legacy-validation-report.json"
    report.write_text(
        json.dumps(
            {
                "mode": "indexed",
                "valid": True,
                "splits": initial["splits"],
                "split_consistency": initial["split_consistency"],
            }
        ),
        encoding="utf-8",
    )
    victim = dataset / "images" / "train" / "train_0.png"
    stat = victim.stat()
    import os

    os.utime(victim, ns=(stat.st_atime_ns, report.stat().st_mtime_ns + 1_000_000_000))
    payload = run_cached_validation(
        dataset,
        cache_path=tmp_path / "cache.json",
        bootstrap_report=report,
    )
    assert payload["cache"]["decision"] == "cold"
    assert payload["cache"]["decoded_images"] == 12


def test_modified_image_triggers_targeted_recheck(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    run_cached_validation(dataset, cache_path=cache)
    victim = dataset / "images" / "train" / "train_1.png"
    Image.new("RGB", (40, 40), (1, 2, 3)).save(victim)
    payload = run_cached_validation(dataset, cache_path=cache)
    assert payload["cache"]["decision"] == "partial"
    assert payload["cache"]["decoded_images"] == 1
    assert payload["cache"]["reused_decode_results"] == 11
    assert payload["cache"]["rechecked_sample_ids"] == ["train\timages/train/train_1.png"]
    assert payload["valid"] is True


def test_membership_change_forces_full_cold_decode(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    run_cached_validation(dataset, cache_path=cache)
    lines = (dataset / "val.txt").read_text(encoding="utf-8").splitlines()
    (dataset / "val.txt").write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
    payload = run_cached_validation(dataset, cache_path=cache)
    assert payload["cache"]["decision"] == "cold"
    assert "membership_changed" in payload["cache"]["decision_reasons"]
    assert payload["cache"]["decoded_images"] == 11
    assert payload["cache"]["reused_decode_results"] == 0


def test_corrupt_image_stays_flagged_on_cache_hit(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    broken = dataset / "images" / "test" / "test_0.png"
    broken.write_bytes(b"definitely not an image")
    cold = run_cached_validation(dataset, cache_path=cache)
    assert cold["valid"] is False
    assert cold["splits"]["test"]["corrupt_images"] == ["images/test/test_0.png"]
    hit = run_cached_validation(dataset, cache_path=cache)
    assert hit["cache"]["decision"] == "hit"
    assert hit["cache"]["decoded_images"] == 0
    assert hit["valid"] is False
    assert hit["splits"]["test"]["corrupt_images"] == ["images/test/test_0.png"]


def test_label_format_is_fully_rechecked_on_cache_hit(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    run_cached_validation(dataset, cache_path=cache)
    (dataset / "labels" / "val" / "val_0.txt").write_text(
        "99 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n", encoding="utf-8"
    )
    payload = run_cached_validation(dataset, cache_path=cache)
    assert payload["cache"]["decoded_images"] == 0
    assert payload["splits"]["val"]["valid"] is False
    assert payload["splits"]["val"]["error_type_counts"] == {"obb_class_id": 1}
    assert payload["valid"] is False


def test_validation_version_change_invalidates_cache(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    run_cached_validation(dataset, cache_path=cache)
    stored = json.loads(cache.read_text(encoding="utf-8-sig"))
    stored["cache_schema_version"] = CACHE_SCHEMA_VERSION + 1
    cache.write_text(json.dumps(stored), encoding="utf-8")
    payload = run_cached_validation(dataset, cache_path=cache)
    assert payload["cache"]["decision"] == "cold"
    assert "validation_version_changed" in payload["cache"]["decision_reasons"]


def test_directory_name_or_timestamp_alone_cannot_satisfy_cache(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    run_cached_validation(dataset, cache_path=cache)
    victim = dataset / "images" / "train" / "train_2.png"
    original = victim.read_bytes()
    victim.write_bytes(original)  # 目录名与内容均未变，仅触碰 mtime
    stat = victim.stat()
    import os

    os.utime(victim, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))
    payload = run_cached_validation(dataset, cache_path=cache)
    assert payload["cache"]["decision"] == "partial"
    assert payload["cache"]["rechecked_sample_ids"] == ["train\timages/train/train_2.png"]


def test_sampled_decode_for_ci_smoke_does_not_touch_formal_cache(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    payload = run_cached_validation(dataset, cache_path=cache, decode_sample=3, decode_seed=42)
    assert payload["cache"]["decision"] == "sampled"
    assert payload["cache"]["decoded_images"] == 3
    assert payload["cache"]["decode_unchecked_images"] == 9
    assert payload["cache"]["cache_written"] is False
    assert payload["cache"]["sampled_decode"]["seed"] == 42
    assert not cache.exists()
    repeat = run_cached_validation(dataset, cache_path=cache, decode_sample=3, decode_seed=42)
    assert repeat["cache"]["decoded_images"] == 3


def test_sampled_mode_decodes_exactly_the_sample(tmp_path: Path, monkeypatch) -> None:
    dataset = _build_dataset(tmp_path)
    calls = []
    real_decode = cache_gate.is_image_decodable

    def counting_decode(path):
        calls.append(path)
        return real_decode(path)

    monkeypatch.setattr(cache_gate, "is_image_decodable", counting_decode)
    payload = run_cached_validation(
        dataset, cache_path=tmp_path / "cache.json", decode_sample=5, decode_seed=42
    )
    assert payload["cache"]["decision"] == "sampled"
    assert len(calls) == 5
    assert payload["cache"]["decoded_images"] == 5
    assert payload["cache"]["decode_unchecked_images"] == 7
    assert payload["valid"] is True


def test_no_cache_forces_cold_and_still_writes_cache(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    run_cached_validation(dataset, cache_path=cache)
    before = cache.read_bytes()
    payload = run_cached_validation(dataset, cache_path=cache, use_cache=False)
    assert payload["cache"]["decision"] == "cold"
    assert payload["cache"]["decoded_images"] == 12
    assert payload["cache"]["cache_written"] is True
    assert cache.read_bytes() != before or json.loads(cache.read_text())["last_decision"] == "cold"


def test_pure_hit_does_not_rewrite_cache(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    run_cached_validation(dataset, cache_path=cache)
    before = cache.read_bytes()
    payload = run_cached_validation(dataset, cache_path=cache)
    assert payload["cache"]["decision"] == "hit"
    assert payload["cache"]["cache_written"] is False
    assert cache.read_bytes() == before


def test_build_file_state_and_decide_cache_units(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    from blade_defect.data import load_indexed_splits, membership_hash

    indexed = load_indexed_splits(dataset / "data.yaml", ("train", "val", "test"))
    states, state_hash = build_file_state(indexed)
    assert len(states) == 12
    decision, reasons, recheck = decide_cache(
        None,
        membership=membership_hash(indexed),
        file_state_hash=state_hash,
        num_classes=4,
        min_area=1e-8,
        states=states,
    )
    assert decision == "cold"
    assert reasons == ["cache_missing"]
    assert recheck == []


def test_missing_cache_entry_is_decoded_not_silently_trusted(tmp_path: Path) -> None:
    dataset = _build_dataset(tmp_path)
    cache = tmp_path / "cache.json"
    run_cached_validation(dataset, cache_path=cache)
    stored = json.loads(cache.read_text(encoding="utf-8-sig"))
    removed_key = next(iter(stored["decode_results"]))
    del stored["decode_results"][removed_key]
    cache.write_text(json.dumps(stored), encoding="utf-8")
    payload = run_cached_validation(dataset, cache_path=cache)
    assert payload["cache"]["decision"] == "partial"
    assert "cache_entry_missing" in payload["cache"]["decision_reasons"]
    assert payload["valid"] is True
