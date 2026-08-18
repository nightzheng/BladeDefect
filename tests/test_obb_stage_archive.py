import os
from pathlib import Path
from types import SimpleNamespace

from scripts.archive_obb_stage_assets import audit_preview_files, verify_hardlinks


def _samples(*paths: Path) -> dict[str, list]:
    return {"train": [SimpleNamespace(image_path=path) for path in paths]}


def test_verify_hardlinks_confirms_real_links(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"image-bytes")
    link = tmp_path / "images" / "train" / "linked.jpg"
    link.parent.mkdir(parents=True)
    os.link(source, link)
    stats = verify_hardlinks(_samples(link))
    assert stats == {"total_indexed": 1, "stat_ok": 1, "stat_failed": 0, "linked": 1}


def test_verify_hardlinks_flags_missing_and_copied_files(tmp_path: Path) -> None:
    copied = tmp_path / "copied.jpg"
    copied.write_bytes(b"image-bytes")
    missing = tmp_path / "missing.jpg"
    stats = verify_hardlinks(_samples(copied, missing))
    assert stats["total_indexed"] == 2
    assert stats["stat_ok"] == 1
    assert stats["stat_failed"] == 1
    assert stats["linked"] == 0


def test_audit_preview_files_requires_exact_name_match(tmp_path: Path) -> None:
    preview_dir = tmp_path / "conversion_preview"
    (preview_dir / "train").mkdir(parents=True)
    (preview_dir / "train" / "001_a.jpg").write_bytes(b"x")
    (preview_dir / "train" / "002_b.jpg").write_bytes(b"x")
    (preview_dir / "preview_summary.json").write_text("{}", encoding="utf-8")
    ok = audit_preview_files(preview_dir, ["001_a.jpg", "002_b.jpg"])
    assert ok["on_disk"] == 2 and ok["missing"] == [] and ok["unexpected"] == []

    (preview_dir / "train" / "002_b.jpg").unlink()
    (preview_dir / "train" / "Thumbs.db").write_bytes(b"x")
    degraded = audit_preview_files(preview_dir, ["001_a.jpg", "002_b.jpg"])
    assert degraded["on_disk"] == 1
    assert degraded["missing"] == ["002_b.jpg"]
    assert degraded["unexpected"] == []

    (preview_dir / "val").mkdir()
    (preview_dir / "val" / "999_extra.png").write_bytes(b"x")
    extra = audit_preview_files(preview_dir, ["001_a.jpg"])
    assert extra["unexpected"] == ["999_extra.png"]
