"""Test experiment registry build and validation."""
import csv, json
from pathlib import Path
import pytest
from scripts.build_experiment_registry import build_experiment_registry

def test_registry_generates_four_files(tmp_path: Path):
    s = tmp_path / "sampled"; s.mkdir()
    r = s / "exp_test"; r.mkdir()
    (r / "metrics.json").write_text(json.dumps({"name":"exp_test","model":"t.pt","imgsz":640,"epochs":50,"status":"ok"}),encoding="utf-8")
    out = build_experiment_registry(tmp_path/"primary", s, tmp_path/"o")
    names = {p.name for p in out}
    assert names >= {"experiment_registry.csv","duplicate_experiment_ids.csv","metric_reconciliation.csv","asset_completeness.csv"}
    with (tmp_path/"o"/"experiment_registry.csv").open(encoding="utf-8-sig") as f:
        assert next(csv.DictReader(f))["experiment_id"] == "exp_test"

def test_duplicate_detection(tmp_path: Path):
    s = tmp_path / "sampled"; s.mkdir()
    for n in ("a","b"):
        r = s / n; r.mkdir()
        (r/"metrics.json").write_text(json.dumps({"name":"dup","model":"t.pt","imgsz":640,"epochs":50,"status":"ok"}),encoding="utf-8")
    build_experiment_registry(tmp_path/"p", s, tmp_path/"o")
    with (tmp_path/"o"/"duplicate_experiment_ids.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1 and rows[0]["experiment_id"] == "dup"
