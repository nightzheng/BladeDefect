"""Test environment metadata in experiment registry."""
import csv, pytest
from pathlib import Path

def test_registry_has_env_fields(tmp_path):
    p = tmp_path / "reg.csv"
    p.write_text("experiment_id,code_commit,device,fps_method,status\nexp1,abc123,NVIDIA RTX 4060,ultralytics_val_speed_inference_ms,ok", encoding="utf-8")
    with open(p, encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert row["experiment_id"] == "exp1"
    assert row["device"]
    assert row["code_commit"]

def test_missing_env_flagged(tmp_path):
    p = tmp_path / "reg2.csv"
    p.write_text("experiment_id,code_commit,device,fps_method,status\nold,unknown,,,legacy", encoding="utf-8")
    with open(p, encoding="utf-8-sig") as f:
        row = next(csv.DictReader(f))
    assert not row["device"] or not row["code_commit"]