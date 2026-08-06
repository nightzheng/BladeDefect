"""Test per-class metrics CSV validation."""
import csv
from pathlib import Path
import pytest

def _read(path): return list(csv.DictReader(path.open(encoding="utf-8-sig")))
S = "metric_branch,class_id,class_name,precision,recall,ap50,ap50_95,experiment_id\nmask,0,c,0.5,0.6,0.55,0.3,x\nmask,1,c2,0.4,0.5,0.45,0.2,x\nbox,0,c,0.55,0.65,0.6,0.35,x\n"
def test_fields(tmp_path: Path):
    p = tmp_path / "a.csv"; p.write_text(S,encoding="utf-8"); rows = _read(p)
    assert len(rows)==3; assert all("metric_branch" in r and "ap50" in r and float(r["precision"])>0 for r in rows)
def test_both_branches(tmp_path: Path):
    p = tmp_path / "b.csv"; p.write_text(S,encoding="utf-8")
    assert {"mask","box"} <= {r["metric_branch"] for r in _read(p)}
def test_empty(tmp_path: Path):
    p = tmp_path/"c.csv"; p.write_text("metric_branch,class_id,class_name,precision,recall,ap50,ap50_95,experiment_id\n",encoding="utf-8")
    assert len(_read(p)) == 0
def test_30_rows(tmp_path: Path):
    lines = ["metric_branch,class_id,class_name,precision,recall,ap50,ap50_95,experiment_id"]
    for b in ("mask","box"):
        for c in range(15): lines.append(f"{b},{c},c,0.5,0.6,0.55,0.3,x")
    p = tmp_path/"d.csv"; p.write_text("\n".join(lines),encoding="utf-8"); assert len(_read(p)) == 30
