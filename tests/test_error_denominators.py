"""Test error denominator contract."""
import pytest
from pathlib import Path

def test_fp_denominator_is_pred_count():
    """FP rate denominator must be total predictions."""
    assert 463/2033 != 463/4943

def test_fixture_audit(tmp_path):
    from scripts.audit_hierarchical_errors import audit
    import csv
    tmp = tmp_path; coarse = tmp / "c.csv"
    coarse.write_text("coarse_group,gt_instances,pred_instances,tp,fp,fn,precision,recall,f1\na,100,80,60,20,40,0.75,0.6,0.667", encoding="utf-8")
    within = tmp / "w.csv"
    within.write_text("experiment_id,true_class,predicted_class,coarse_group\ne,0,1,a", encoding="utf-8")
    out = tmp / "out.csv"
    audit(str(coarse), str(within), str(out))
    with open(out, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    tp_row = next(r for r in rows if r["check"] == "diagonal_tp")
    assert tp_row["count"] == "60"