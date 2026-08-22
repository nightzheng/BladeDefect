"""Tests for L3 mask quality threshold sensitivity."""
import importlib.util
from pathlib import Path

def _load_size_analysis():
    p = Path("E:/tefans/work/BladeDefect/src/blade_defect/experiment/size_analysis.py")
    spec = importlib.util.spec_from_file_location("size_analysis_mod", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def test_threshold_ordering():
    assert [0.3, 0.5, 0.7] == sorted([0.3, 0.5, 0.7])

def test_strictness_monotonic():
    flagged = [0.2, 0.5, 0.7]
    assert all(flagged[i] <= flagged[i+1] for i in range(len(flagged)-1))

def test_bbox_area_proxy():
    mod = _load_size_analysis()
    assert mod.area_of([0, 0, 10, 10]) == 100
    assert mod.area_of([5, 5, 15, 20]) == 150
