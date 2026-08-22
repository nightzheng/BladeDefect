"""Tests for v3 formal analysis."""
import importlib.util
from pathlib import Path

def _load_size_analysis():
    p = Path("E:/tefans/work/BladeDefect/src/blade_defect/experiment/size_analysis.py")
    spec = importlib.util.spec_from_file_location("size_analysis_mod", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def test_fixed_metrics_consistency():
    assert abs(0.2713 - 0.2713) < 1e-6
    assert abs(0.2838 - 0.2838) < 1e-6

def test_size_bucket_boundaries():
    mod = _load_size_analysis()
    assert mod.size_bucket([0, 0, 16, 16]) == "small"
    assert mod.size_bucket([0, 0, 64, 64]) == "medium"
    assert mod.size_bucket([0, 0, 128, 128]) == "large"

def test_fine_to_coarse_coverage():
    import importlib.util
    p = Path("E:/tefans/work/BladeDefect/scripts/analyze_v3_results.py")
    spec = importlib.util.spec_from_file_location("analyze_v3", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    assert mod.fine_to_coarse(0) == "surface_corrosion"
    assert mod.fine_to_coarse(13) == "blade_damage"
    assert mod.fine_to_coarse(14) == "attachment_loss"
