"""Tests for v3 formal analysis."""


def _load_size_analysis():
    from blade_defect.experiment import size_analysis

    return size_analysis

def test_fixed_metrics_consistency():
    assert abs(0.2713 - 0.2713) < 1e-6
    assert abs(0.2838 - 0.2838) < 1e-6

def test_size_bucket_boundaries():
    mod = _load_size_analysis()
    assert mod.size_bucket([0, 0, 16, 16]) == "small"
    assert mod.size_bucket([0, 0, 64, 64]) == "medium"
    assert mod.size_bucket([0, 0, 128, 128]) == "large"

def test_fine_to_coarse_coverage():
    from scripts.analyze_v3_results import fine_to_coarse

    assert fine_to_coarse(0) == "surface_corrosion"
    assert fine_to_coarse(13) == "blade_damage"
    assert fine_to_coarse(14) == "attachment_loss"
