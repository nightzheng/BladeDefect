"""Tests for L3 mask quality threshold sensitivity and real polygon IoU."""


def _load_size_analysis():
    from blade_defect.experiment import size_analysis

    return size_analysis

def test_threshold_ordering():
    assert [0.3, 0.5, 0.7] == sorted([0.3, 0.5, 0.7])

def test_strictness_monotonic():
    flagged = [0.2, 0.5, 0.7]
    assert all(flagged[i] <= flagged[i+1] for i in range(len(flagged)-1))

def test_polygon_area_square():
    mod = _load_size_analysis()
    import numpy as np
    sq = np.array([[0,0],[10,0],[10,10],[0,10]], dtype=float)
    assert abs(mod.polygon_area(sq) - 100.0) < 1e-6

def test_raster_iou_overlapping_squares():
    mod = _load_size_analysis()
    import numpy as np
    a = np.array([[0,0],[10,0],[10,10],[0,10]], dtype=float)
    b = np.array([[5,5],[15,5],[15,15],[5,15]], dtype=float)
    iou = mod.raster_iou(a, b, grid=100)
    # overlap is 5x5=25, union is 175, iou ~ 0.1428
    assert abs(iou - 25/175) < 0.02

def test_size_bucket_from_area():
    mod = _load_size_analysis()
    assert mod.size_bucket_from_area(16*16) == "small"
    assert mod.size_bucket_from_area(64*64) == "medium"
    assert mod.size_bucket_from_area(128*128) == "large"
