"""Synthetic val exports only: lock the archived polygon matching contract."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.decompose_val_errors import _poly, decompose


def _rect(x=0, y=0, width=100, height=100):
    return [[x, y], [x + width, y], [x + width, y + height], [x, y + height]]


def _gt(points=None, cls=0):
    points = _rect() if points is None else points
    # Deliberately non-square image to check normalized GT vs pixel predictions.
    return {"class_id": cls, "polygon": [v / scale for p in points for v, scale in zip(p, (200, 400))]}


def _pred(points=None, cls=0, field="mask_polygon"):
    return {"class_id": cls, field: _rect() if points is None else points}


def _export(tmp_path, gts, preds):
    path = tmp_path / "synthetic_val.json"
    path.write_text(json.dumps({"samples": [{"image_width": 200, "image_height": 400,
                                             "ground_truth": gts, "predictions": preds}]}), encoding="utf-8")
    return path


def _expected(per_class=None, confusion=None, fp=None):
    return {"per_class": per_class or {}, "confusion_pairs_top30": confusion or {},
            "false_positives": fp or {}}


@pytest.mark.parametrize("field", ["mask_polygon", "obb_points"])
def test_seg_and_obb_polygon_match(tmp_path, field):
    # A rotated four-point polygon, represented as flat normalized GT and pixel pairs.
    points = [[50, 0], [100, 50], [50, 100], [0, 50]]
    result = decompose(_export(tmp_path, [_gt(points)], [_pred(points, field=field)]))
    assert result == _expected({"0": {"ok": 1}})


@pytest.mark.parametrize(("width", "bucket", "fp"), [
    (100, "ok", {}), (75, "ok", {}), (74.9, "mislocalized", {}),
    (50, "mislocalized", {}), (49.9, "missed", {"0": 1}),
])
def test_iou_thresholds(tmp_path, width, bucket, fp):
    result = decompose(_export(tmp_path, [_gt()], [_pred(_rect(width=width))]))
    assert result == _expected({"0": {bucket: 1}}, fp=fp)


@pytest.mark.parametrize("width", [50, 75, 100])
def test_wrong_class_above_match_threshold(tmp_path, width):
    result = decompose(_export(tmp_path, [_gt(cls=2)], [_pred(_rect(width=width), cls=3)]))
    assert result == _expected({"2": {"misclassified": 1}}, {"2>3": 1})


def test_empty_predictions_are_missed(tmp_path):
    assert decompose(_export(tmp_path, [_gt()], [])) == _expected({"0": {"missed": 1}})


def test_unmatched_predictions_are_fp(tmp_path):
    result = decompose(_export(tmp_path, [_gt()], [_pred(), _pred(_rect(x=150), cls=1)]))
    assert result == _expected({"0": {"ok": 1}}, fp={"1": 1})


def test_multiple_gt_and_predictions_match_once_by_descending_iou(tmp_path):
    # The broad prediction overlaps both GT at IoU=.5. The exact match must win
    # first, leaving the broad prediction to match the other GT exactly once.
    result = decompose(_export(tmp_path, [_gt(), _gt(_rect(x=100), cls=1)],
                               [_pred(_rect(width=200), cls=1), _pred(), _pred(_rect(x=150), cls=2)]))
    assert result == _expected({"0": {"ok": 1}, "1": {"mislocalized": 1}}, fp={"2": 1})


def test_one_prediction_cannot_match_two_gt(tmp_path):
    result = decompose(_export(tmp_path, [_gt(), _gt(_rect(x=100))], [_pred(_rect(width=200))]))
    assert result == _expected({"0": {"mislocalized": 1, "missed": 1}})


def test_duplicate_prediction_is_fp(tmp_path):
    assert decompose(_export(tmp_path, [_gt()], [_pred(), _pred()])) == _expected(
        {"0": {"ok": 1}}, fp={"0": 1})


def test_matching_is_class_agnostic_and_uses_iou_not_confidence(tmp_path):
    wrong = {**_pred(cls=1), "confidence": 0.25}
    same = {**_pred(_rect(width=75)), "confidence": 0.99}
    result = decompose(_export(tmp_path, [_gt()], [wrong, same]))
    assert result == _expected({"0": {"misclassified": 1}}, {"0>1": 1}, {"0": 1})


def test_self_intersecting_polygon_keeps_buffer_zero_repair(tmp_path):
    bowtie = [[0, 0], [100, 100], [0, 100], [100, 0]]
    repaired = _poly(bowtie)
    assert repaired.is_valid and repaired.area > 0
    result = decompose(_export(tmp_path, [_gt(bowtie)], [_pred(bowtie)]))
    assert result == _expected({"0": {"ok": 1}})


def test_zero_area_gt_excluded_and_zero_area_prediction_is_fp(tmp_path):
    line = [[0, 0], [50, 0], [100, 0]]
    result = decompose(_export(tmp_path, [_gt(line, cls=1), _gt()], [_pred(line, cls=2)]))
    assert result == _expected({"0": {"missed": 1}}, fp={"2": 1})


def test_missing_geometry_keeps_archived_exclusion_policy(tmp_path):
    result = decompose(_export(tmp_path, [_gt([])], [{"class_id": 1}, _pred([])]))
    assert result == _expected()


@pytest.mark.parametrize("points", [[0, 0, 1], [[0, 0], [1, 1]], [[0], [1], [2]],
                                         [[0, 0], [1, 1], [float("nan"), 2]]])
def test_malformed_polygon_rejected_explicitly(tmp_path, points):
    with pytest.raises(ValueError, match="polygon"):
        decompose(_export(tmp_path, [_gt()], [_pred(points)]))


@pytest.mark.parametrize(("gts", "preds", "summary", "expected"), [
    ([_gt()], [_pred()], "GT=1", _expected({"0": {"ok": 1}})),
    ([_gt()], [], "GT=1", _expected({"0": {"missed": 1}})),
    ([], [_pred()], "GT=0", _expected(fp={"0": 1})),
    ([], [], "GT=0", _expected()),
])
def test_cli_runs_on_synthetic_export_including_no_gt(tmp_path, gts, preds, summary, expected):
    path = _export(tmp_path, gts, preds)
    output = tmp_path / "output" / "decomposition.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "decompose_val_errors.py"
    proc = subprocess.run([sys.executable, str(script), str(output), f"synthetic={path}"],
                          capture_output=True, text=True, encoding="utf-8")
    assert proc.returncode == 0, proc.stderr
    assert summary in proc.stdout
    if not gts:
        assert "N/A" in proc.stdout
    assert json.loads(output.read_text(encoding="utf-8")) == {"synthetic": expected}
