import csv
from pathlib import Path

from blade_defect.experiment.metadata import sha256_file
from scripts.build_stage_chart_data import (
    box_iou,
    match_predictions,
    polygon_envelope,
    reliability_bins,
    sweep_pr_f1,
)
from scripts.export_obb_val_predictions import (
    normalized_polygon_to_envelope,
    obb_points_to_envelope,
    parse_obb_label,
)
from scripts.generate_v3_charts import (
    render_confidence_reliability,
    render_f1_curve,
    render_pr_curve,
    render_training_curves,
)


def test_polygon_envelope_maps_normalized_polygon_to_pixels() -> None:
    polygon = [0.1, 0.2, 0.5, 0.2, 0.5, 0.8, 0.1, 0.8]
    assert polygon_envelope(polygon, 1000, 2000) == (100.0, 400.0, 500.0, 1600.0)


def test_box_iou_identity_disjoint_partial() -> None:
    a = (0.0, 0.0, 10.0, 10.0)
    assert box_iou(a, a) == 1.0
    assert box_iou(a, (20.0, 20.0, 30.0, 30.0)) == 0.0
    overlap = box_iou(a, (5.0, 5.0, 15.0, 15.0))
    assert abs(overlap - 25.0 / 175.0) < 1e-9


def _fixture_samples() -> list[dict]:
    return [
        {
            "gt": [(0, (0.0, 0.0, 10.0, 10.0)), (1, (20.0, 20.0, 30.0, 30.0))],
            "preds": [
                (0, 0.9, (0.0, 0.0, 10.0, 10.0)),
                (0, 0.8, (100.0, 100.0, 110.0, 110.0)),
                (1, 0.5, (21.0, 21.0, 31.0, 31.0)),
                (2, 0.4, (20.0, 20.0, 30.0, 30.0)),
            ],
        }
    ]


def test_match_predictions_class_aware_and_single_match() -> None:
    tp, fp, fn = match_predictions(_fixture_samples(), conf_threshold=0.0)
    assert (tp, fp, fn) == (2, 2, 0)
    tp, fp, fn = match_predictions(_fixture_samples(), conf_threshold=0.85)
    assert (tp, fp, fn) == (1, 0, 1)


def test_match_predictions_second_pred_cannot_steal_matched_gt() -> None:
    samples = [
        {
            "gt": [(0, (0.0, 0.0, 10.0, 10.0))],
            "preds": [(0, 0.9, (0.0, 0.0, 10.0, 10.0)), (0, 0.8, (1.0, 1.0, 11.0, 11.0))],
        }
    ]
    tp, fp, fn = match_predictions(samples, conf_threshold=0.0)
    assert (tp, fp, fn) == (1, 1, 0)


def test_sweep_pr_f1_tracks_threshold_monotonic_recall() -> None:
    sweep = sweep_pr_f1(_fixture_samples(), gt_total=2)
    by_threshold = {row["confidence_threshold"]: row for row in sweep}
    assert set(by_threshold) == {0.9, 0.8, 0.5, 0.4}
    low = by_threshold[0.4]
    assert (low["tp"], low["fp"], low["fn"]) == (2, 2, 0)
    assert low["recall"] == 1.0 and abs(low["precision"] - 0.5) < 1e-9
    high = by_threshold[0.9]
    assert (high["tp"], high["fp"], high["fn"]) == (1, 0, 1)
    assert high["precision"] == 1.0 and high["recall"] == 0.5


def test_reliability_bins_counts_and_ece() -> None:
    rows, ece = reliability_bins(_fixture_samples(), conf_min=0.25, bins=3)
    assert [row["count"] for row in rows] == [1, 1, 2]
    top = rows[2]
    assert abs(top["mean_confidence"] - 0.85) < 1e-6
    assert top["accuracy"] == 0.5
    expected_ece = (1 / 4) * abs(0.0 - 0.4) + (1 / 4) * abs(1.0 - 0.5) + (2 / 4) * abs(0.5 - 0.85)
    assert abs(ece - round(expected_ece, 6)) < 1e-6


def test_parse_obb_label_reads_four_point_polygons(tmp_path: Path) -> None:
    label = tmp_path / "a.txt"
    label.write_text(
        "3 0.1 0.2 0.3 0.2 0.3 0.4 0.1 0.4\n"
        "bad line\n"
        "4 0.5 0.5 0.6 0.5 0.6 0.6 0.5 0.6\n",
        encoding="utf-8",
    )
    names = {3: "cls3", 4: "cls4"}
    instances = parse_obb_label(label, names)
    assert [item["class_id"] for item in instances] == [3, 4]
    assert instances[0]["class_name"] == "cls3"
    assert instances[0]["polygon"] == [0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.1, 0.4]
    assert parse_obb_label(tmp_path / "missing.txt", names) == []


def test_obb_points_to_envelope_and_normalized_variant() -> None:
    points = [[4.0, 2.0], [8.0, 3.0], [7.0, 9.0], [3.0, 8.0]]
    assert obb_points_to_envelope(points) == [3.0, 2.0, 8.0, 9.0]
    envelope = normalized_polygon_to_envelope([0.1, 0.2, 0.4, 0.2, 0.4, 0.6, 0.1, 0.6], 100, 50)
    assert envelope == [10.0, 10.0, 40.0, 30.0]


def _write_curve_csvs(data_dir: Path) -> None:
    data_dir.mkdir(parents=True)
    with (data_dir / "pr_curve.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment_id", "confidence_threshold", "precision", "recall", "tp", "fp", "fn", "source"])
        for experiment_id in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50"):
            writer.writerow([experiment_id, 0.9, 1.0, 0.2, 1, 0, 4, "fixture"])
            writer.writerow([experiment_id, 0.3, 0.5, 0.8, 4, 4, 1, "fixture"])
    with (data_dir / "f1_curve.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment_id", "confidence_threshold", "f1", "precision", "recall", "source"])
        for experiment_id in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50"):
            writer.writerow([experiment_id, 0.9, 0.33, 1.0, 0.2, "fixture"])
            writer.writerow([experiment_id, 0.3, 0.62, 0.5, 0.8, "fixture"])
    with (data_dir / "training_curves.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment_id", "task", "epoch", "metric", "value", "source"])
        for experiment_id in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50", "v3_hier_coarse_yolo11s_seg_960_e50"):
            writer.writerow([experiment_id, "segment", 1, "metrics/mAP50(B)", 0.1, "fixture"])
            writer.writerow([experiment_id, "segment", 2, "metrics/mAP50(B)", 0.2, "fixture"])
            writer.writerow([experiment_id, "segment", 1, "train/box_loss", 2.0, "fixture"])
            writer.writerow([experiment_id, "segment", 2, "train/box_loss", 1.5, "fixture"])
    with (data_dir / "confidence_reliability.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["experiment_id", "bin_index", "bin_low", "bin_high", "count", "mean_confidence", "accuracy", "ece", "source"])
        for experiment_id in ("v3_yolo11s_seg_960_e100", "v3_yolo11s_obb_960_e50"):
            writer.writerow([experiment_id, 0, 0.25, 0.625, 10, 0.4, 0.5, 0.1, "fixture"])
            writer.writerow([experiment_id, 1, 0.625, 1.0, 0, 0.0, 0.0, 0.1, "fixture"])


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def test_figure_renderers_emit_valid_pngs_and_skip_empty_bins(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "figures"
    output_dir.mkdir()
    _write_curve_csvs(data_dir)

    meta_pr = render_pr_curve(_load_csv(data_dir / "pr_curve.csv"), output_dir / "PR_curve.png")
    meta_f1 = render_f1_curve(_load_csv(data_dir / "f1_curve.csv"), output_dir / "F1_curve.png")
    meta_train = render_training_curves(_load_csv(data_dir / "training_curves.csv"), output_dir / "training_curves.png")
    meta_rel = render_confidence_reliability(_load_csv(data_dir / "confidence_reliability.csv"), output_dir / "confidence_reliability.png")

    for name in ("PR_curve.png", "F1_curve.png", "training_curves.png", "confidence_reliability.png"):
        blob = (output_dir / name).read_bytes()
        assert blob[:8] == b"\x89PNG\r\n\x1a\n"
        assert len(blob) > 10_000
    assert meta_pr["plotted_points"]["v3_yolo11s_seg_960_e100"] == 2
    assert meta_f1["peak"]["v3_yolo11s_obb_960_e50"]["f1"] == 0.62
    assert meta_train["plotted_points"]["v3_hier_coarse_yolo11s_seg_960_e50"]["metrics/mAP50(B)"] == 2
    assert meta_rel["plotted_points"]["v3_yolo11s_seg_960_e100"] == 1
    assert meta_rel["ece"]["v3_yolo11s_seg_960_e100"] == 0.1
    assert sha256_file(data_dir / "pr_curve.csv") == sha256_file(data_dir / "pr_curve.csv")
