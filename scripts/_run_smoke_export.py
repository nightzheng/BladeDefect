from blade_defect.experiment.prediction_exporter import export_validation_predictions

export_validation_predictions(
    model_path="runs/hier_coarse_yolo11s_seg_960_smoke/weights/best.pt",
    data_yaml="datasets/blade-v2-6class/data.yaml",
    output_path="runs/hier_coarse_yolo11s_seg_960_smoke/validation_predictions.json",
    experiment_id="hier_coarse_yolo11s_seg_960_smoke",
    imgsz=960,
    device="0",
)
print("EXPORT_DONE")
