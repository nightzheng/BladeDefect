"""内置 YOLO baseline 实验注册表。"""
from .config import ExperimentConfig

EXPERIMENTS = [
    ExperimentConfig(name="exp001_yolov8n_seg_640", model="yolov8n-seg.pt"),
    ExperimentConfig(name="exp002_yolov8s_seg_640", model="yolov8s-seg.pt"),
    ExperimentConfig(name="exp003_yolo11n_seg_640", model="yolo11n-seg.pt"),
    ExperimentConfig(name="exp004_yolo11s_seg_640", model="yolo11s-seg.pt"),
]
