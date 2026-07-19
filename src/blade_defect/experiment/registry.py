"""内置 YOLO baseline 实验注册表。"""
from .config import ExperimentConfig

EXPERIMENTS = [
    ExperimentConfig(name="exp001_yolov8n_seg_640", model="yolov8n-seg.pt"),
    ExperimentConfig(name="exp002_yolov8s_seg_640", model="yolov8s-seg.pt"),
    ExperimentConfig(name="exp003_yolo11n_seg_640", model="yolo11n-seg.pt"),
    ExperimentConfig(name="exp004_yolo11s_seg_640", model="yolo11s-seg.pt"),
    ExperimentConfig(name="exp005_yolov8n_seg_960", model="yolov8n-seg.pt", imgsz=960),
    ExperimentConfig(name="exp006_yolov8s_seg_960", model="yolov8s-seg.pt", imgsz=960),
    ExperimentConfig(name="exp007_yolo11n_seg_960", model="yolo11n-seg.pt", imgsz=960),
    ExperimentConfig(name="exp008_yolo11s_seg_960", model="yolo11s-seg.pt", imgsz=960),
    ExperimentConfig(name="exp009_yolov8n_seg_1024", model="yolov8n-seg.pt", imgsz=1024),
    ExperimentConfig(name="exp010_yolov8s_seg_1024", model="yolov8s-seg.pt", imgsz=1024),
    ExperimentConfig(name="exp011_yolo11n_seg_1024", model="yolo11n-seg.pt", imgsz=1024),
    ExperimentConfig(name="exp012_yolo11s_seg_1024", model="yolo11s-seg.pt", imgsz=1024),
    ExperimentConfig(name="exp013_yolov8n_seg_1280", model="yolov8n-seg.pt", imgsz=1280),
    ExperimentConfig(name="exp014_yolov8s_seg_1280", model="yolov8s-seg.pt", imgsz=1280),
    ExperimentConfig(name="exp015_yolo11n_seg_1280", model="yolo11n-seg.pt", imgsz=1280),
    ExperimentConfig(name="exp016_yolo11s_seg_1280", model="yolo11s-seg.pt", imgsz=1280),
]
