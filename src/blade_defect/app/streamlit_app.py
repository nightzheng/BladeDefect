"""Streamlit UI for blade defect segmentation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

from blade_defect.models import SegmentationPredictor
from blade_defect.utils.visualization import result_to_rgb

st.set_page_config(page_title="BladeDefect", page_icon="🛩️", layout="wide")
st.title("风机叶片缺陷检测")
st.caption("Ultralytics YOLO segmentation 可视化基础应用")

weights = st.sidebar.text_input("模型权重", "runs/segment/train/weights/best.pt")
confidence = st.sidebar.slider("置信度阈值", 0.0, 1.0, 0.25, 0.01)
iou = st.sidebar.slider("IoU 阈值", 0.0, 1.0, 0.70, 0.01)
device = st.sidebar.text_input("计算设备", "0", help="默认使用首张 GPU；无 GPU 时填写 cpu")
uploaded = st.file_uploader("上传巡检图像", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])

if uploaded:
    original = Image.open(uploaded).convert("RGB")
    left, right = st.columns(2)
    left.image(original, caption="原始图像", use_container_width=True)
    if st.button("开始检测", type="primary"):
        if not Path(weights).exists():
            st.error(f"模型权重不存在：{weights}")
        else:
            suffix = Path(uploaded.name).suffix or ".jpg"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
                temp.write(uploaded.getbuffer())
                temp_path = temp.name
            try:
                with st.spinner("正在检测..."):
                    predictor = SegmentationPredictor(weights)
                    results = predictor.predict(temp_path, conf=confidence, iou=iou, device=device)
                if results:
                    right.image(result_to_rgb(results[0]), caption="检测结果", use_container_width=True)
                    count = 0 if results[0].boxes is None else len(results[0].boxes)
                    st.success(f"检测完成，共发现 {count} 个目标实例。")
            finally:
                Path(temp_path).unlink(missing_ok=True)
