# BladeDefect

面向风机叶片无人机巡检的 YOLO segmentation 缺陷检测工程模板。项目支持
Ultralytics YOLOv8/YOLO11 的训练、验证与推理，并为后续 RGB-T 配准和融合预留接口。

## 环境安装

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

建议使用 Python 3.10+。如需 GPU，请根据本机 CUDA 环境先安装对应版本的 PyTorch。
训练、验证、推理和可视化应用默认使用首张 GPU（`device=0`）；如需使用 CPU，
请在配置文件中设置 `device: cpu`，或在命令行中传入 `--device cpu`。

## 数据格式

数据集采用 Ultralytics YOLO-seg 格式：

```text
datasets/blade/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

每行标注格式为：

```text
class_id x1 y1 x2 y2 ... xn yn
```

坐标均归一化到 `[0, 1]`，每个多边形至少包含 3 个点。

## 常用命令

```powershell
# 检查标注
blade-defect check-labels --images datasets/raw/images --labels datasets/raw/labels

# 扫描损坏图像、重复图像和空标注（不会自动删除数据）
blade-defect clean --images datasets/raw/images --labels datasets/raw/labels

# 划分数据集（复制文件）
blade-defect split --images datasets/raw/images --labels datasets/raw/labels --output datasets/blade

# 训练
blade-defect train --config configs/train.yaml

# 验证并输出指标 JSON
blade-defect evaluate --model runs/segment/train/weights/best.pt --data configs/data.yaml

# 推理
blade-defect predict --model runs/segment/train/weights/best.pt --source assets/demo.jpg

# 启动可视化应用
streamlit run src/blade_defect/app/streamlit_app.py

# 执行示例消融实验
blade-defect ablation --config configs/ablation.yaml
```

## 工程结构

```text
BladeDefect/
├── configs/                  # 数据、训练和消融实验配置
├── scripts/                  # 独立脚本入口
├── src/blade_defect/
│   ├── app/                  # Streamlit 检测可视化
│   ├── data/                 # 清洗、标注检查、数据集划分
│   ├── evaluation/           # 指标与自动消融实验
│   ├── fusion/               # RGB-T 配准/融合抽象接口
│   ├── models/               # Ultralytics 训练与推理封装
│   └── utils/                # 日志、文件和可视化工具
└── tests/                    # 不依赖模型权重的基础测试
```

配置中的 `model` 可使用 `yolov8n-seg.pt` 或 `yolo11n-seg.pt`。首次使用预训练权重时，
Ultralytics 可能自动联网下载文件。
