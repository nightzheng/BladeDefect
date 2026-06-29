# BladeDefect

面向风机叶片无人机巡检的 YOLO segmentation 缺陷检测工程模板。项目支持
Ultralytics YOLOv8/YOLO11 的训练、验证与推理，并为后续 RGB-T 配准和融合预留接口。

## 环境安装

推荐使用独立的 Conda 环境，避免与其他项目的 Python 和依赖版本发生冲突。

### 推荐：Conda

```powershell
conda env create -f environment.yml
conda activate bladedefect
python -m pip install -e .
```

### 备选：pip

如不使用 Conda，可保留并使用原有的 `requirements.txt` 安装方式：

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

Conda 环境固定使用 Python 3.10。默认配置使用 `device: auto`：CUDA 可用时选择首张 GPU
（`device=0`），否则自动回退到 CPU。也可通过 `--device 0`、`--device cpu`，或在配置文件中
设置 `device: 0` / `device: cpu` 显式指定。

`environment.yml` 不强行锁死 CUDA 版 PyTorch。GPU 用户应根据本机驱动和 CUDA 支持安装匹配的
PyTorch；完整的 Windows、Linux、VSCode 和学校算力服务器配置说明见
[docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

## 跨平台路径与配置

项目支持 Windows 和 Linux 的相对路径与绝对路径。CLI 收到的文件路径会统一展开并解析，配置路径
则以配置文件中的 `project_root` 为基准；相对的 `project_root` 以配置文件所在目录为基准。例如：

```yaml
project_root: ..
data: configs/data.yaml
project: runs/segment
device: auto
```

`data.yaml` 中的 `path` 以该 YAML 文件所在目录为基准。运行时会生成临时的规范化配置，将数据集
根目录转换为绝对路径和正斜杠，避免工作目录或 Windows 反斜杠影响 Ultralytics。

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
# 查看 CLI 帮助
blade-defect --help

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

# evaluate 的兼容短别名
blade-defect eval --model runs/segment/train/weights/best.pt --data configs/data.yaml

# 推理
blade-defect predict --weights runs/.../best.pt --source path/to/images

# 启动可视化应用
streamlit run src/blade_defect/app/streamlit_app.py

# 执行示例消融实验
blade-defect ablation --config configs/ablation.yaml
```

推理命令同时兼容原有的 `--model` 参数。`train`、`predict`、`evaluate` / `eval` 和 `ablation`
均可使用 `--device auto`、`--device 0` 或 `--device cpu`。

## 测试与检查

```powershell
python -m pytest
python -m compileall src
blade-defect --help
blade-defect evaluate --help
blade-defect eval --help
```

## 工程结构

```text
BladeDefect/
├── configs/                  # 数据、训练和消融实验配置
├── docs/                     # 跨平台环境说明
├── scripts/                  # 独立脚本入口
├── src/blade_defect/
│   ├── app/                  # Streamlit 检测可视化
│   ├── data/                 # 清洗、标注检查、数据集划分
│   ├── evaluation/           # 指标与自动消融实验
│   ├── fusion/               # RGB-T 配准/融合抽象接口
│   ├── models/               # Ultralytics 训练与推理封装
│   └── utils/                # 路径、设备、日志和可视化工具
└── tests/                    # 不依赖模型权重的基础测试
```

配置中的 `model` 可使用 `yolov8n-seg.pt` 或 `yolo11n-seg.pt`。首次使用预训练权重时，
Ultralytics 可能自动联网下载文件。
