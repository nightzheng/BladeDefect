# BladeDefect

> 自动化 baseline 实验、结果汇总与论文图表说明见
> [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

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
PyTorch；国内网络环境可使用阿里云镜像安装固定的 CUDA 12.8 版本。完整命令及 Windows、Linux、
VSCode 和学校算力服务器配置说明见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md#pytorch-与-cuda)。

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

当前数据集包含 15 类风机叶片缺陷（类别 ID 为 0-14），完整类别名称以
[`configs/data.yaml`](configs/data.yaml) 为准。类别按缺陷性质分为：

- 表面腐蚀：0-3
- 表面裂纹：4-5
- 表面缺陷：6-9
- 维修痕迹：10
- 叶片损伤：11-13
- 附件脱落：14

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

# 生成类别、图像尺寸、bbox、mask、目标大小、异常标注和图表统计
python scripts/analyze_dataset.py --images D:\images --labels D:\labels --data configs\data.yaml --output results\dataset --filter-config configs\dataset_filter.yaml

# 划分数据集（复制文件）
blade-defect split --images datasets/raw/images --labels datasets/raw/labels --output datasets/blade --filter-config configs/dataset_filter.yaml

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

# 顺序运行全部 YOLO baseline，并生成 results/summary.csv
blade-defect experiment run-all

# 从已有 runs 重新生成汇总
blade-defect experiment summary

# 生成 results/analysis 下的论文级图表
blade-defect experiment analyze
```

`configs/dataset_filter.yaml` 按文件名记录数据质量处理决定：`exclude` 和 `review`
不会进入任何数据划分，`keep_negative` 会保留图片并在输出数据集中创建同名空标签，作为
无缺陷负样本使用。缺少多边形坐标的缺陷图片不会由程序自动补标。

`scripts/analyze_dataset.py` 只读取原始图片和标签。位于 `[-0.01, 1.01]` 的轻微越界坐标
会逐坐标记录，并只在统计内存中重置到 `[0,1]`；超出该区间的严重越界整行不计算几何面积，
等待排除或人工复查。`results/dataset/out_of_bounds_coordinates.csv` 记录具体坐标、点序号、
x/y位置、越界方向、越界幅度、类别和处理动作。数据划分和小样本生成会先复制数据，再将
soft error修正结果写入新数据集的同名标签，所有处理均不会回写原标签。未经人工确认的
hard error不会自动删除；生成流程会停止并保留问题样本，
待人工确认后修复，或通过 `dataset_filter.yaml` 明确排除。

## blade-v2 正式数据版本

第二周已冻结 `datasets/blade-v2`。该版本保留源train/val划分，通过 `train.txt`、`val.txt`
固定48,291张有效图片；修正后的48,291个标签实际保存在 `labels/train|val`。由于原图约
180.15GB且本地空间不足，`images/train|val` 使用NTFS目录联接只读访问 `D:\images`，
训练和统计必须使用清单及 `configs/dataset_filter.yaml`，不能直接枚举联接目录中的全部图片。

131张hard error图片已经逐图复核，全部记录为 `repair_confirmed`；202个hard行内坐标和
747个soft坐标只在新标签中重置到 `[0,1]`。严格校验结果为：49,471个实例，图片完整解码
失败、缺标签、孤立标签、非法类别、polygon点数异常和越界坐标均为0。

复现、哈希和存储说明见 `docs/dataset_version.md`；完整质量报告见 `docs/dataset_report.md`。

```powershell
# 正式版本统计（必须带同一filter）
python scripts\analyze_dataset.py `
  --images datasets\blade-v2\images `
  --labels datasets\blade-v2\labels `
  --data datasets\blade-v2\data.yaml `
  --output results\dataset_v2 `
  --filter-config configs\dataset_filter.yaml `
  --workers 16

# 6个上级类别分布
python scripts\analyze_class_hierarchy.py `
  --dataset datasets\blade-v2 `
  --hierarchy configs\class_hierarchy.yaml `
  --output results\hierarchy

# train/val泄漏候选扫描
python scripts\check_split_leakage.py `
  --dataset datasets\blade-v2 `
  --output results\dataset_review\split_leakage_review.csv `
  --workers 16
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
