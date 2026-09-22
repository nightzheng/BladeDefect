# BladeDefect

> 自动化 baseline 实验、结果汇总与论文图表说明见
> [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。
> 实验版本、数据哈希和运行环境追溯规则见 [docs/experiment_registry.md](docs/experiment_registry.md)。

面向风机叶片无人机巡检的 YOLO 缺陷检测工程。当前正式版本为 v3，支持
YOLO11 segmentation 与由 polygon 自动派生的 OBB baseline，并提供统一的训练、断点恢复、
扩轮、验证和结果追溯入口；RGB-T 配准与融合接口仍作为后续扩展保留。

当前正式数据与实验入口：

- segmentation：`datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml`
- OBB：`datasets/blade-v3-grouped-obb/data.yaml`
- 统一入口：`python scripts/run_v3_baselines.py status|run|stop|seal|plots|extend`
- 正式预算：YOLO11s @ 960，seg / OBB 上限各 200 epochs；达到平台期可提前封存
- 当前进度（2026-09-19）：seg 在 117/200 主动停止（最佳 epoch 89）；OBB e200 正在训练
- 经验收敛区间：seg 主收敛约 80–95、稳健停止约 105–120；OBB 暂估 80–120，需在 e60/e80 复核

v3 的完整执行说明见 [docs/experiments/v3_unified_runner.md](docs/experiments/v3_unified_runner.md)，
数据卡见 [docs/dataset_card_blade_v3.md](docs/dataset_card_blade_v3.md)。

## 环境安装

推荐使用独立的 Conda 环境，避免与其他项目的 Python 和依赖版本发生冲突。

### 推荐：Conda

```powershell
conda env create -f environment.yml
conda activate bladedefect
python -m pip install -e .
```

后续训练和测试统一调用激活环境中的 `python`，不要在命令或配置中写死 Conda 安装路径：

```powershell
python scripts/run_v3_baselines.py status
python scripts/run_v3_baselines.py run --task all --device 0
```

`environment.yml` 固定正式 v3 所需的 Ultralytics 版本；环境名称可在创建时覆盖，运行入口
不依赖环境名称、Windows 盘符或 Miniconda/Anaconda 的安装位置。

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

## v3 数据与标注格式

正式 v3 数据集共 48,291 张图片，按分组规则固定为 train 33,804、val 7,244、
test 7,243；三份清单互不重叠。test 仅用于最终获批评估，不参与训练、best epoch、
阈值或模型选择。

segmentation 标注采用 Ultralytics YOLO-seg 格式：

当前数据集包含 15 类风机叶片缺陷（类别 ID 为 0-14），完整类别名称以
[`configs/data.yaml`](configs/data.yaml) 为准。类别按缺陷性质分为：

- 表面腐蚀：0-3
- 表面裂纹：4-5
- 表面缺陷：6-9
- 维修痕迹：10
- 叶片损伤：11-13
- 附件脱落：14

```text
datasets/
├── v3-index-rebuild/
│   ├── blade-v3-grouped-202608/  # seg 的 data.yaml 与固定 split 清单
│   └── blade-v2/                 # v3 清单使用的兼容数据入口，请勿单独删除
├── blade-v2-rebuild-v3-index/    # v3 seg 所需的底层图片联接与冻结标签
└── blade-v3-grouped-obb/         # v3 OBB 派生数据、清单与标签
```

每行标注格式为：

```text
class_id x1 y1 x2 y2 ... xn yn
```

坐标均归一化到 `[0, 1]`，每个多边形至少包含 3 个点。

OBB 标注由 v3 polygon 通过 `cv2.minAreaRect` 一对一派生，不等同于人工旋转框真值。
seg 与 OBB 只比较共有的 Box 指标，不比较 Mask mAP。详见
[docs/experiments/v3_obb_baseline.md](docs/experiments/v3_obb_baseline.md)。

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

## v3 正式数据版本

`blade-v3-grouped-202608` 是当前正式的 15 类无泄漏版本。它在旧版冻结样本基础上按图片哈希、
拍摄序列和类别覆盖进行确定性分组，形成 train / val / test 三份互斥清单；共 48,291 张图片、
48,291 个标签文件和 49,471 个实例，15 个细类与 6 个粗类在三个 split 中均有覆盖。

正式 seg 使用 `datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml`。该清单中的
`../blade-v2/...` 是位于 `datasets/v3-index-rebuild/` 内部的兼容入口，实际依赖
`datasets/blade-v2-rebuild-v3-index`，不是旧的顶层 `datasets/blade-v2` 数据集。清理磁盘时
必须保留这两个 v3 支撑目录。

```powershell
# 只读检查数据、环境、训练状态与断点恢复能力
python scripts/run_v3_baselines.py status

# 租用 Windows/Linux 算力节点开跑前：全量核对环境、CUDA、权重和 train/val 引用
python scripts/run_v3_score_sweep.py preflight --require-cuda

# 已封存基线的状态检查（seg 117/200，OBB 82/200）
python scripts/run_v3_baselines.py run --task seg --device 0
python scripts/run_v3_baselines.py run --task obb --device 0

# 顺序处理两项：完成项跳过，中断项从 last.pt 恢复
python scripts/run_v3_baselines.py run --task all --device 0

# 平台期主动停止；进程已停则用 seal 封存并关闭自动恢复
python scripts/run_v3_baselines.py stop --task seg --reason "validation plateau"
python scripts/run_v3_baselines.py seal --task seg --reason "validation plateau"

# 补生成 YOLO 原生曲线；完整 PR/F1/混淆矩阵需去掉 --training-only 并指定 GPU
python scripts/run_v3_baselines.py plots --task seg --training-only

# 平台期后的提分实验：先看计划，再按 stage 1 顺序执行，完成后自动汇总
python scripts/run_v3_score_sweep.py plan --stage all
python scripts/run_v3_score_sweep.py run --task all --stage 1 --device 0
python scripts/run_v3_score_sweep.py analyze
```

当前同口径从头训练的建议预算为 seg 110 epochs、OBB 90 epochs；200 仅是已封存
基线的历史预算上限。提分实验统一从各自 `best.pt` 开新目录，优先比较低学习率
AdamW/cosine 与 1280 小目标方案，不继续恢复平台期基线。

Linux/云 GPU 的数据同步、环境安装、多节点拆分和校验命令见
[`docs/linux_v3_training.md`](docs/linux_v3_training.md)。不要把本机盘符、运行时
`normalized_data.yaml` 或 Ultralytics cache 复制到另一操作系统。

旧版 v2 仅用于历史实验追溯，不再作为正式训练入口。历史说明仍保留在
`docs/dataset_version.md` 与旧实验配置中；当前使用者应以 v3 数据卡、统一 runner 文档和
`configs/v3_baselines.yaml` 为准。

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
