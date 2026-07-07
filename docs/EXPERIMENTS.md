# 自动化实验与论文分析

BladeDefect 的实验模块统一管理 YOLO segmentation baseline，复用现有训练与评估封装，
不会改变 `train`、`predict`、`evaluate` / `eval` 的行为。

## 运行全部 baseline

默认注册表包含 YOLOv8n、YOLOv8s、YOLO11n 和 YOLO11s，并分别运行
640、960、1024、1280 四档输入尺寸，共 16 组实验：

| 实验编号 | 输入尺寸 | 模型 |
|---|---:|---|
| exp001–exp004 | 640 | YOLOv8n / YOLOv8s / YOLO11n / YOLO11s |
| exp005–exp008 | 960 | YOLOv8n / YOLOv8s / YOLO11n / YOLO11s |
| exp009–exp012 | 1024 | YOLOv8n / YOLOv8s / YOLO11n / YOLO11s |
| exp013–exp016 | 1280 | YOLOv8n / YOLOv8s / YOLO11n / YOLO11s |

`run-all` 的输入尺寸由 `src/blade_defect/experiment/registry.py` 中每个实验的
`ExperimentConfig.imgsz` 决定，并覆盖 `train.yaml` 的同名参数。修改 `train.yaml` 中的
`imgsz` 只影响单独执行的 `blade-defect train`，不会批量改变已注册实验的尺寸。

```powershell
blade-defect experiment run-all
```

只运行一种输入尺寸对应的四个模型，使用 `--imgsz`：

```powershell
# 只运行 YOLOv8n、YOLOv8s、YOLO11n、YOLO11s 的 960 实验
blade-defect experiment run-all --imgsz 960 --device 0
```

可选尺寸为 `640`、`960`、`1024`、`1280`。不传 `--imgsz` 时运行注册表中的
全部 16 组实验。该参数属于本次批量实验的筛选条件，因此不写入 `train.yaml`；
`train.yaml` 的 `imgsz` 仍只影响单独执行的 `blade-defect train`。

按完整实验名称或实验 ID 精确选择时，可重复传入 `--experiment`：

```powershell
# 完整名称与短 ID 可以混用
blade-defect experiment run-all `
  --experiment exp014_yolov8s_seg_1280 `
  --experiment exp016 `
  --device 0
```

数字 ID 也可省略 `exp` 前缀，例如 `--experiment 14 --experiment 16`。
同时传入 `--imgsz` 和 `--experiment` 时会取两者交集；名称或 ID 不存在时，
命令会在数据检查和训练开始前直接报错。

也可以覆盖数据、设备和输出位置：

```powershell
blade-defect experiment run-all `
  --config configs/train.yaml `
  --device 0 `
  --runs-dir runs `
  --output results/summary.csv
```

Linux shell 使用反斜杠 `\` 续行，或将命令写在同一行。

`run-all` 统一读取 `configs/train.yaml`，所有实验共用其中引用的原始 dataset YAML；
该流程不会生成临时 dataset YAML。每个实验输出到 `runs/<experiment_name>/`，最终指标写入 `metrics.json`。

训练启动前会对 train/val 自动执行 strict label validation gate。存在缺失图片、缺失标注、
非法 polygon 或数据目录缺失时，`run-all` 会在加载模型前终止。仅在明确了解风险时跳过：

```powershell
blade-defect experiment run-all --skip-validation
```
完成全部实验后自动生成 `results/summary.csv`，字段为模型、mAP50、mAP50-95、
Precision、Recall 和 FPS。已有结果可以重新汇总：

```powershell
blade-defect experiment summary
```

## 生成论文图表

```powershell
blade-defect experiment analyze
```

自定义输入和输出路径：

```powershell
blade-defect experiment analyze `
  --summary results/summary.csv `
  --runs-dir runs `
  --output-dir results/analysis
```

分析器只使用 Matplotlib，并配置 Windows、Linux 常见中文字体 fallback。默认生成：

- `summary_plot.png`：mAP50、mAP50-95、FPS 及 FPS–accuracy 权衡四面板图；
- `pr_curve.png`：各模型 Precision / Recall 对比；
- `loss_curve_comparison.png`：各实验 train/val loss 与 mAP 曲线；
- `class_distribution.png`：存在类别统计时生成。

训练曲线优先读取 `runs/*/metrics.json` 的 `curves` 数据，同时兼容 Ultralytics
自动产生的 `runs/*/results.csv`。旧实验若只有最终标量指标，仍会生成带缺失数据说明的
`loss_curve_comparison.png`。类别分布仅在 `metrics.json` 包含如下可选数据时生成：

```json
{
  "class_distribution": {
    "crack": 120,
    "corrosion": 80
  }
}
```

## 输出目录

```text
runs/
├── exp001_yolov8n_seg_640/
│   ├── metrics.json
│   └── results.csv
└── ...
results/
├── summary.csv
└── analysis/
    ├── summary_plot.png
    ├── pr_curve.png
    ├── loss_curve_comparison.png
    └── class_distribution.png  # 有类别统计时
```

## 生成可直接训练的数据集

YOLO 数据集需要同时满足三层正确性：

| 层级 | 检查内容 | 生成器行为 |
|---|---|---|
| L1 | 文件路径存在 | 缺失 image 或 label 的样本不进入抽样池 |
| L2 | image/label 配对与 polygon 合法 | 轻微越界自动 clamp，不可修复样本整对删除 |
| L3 | 图片可被训练运行时解码 | 抽样后惰性使用 OpenCV `imdecode`，失败时自动补位 |

L3 必须使用与 Ultralytics DataLoader 一致的 OpenCV 解码语义。某些文件虽然存在且能通过
Pillow 验证，但 OpenCV 仍可能返回 `None`；Ultralytics 会将这种情况报告为
`FileNotFoundError: Image Not Found`。

为避免在 4.3 万张源图片上执行全量重解码，生成器先建立轻量候选池，再随机打乱候选，
只解码到目标数量满足为止。遇到坏图会继续检查下一候选并自动补位，因此通常只需解码
约 `train_count + val_count` 张，而不是解码整个源数据集。该策略不会用 1% 抽检替代完整性：
所有最终写入的数据都已经通过 OpenCV 解码。

从原始数据重新生成 blade-v2：

```powershell
python scripts/create_small_dataset.py `
  --source D:\pictures\bladeYoloData0721 `
  --output datasets/blade-v2 `
  --train-count 4000 `
  --val-count 1000 `
  --seed 42
```

输出目录必须为空。生成统计中的关键字段：

- `missing_images` / `missing_labels`：源数据双向配对缺失数；
- `corrupt_images`：OpenCV 无法解码而被拒绝的图片数；
- `decode_checked`：为凑满目标数量实际执行 OpenCV 解码的候选数；
- `fixed_points` / `fixed_files`：自动 clamp 的 polygon 点和标注文件数；
- `removed_files`：因 polygon 或标注格式不可修复而删除的样本数；
- `decode_removed`：兼容统计字段；惰性解码架构下通常为 0；
- `copied`：最终通过全部检查、实际写入的数据量。

若重建前手动保留了旧目录，需要清除 Ultralytics 索引缓存；正常删除整个输出目录后重建时，
cache 会随目录一起消失：

```powershell
Remove-Item datasets/blade-v2/labels/train.cache -ErrorAction SilentlyContinue
Remove-Item datasets/blade-v2/labels/val.cache -ErrorAction SilentlyContinue
```

`run-all` 的 strict validation gate 仍作为防御层，用于发现训练前被外部修改的数据；它不能替代
生成阶段的 OpenCV decode gate。

## 验证

```powershell
python -m pytest
python -m compileall -q src tests
```
