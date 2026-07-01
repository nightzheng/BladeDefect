# 自动化实验与论文分析

BladeDefect 的实验模块统一管理 YOLO segmentation baseline，复用现有训练与评估封装，
不会改变 `train`、`predict`、`evaluate` / `eval` 的行为。

## 运行全部 baseline

默认注册表包含 YOLOv8n、YOLOv8s、YOLO11n 和 YOLO11s 四组 640 像素实验：

```powershell
blade-defect experiment run-all
```

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
| L3 | 图片可被训练运行时解码 | 使用 OpenCV `imdecode` 检查，失败样本整对删除 |

L3 必须使用与 Ultralytics DataLoader 一致的 OpenCV 解码语义。某些文件虽然存在且能通过
Pillow 验证，但 OpenCV 仍可能返回 `None`；Ultralytics 会将这种情况报告为
`FileNotFoundError: Image Not Found`。

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
- `fixed_points` / `fixed_files`：自动 clamp 的 polygon 点和标注文件数；
- `removed_files`：因 polygon 或标注格式不可修复而删除的样本数；
- `decode_removed`：复制后解码复检失败而删除的样本数；
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
