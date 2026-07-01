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

## 验证

```powershell
python -m pytest
python -m compileall -q src tests
```
