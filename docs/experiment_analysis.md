# 实验可视化与趋势分析

本模块汇总已有实验结果，不补造缺失指标。输入为 `results/summary.csv`、各实验目录下的
`metrics.json`，以及可选的混淆矩阵图片。

## 使用方法

```powershell
python scripts/analyze_experiments.py `
  --summary results/summary.csv `
  --runs-dir runs `
  --output-dir results/analysis `
  --publish-docs
```

固定输出包括：

- `model_comparison.csv`：模型、输入尺寸、mAP、Precision、Recall、F1、FPS；
- `per_class_metrics.csv`：从 `metrics.json` 的 `per_class_metrics` 或 `per_class` 字段提取；
- `input_size_map.png`、`input_size_fps.png`：输入尺寸趋势；
- `model_comparison.png`、`pr_curve.png`、`f1_curve.png`：模型指标对比；
- `confusion_matrix.png`：复制第一个真实实验矩阵；不存在时生成明确标注的占位图。

指定 `--publish-docs` 后，仅将报告引用的三张代表图复制到 `docs/assets/analysis/`：

- `input_size_map.png`
- `input_size_fps.png`
- `model_comparison.png`

仓库已运行一次 `ppt_reference_yolo11n_seg_640` smoke 实验，并用其真实结果覆盖该目录图片。
由于实验只有13张 PPT 参考图且 train/val 完全重合，这些图仅证明流程可运行，不能作为正式
baseline 结论。获得独立验证集上的真实指标后，应再次运行上述命令覆盖它们。

输入尺寸优先读取 `imgsz`、`image_size` 或 `input_size` 列，也可从实验名中的
`640/960/1024/1280` 推断。逐类数据缺失时只输出带表头的空 CSV。

## 趋势分析原则

比较时应同时关注 mAP50-95 和 FPS，避免仅凭单个指标选择模型。输入尺寸提升通常会增加
小目标可见性，但吞吐率可能下降；是否值得应由真实曲线确认。逐类 Precision/Recall 可用于
定位易混淆或漏检类别，混淆矩阵只使用真实评估产物。
