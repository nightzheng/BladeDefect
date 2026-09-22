# 全量训练计划（full training plan）

> 历史 v2 计划：本页的 50/100 epochs 结论用于旧采样与旧全量实验追溯，不是当前
> v3 seg/OBB 的训练计划。当前口径见 `v3_training_plan.md` 与 `v3_unified_runner.md`。

## 1. 数据版本

| 用途 | dataset_id | 规模 | 说明 |
|---|---|---|---|
| 全量主 baseline | `blade-v2`（即 `blade-v2-full-frozen-48291`） | train 43,463 / val 4,828，49,471 实例，15 类 | 冻结版，txt 清单索引，门禁已通过 |
| 6 类 smoke | `blade-v2-sampled-4987-6class` | train 3,991 / val 996，5,099 实例，6 类 | 修复索引后的采样版 remap |
| 阶段性模型筛选 | `blade-v2-sampled-4987` | exp001—exp008 | 采样目录已被全量版覆盖，结果仅作阶段性参考 |

门禁校验结果：`results/dataset_review/full_gate_validation.json`、
`results/dataset_review/coarse6_gate_validation.json`。

## 2. 主 baseline 选择与启动

- 模型：`YOLO11s-seg`，imgsz=960，epochs=50，batch=8，seed=42，amp=True，
  cache=False，workers=4，save_period=5（周期性 checkpoint）。
- 选择依据：抽样 exp008（YOLO11s-seg@960）为八组采样实验中容量最大的
  配置；按任务约定优先启动 `YOLO11s-seg + 960 + 50 epochs`。
- 启动脚本：`scripts/run_full_primary.py --config configs/experiments/full_yolo11s_seg_960.yaml`
- 配置：`configs/experiments/full_yolo11s_seg_960.yaml`
- 断点续训：脚本自动检测 `runs/full_primary_yolo11s_seg_960/weights/last.pt`
  并 resume；`--no-resume` 强制从头训练。
- 吞吐校准与预计总时长：见 `results/training_calibration/estimated_training_time.md`。

## 3. 6 类 smoke

- 配置：`configs/experiments/hier_coarse_yolo11s_seg_960.yaml`
- 启动：`scripts/run_full_primary.py --config configs/experiments/hier_coarse_yolo11s_seg_960.yaml`
- 目的：验证 6 类标签实际被读取（修复前 images 目录指向全量数据导致
  39k 图片无 6 类标签，旧 0.316 结果无效，不进入正式汇总）。

## 4. 交付口径

遵循 `docs/experiment_metric_contract.md`：Mask/Box 分字段记录、
FPS 使用 `ultralytics_val_speed_inference_ms` 口径、逐类指标
`per_class_metrics.csv`（metric_branch ∈ {mask, box}）、逐样本/逐实例
原始预测 `validation_predictions.json`、每组实验 `run_manifest.json`。

## 5. 收敛性与 100-epoch 对照

抽样 exp008（50 epochs）曲线检查：best epoch=43/50（Mask mAP50-95=0.0693），
末 10 轮仍在波动上行（0.0588→0.0693，区间增幅约 +0.01），后半程最优
明显优于前半程（0.0693 vs 0.0526），50 epochs 结束时未收敛证据充分。
→ 建议将**单组** YOLO11s-seg@960 100-epoch 对照列为 P2 任务；
不立即重跑全部模型。注意该结论基于采样版数据，全量主 baseline 的
50 epochs 曲线出来后需重新评估。
