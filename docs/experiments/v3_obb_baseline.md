# v3 OBB 正式 baseline 执行说明

本文记录正式 v3（blade-v3-grouped-202608）OBB 派生数据激活、真实 smoke 与 200 epochs 正式 baseline 的执行链路与口径。转换/校验/训练入口复用负责人既有实现（`convert_seg_to_obb.py`、`check_obb_dataset.py`、`visualize_obb_labels.py`、`run_obb_smoke.py`），不重新实现转换算法。

## 口径声明

- **自动 OBB 来自 polygon 最小外接矩形**（`cv2.minAreaRect`，角点按"顶部最左起顺时针"排序），**不等同于人工旋转框真值**；所有 OBB 指标均为该自动标注口径。
- OBB 与 seg 只比较**共有 Box 指标**（Precision/Recall/mAP50/mAP50-95）、逐类 AP 与速度，**不比较 Mask mAP**（OBB 无 Mask 分支）。
- test 保持锁定：仅参与转换与数据完整性校验，不用于训练、验证、best epoch、阈值选择或预测样例选择；本周不运行任何 test 评估。
- smoke（1 epoch）与正式 baseline（200 epochs）分开标记，smoke 指标不得冒充正式结果。

当前状态（2026-09-22）：正式运行目录为 `v3_yolo11s_obb_960_e200`，200 epochs 是
预算上限；该运行已在 82/200 主动封存，最佳 e81，Box mAP50-95=0.17340。
最后 10 轮相对前 10 轮均值仅 +0.00124，最后 5 轮仅 +0.00098，验证损失也基本
持平，因此判断主收敛区间为 e70–e82，未来同口径从头训练建议 90 epochs。

## 数据身份

| 项 | 值 |
|---|---|
| parent dataset | blade-v3-grouped-202608（48,291 样本：train 33,804 / val 7,244 / test 7,243） |
| derived dataset | blade-v3-grouped-obb（`datasets/blade-v3-grouped-obb`） |
| membership_sha256 | 与 parent 完全一致（一对一派生，不重新划分，见 `dataset_manifest.json`） |
| split 继承 | train/val/test 逐 split identity 与 parent 完全相同，无跨 split 移动 |
| test 策略 | `allowed_for_training/model_selection/threshold_selection = false` |

派生方式为 txt 清单驱动：图像硬链接（不复制）、标签逐实例 polygon→OBB 转换；失败策略为"只丢弃失败实例，保留图像与同图其他实例"，全部 warning/error 记录在 `results/obb_v3/invalid_obb_labels.csv`。

> 注意：转换器按可移植契约写 `path: .`，而 Ultralytics 将相对 `path` 按**当前工作目录**解析。交给 Ultralytics 训练/验证前必须绝对化——smoke 复用 `resolved_data_yaml`（与 seg 正式实验同一链路）；正式 baseline 在 `runs/v3_yolo11s_obb_960_e200/normalized_data.yaml` 落盘稳定副本（train/val 清单条目绝对化、不含 test 键），断点续训时 checkpoint 记录的同一路径仍然有效。

## 执行链路

```powershell
# 0) 只读预检（不写任何文件；valid=false 必须先处理数据问题）
python scripts/convert_seg_to_obb.py --source-data configs/data.yaml --dry-run-index-check

# 1) 全量转换（48,291 张；已存在时拒绝覆盖，除非显式 --overwrite）
python scripts/convert_seg_to_obb.py --source-data configs/data.yaml `
  --output datasets/blade-v3-grouped-obb --results results/obb_v3

# 2) 全量 validator（含 test 完整性；valid=false 禁止进入训练）
python scripts/check_obb_dataset.py --dataset datasets/blade-v3-grouped-obb `
  --report results/obb_v3/validation_report.json

# 3) 并排预览（train/val/test 各 30 张，验收抽样覆盖 12/13/14、细长、倾斜、贴边等特征）
python scripts/visualize_obb_labels.py --source-seg datasets/blade-v3-grouped-202608 `
  --source-obb datasets/blade-v3-grouped-obb --output results/obb_v3/conversion_preview `
  --count-per-split 30 --seed 42 --acceptance results/obb_v3/preview_review.csv

# 或一步执行 0-3（任一失败即中止）
python scripts/activate_v3_obb_pipeline.py

# 4) 真实 1 epoch smoke（先数据门禁，再官方权重溯源核验，再训练→验证→预测）
python scripts/run_obb_smoke.py --config configs/experiments/obb_yolo11s_960_v3_smoke.yaml

# 5) batch 真实显存校准（短探针训练；不写正式 run）
python scripts/run_v3_obb_baseline.py --config configs/experiments/v3_yolo11s_obb_960_e200.yaml `
  --calibrate-only --calibrate-batch 8

# 6) 正式 200 epochs baseline（自动断点续训；每 5 轮 checkpoint）
python scripts/run_v3_baselines.py run --task obb --device 0

# 7) 与 15 类 seg 的共有 Box 指标对比（OBB 未完成时只输出状态，不编造数值）
python scripts/compare_obb_seg_common_metrics.py `
  --obb-run runs/v3_yolo11s_obb_960_e200 `
  --seg-run runs/v3_yolo11s_seg_960_e200
```

## 权重溯源

- 仅允许官方 Ultralytics `yolo11s-obb.pt`，来源固定为
  `https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s-obb.pt`；
- `results/obb_v3/weight_provenance.json` 记录来源、大小、SHA-256 与 `verified` 状态；
  缓存权重必须匹配已验证溯源才允许复用，拒绝任何来源不明权重；
- smoke 与正式 baseline 共用同一权重与同一溯源记录。

## 训练配置口径

| 参数 | smoke | 正式 baseline |
|---|---|---|
| task / model | obb / yolo11s-obb.pt | obb / yolo11s-obb.pt |
| imgsz | 960 | 960 |
| epochs | 1 | 历史预算上限 200；实际 e82 封存，未来建议 90 + `patience=15–20` |
| batch | 4（继承负责人既有 smoke 配置） | 8（按真实显存探针校准，见 `results/obb_v3/batch_calibration.json`） |
| workers | 2 | 4 |
| device / seed | 0 / 42 | 0 / 42 |
| pretrained / cache / amp | true / false / true | true / false / true |
| save_period | —（单轮） | 5（周期 checkpoint + last.pt 断点续训） |

## 工件与可追溯性

正式 baseline `runs/v3_yolo11s_obb_960_e200/`：

- `environment.json`：与 seg 正式实验同一 schema（`collect_environment_metadata`）；
- `run_manifest.json`：experiment_id、dataset_id、parent_dataset_id、code_commit、hardware、
  config、command、started/finished_at、resume_supported、weight_provenance、工件 SHA-256；
- `metrics.json`：Box P/R/mAP50/mAP50-95/F1、FPS（`ultralytics_val_speed_inference_ms`）、
  train_seconds、seconds_per_epoch、test_used=false；
- `per_class_metrics.csv`：box 分支逐类 P/R/AP50/AP50-95；
- `predictions/`：按类别覆盖选取的代表性 val 预测图；
- `weights/`：best.pt、last.pt、每 5 轮 checkpoint。

该基线已标记 `stopped_early` 且关闭自动恢复；重新执行第 6 步会安全跳过。提分实验使用
`scripts/run_v3_score_sweep.py` 从 e81 `best.pt` 新建独立目录，不覆盖该基线。

## 结果归档（交付分析负责人）

以下归档说明针对既有历史 15 类 e100 与 6 类 e50 结果；新的 e200 主实验完成后应以
e200 运行目录生成独立归档，不覆盖历史工件。`scripts/archive_experiment_assets.py` 为历史两项
seg 正式实验生成：

- `results/v3_experiment_handoff/experiment_inventory.csv`：experiment_id、dataset_id、
  label_level、code_commit、关键指标、环境、资产统计；
- `results/v3_experiment_handoff/artifact_sha256.csv`：全部文件 SHA-256，供复核防漂移；
- `results/v3_experiment_handoff/handoff_validation.json`：五件套、身份、环境、membership、
  test 锁定的结构化校验；
- `results/v3_experiment_handoff/handoff_notes.md`：交付说明。

## 验证

```powershell
pytest
python -m compileall src scripts
git diff --check
```
