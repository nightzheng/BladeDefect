# v3 无泄漏训练计划（blade-v3-grouped-202608）

状态（2026-09-22）：**v3 seg 已在 117/200 主动封存，派生 OBB 已在 82/200 主动封存**。
200 轮是单阶段训练预算上限，不是必须跑满的轮数；出现经滚动窗口确认的平台期时允许
使用统一入口主动停止。全程 test 锁定。

## 数据版本核对（任务 1）

接收 `blade-v3-grouped-202608` 后首先核对：

- `dataset_id`、manifest 生成时间、split 策略；
- `sample_lists_sha256` / `frozen_labels_sha256` 与负责人公布值一致；
- 图片/实例数量（预期 train 33,804 / val 7,244 / test 7,243，以 manifest 为准）；
- 运行 `assert_test_split_isolation`（`src/blade_defect/data/split_isolation.py`）
  确认 train/val/test 两两不相交，任何重叠即中止并上报。

## 禁止事项（任务 2）

- 禁止直接使用旧全量 `runs/full_primary_yolo11s_seg_960/weights/best.pt`
  在 v3 上续训后作为正式结果：旧模型在 blade-v2 全量 split 上训练，已见过
  部分新 val/test 样本（该 run 已标记 `split_leakage_known=true`）。
- test 不参与训练、验证、best epoch 选择或阈值选择；正式 test 评估等待
  负责人批准。

## 执行顺序

| 步骤 | 内容 | 命令 |
|---|---|---|
| 1 | 数据核对 + split 隔离校验 | `python -c "from blade_defect.data.split_isolation import assert_test_split_isolation; assert_test_split_isolation('datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml')"` |
| 2 | 1 epoch 吞吐校准（v3 规模） | `python scripts/run_full_primary.py --config configs/experiments/v3_yolo11s_seg_960_e200.yaml --epochs 1 --run-name v3_throughput_calibration` |
| 3 | seg 200 epochs 上限主实验（15 类；已在 e117 封存） | `python scripts/run_v3_baselines.py run --task seg --device 0` |
| 4 | OBB 200 epochs 上限主实验（15 类；已在 e82 封存） | `python scripts/run_v3_baselines.py run --task obb --device 0` |
| 5 | 6 类历史正式实验（50 epochs，保留作对照） | `python scripts/run_full_primary.py --config configs/experiments/v3_hier_coarse_yolo11s_seg_960_e50.yaml` |

15 类正式主实验使用 batch=8、seed=42、YOLO11s-seg@960、从官方预训练权重
启动并设置 `patience=0`，把停止权交给可审计的平台期判断，最大执行 200 轮；
checkpoint 每 5 轮保存，中断后直接重跑同一命令即可断点续训（自动检测
`weights/last.pt`）；中断检测用 `python scripts/check_run_status.py`。

## 当前收敛判断

| 任务 | 已观测证据 | 主收敛区间 | 建议停止/复核区间 | 置信度 |
|---|---|---|---|---|
| seg | 实际最佳 e89，e117 Mask mAP50-95=0.10414，最近 10 轮较前 10 轮均值仅 +0.00060；val seg loss 后段略回升 | e80–e95 | e105–e120；已于 e117 封存 | 高 |
| OBB | 实际最佳 e81，e82 Box mAP50-95=0.17340；最后 10 轮较前 10 轮均值仅 +0.00124，最后 5 轮仅 +0.00098，val loss 基本持平 | e70–e82 | e80–e90；已于 e82 封存 | 中高 |

建议以连续 10 轮与前 10 轮的主指标均值差、最佳 epoch 距离、val loss 是否持续改善
联合判断，不以单轮抖动或预算上限代替收敛判断。seg 使用 Mask mAP50-95，OBB 使用
Box mAP50-95。未来同口径从头重训建议预算设为 seg 110、OBB 90，并保留
`patience=15–20`；这两个数字是资源规划值，仍以实际验证平台为停止依据。

## 平台期后的提分实验

基线不再恢复训练。统一使用 `scripts/run_v3_score_sweep.py` 从各任务 `best.pt` 新建
独立微调实验：stage 1 比较低学习率 AdamW/cosine 与 1280 小目标方案，只有主指标
绝对提升至少 0.002 时才自动进入 stage 2 的增强实验。每个实验结束后刷新
`results/v3_score_sweep/summary.csv`、`analysis.md` 与 `human_review.csv`；人工复核逐类退化、
混淆矩阵和误检漏检后再决定下一轮参数。

## 环境记录

每次训练自动生成 `environment.json` 与 `run_manifest.json`（张劲焱 metadata
模块已并入主流程）：覆盖操作系统、Python、PyTorch、CUDA、GPU、Ultralytics、
代码提交与数据版本；无法自动获取的字段显式置空并在 notes 说明。校准阶段
额外输出 `results/training_calibration_v3/environment_capture_check.json`
确认字段完整后再启动主实验。

## 结果命名与交付

- 15 类主实验：`runs/v3_yolo11s_seg_960_e200/`（label_level=fine）
- 6 类正式实验：`runs/v3_hier_coarse_yolo11s_seg_960_e50/`（label_level=coarse）
- 两者 experiment_id 不同，6 类结果与 15 类预测聚合结果分开汇总；
- 每次 validation 导出 Mask/Box 分支指标，训练结束输出 per_class_metrics.csv
  与 validation_predictions.json，按统一 schema 交付张劲焱；
- 权重不提交 Git。
