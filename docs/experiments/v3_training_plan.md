# v3 无泄漏训练计划（blade-v3-grouped-202608）

状态：**等待负责人发布 v3 数据集**。数据集到达后按本计划执行，全程 test 锁定。

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
| 1 | 数据核对 + split 隔离校验 | `python -c "from blade_defect.data.split_isolation import assert_test_split_isolation; assert_test_split_isolation('datasets/blade-v3-grouped-202608/data.yaml')"` |
| 2 | 1 epoch 吞吐校准（v3 规模） | `python scripts/run_full_primary.py --config configs/experiments/v3_yolo11s_seg_960_e100.yaml --epochs 1 --run-name v3_throughput_calibration` |
| 3 | 100 epochs 主实验（15 类） | `python scripts/run_full_primary.py --config configs/experiments/v3_yolo11s_seg_960_e100.yaml` |
| 4 | 6 类正式实验（50 epochs） | `python scripts/run_full_primary.py --config configs/experiments/v3_hier_coarse_yolo11s_seg_960_e50.yaml` |

两个实验统一 batch=8、seed=42、YOLO11s-seg@960、从官方预训练权重启动；
checkpoint 每 5 轮保存，中断后直接重跑同一命令即可断点续训（自动检测
`weights/last.pt`）；中断检测用 `python scripts/check_run_status.py`。

## 环境记录

每次训练自动生成 `environment.json` 与 `run_manifest.json`（张劲焱 metadata
模块已并入主流程）：覆盖操作系统、Python、PyTorch、CUDA、GPU、Ultralytics、
代码提交与数据版本；无法自动获取的字段显式置空并在 notes 说明。校准阶段
额外输出 `results/training_calibration_v3/environment_capture_check.json`
确认字段完整后再启动主实验。

## 结果命名与交付

- 15 类主实验：`runs/v3_yolo11s_seg_960_e100/`（label_level=fine）
- 6 类正式实验：`runs/v3_hier_coarse_yolo11s_seg_960_e50/`（label_level=coarse）
- 两者 experiment_id 不同，6 类结果与 15 类预测聚合结果分开汇总；
- 每次 validation 导出 Mask/Box 分支指标，训练结束输出 per_class_metrics.csv
  与 validation_predictions.json，按统一 schema 交付张劲焱；
- 权重不提交 Git。
