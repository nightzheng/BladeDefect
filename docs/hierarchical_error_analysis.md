# 层级错误分析

## 概述

本模块基于 `configs/class_hierarchy.yaml` 的 15 类→6 大类映射，将错误拆分为三个层级。核心原则：**漏检是定位/召回失败，不属于分类错误**。

## 三层错误层级

| 层级 | 类别 | 定义 | 分母 |
|---|---|---|---|
| L1 定位/召回 | localization_miss (FN) | GT 实例未被任何预测召回 | 总 GT 实例 |
| L1 定位/召回 | localization_false_positive (FP) | 预测未匹配任何 GT | 总预测数 |
| L2 分类（已匹配） | coarse_group_error | 已匹配 GT-pred 对中，6大类不同 | 已匹配对数 |
| L2 分类（已匹配） | within_group_confusion | 已匹配 GT-pred 对中，同大类但不同细类 | 已匹配对数 |
| L3 mask 质量 | mask_quality_error | 已匹配对中 mask IoU 低于阈值 | 已匹配对数 |

## 层级映射

| 大类 key | 包含细类 ID | 描述 |
|---|---|---|
| surface_corrosion | 0, 1, 2, 3 | 表面腐蚀 |
| surface_crack | 4, 5 | 表面裂纹 |
| surface_defect | 6, 7, 8, 9 | 表面缺陷 |
| repair_trace | 10 | 维修痕迹 |
| blade_damage | 11, 12, 13 | 叶片损伤 |
| attachment_loss | 14 | 附件脱落 |

## 正式结果 (full_primary_yolo11s_seg_960, blade-v2)

### L1 定位失败

- **漏检 (FN)**: 3,382 / 4,943 = **68.4%** of GT instances
- **误检 (FP)**: 463 / 2,033 = **22.8%** of predictions
- **正确召回**: 1,491 / 4,943 = 30.2%

### L2 分类错误（仅计算已匹配的 1,491 对）

- **跨大类错误 (coarse_group_error)**: 129 / 1,491 = **8.7%**
- **同组细类混淆 (within_group_confusion)**: 750 对（同一大类内部细类混淆，来自 per-fine-class-pair 计数，非 per-instance）

### L3 mask 质量

- 阈值待团队确定。建议 mask IoU < 0.5 作为质量问题标志。

### 关键结论

1. **错误主要发生在跨大类之间还是同一大类内部？**
   - 错误主要在 L1（漏检/召回失败），FN 率 68.4%。一旦成功检测（1,491 对），大类分类准确率 91.3%（仅 8.7% 跨大类错误）。
   - 同组细类混淆 750 对与跨大类错误 129 来自不同计数粒度，不能直接比较。

2. **哪些大类混淆组合最严重？**
   - 腐蚀→缺陷 (22 实例) 和 缺陷→裂纹 (61 实例) 是主要跨大类混淆
   - 腐蚀组内 (0-3) 和缺陷组内 (6-9) 是同组细类混淆的主要来源

3. **困难类别**
   - 类别13（结构损伤，27 GT）AP50=0.102
   - 类别8（胶衣脱落）AP50=0.128
   - 类别14（接闪器脱落）AP50=0.292

## 输出文件

路径：`results/analysis/`

| 文件 | 说明 |
|---|---|
| `coarse_confusion_matrix.png` | 6x6 混淆矩阵 |
| `coarse_class_metrics.csv` | 逐大类 Precision/Recall/F1 |
| `within_group_confusion.csv` | 同组细类混淆 |
| `hierarchical_error_distribution.png` | 四类错误分布 |

路径：`results/error_contract/`

| 文件 | 说明 |
|---|---|
| `error_metric_contract.md` | 错误统计合同（三层定义、分母） |
| `denominator_audit.csv` | 分母审计 |
| `duplicate_count_audit.csv` | 129 vs 750 计数口径说明 |

## 注意事项

- 15类预测聚合为6类属于**诊断分析**，独立6类重训练属于**模型实验**，两者不互相替代
- 旧全量模型（split_leakage_known=true）的指标不能与 v3 无泄漏正式结果直接混排
- FN/FP 场景标签（过曝、阴影等）与模型错误类型分开保存
