# 层级错误分析

## 概述

本模块基于 configs/class_hierarchy.yaml 的 15 类→6 大类映射，对正式实验预测结果进行
层级错误分析，回答以下问题：

1. 错误主要发生在大类之间还是同一大类内部？
2. 哪些大类混淆组合最严重？
3. 各类错误（漏检、误检、类别混淆、定位误差、mask 质量）的比例如何？

## 层级映射

| 大类 key           | 包含细类 ID            | 描述     |
|--------------------|------------------------|----------|
| surface_corrosion  | 0, 1, 2, 3             | 表面腐蚀 |
| surface_crack      | 4, 5                    | 表面裂纹 |
| surface_defect     | 6, 7, 8, 9              | 表面缺陷 |
| repair_trace       | 10                      | 维修痕迹 |
| blade_damage       | 11, 12, 13              | 叶片损伤 |
| attachment_loss    | 14                      | 附件脱落 |

## 聚合方法

将 15 类模型的 val 预测结果和真实标签按映射表聚合为 6 大类，以此计算：

- **6×6 混淆矩阵**：展示大类间的混淆模式
- **逐大类 Precision / Recall / F1**：评估每个大类的可识别性
- **错误分类统计**：
  - coarse_group_error：预测大类与真实大类不同
  - within_group_confusion：预测与真实属于同一大类，但细类不同
  - localization_error：IOU 低于阈值的匹配
  - mask_quality_error：mask 边界质量低于标准的匹配

## 重点分析

- 类别 0–3（表面腐蚀）内部混淆
- 类别 4–5（表面裂纹）内部混淆
- 类别 6–9（表面缺陷）内部混淆
- 类别 11–13（叶片损伤）内部混淆
- 大类间高频混淆组合（如腐蚀 vs 裂纹）

## 输出文件

路径：esults/hierarchy/

| 文件                                  | 说明                           |
|---------------------------------------|--------------------------------|
| coarse_confusion_matrix.png           | 6 类混淆矩阵热力图             |
| coarse_class_metrics.csv              | 逐大类 Precision/Recall/F1    |
| within_group_confusion.csv            | 同组内部细类混淆记录           |
| hierarchical_metrics.csv              | 层级指标汇总                   |
| hierarchical_error_distribution.png   | 四类错误比例饼图               |

## 使用方法

`powershell
# 运行完整分析（读取 runs/ 下的预测结果）
python -c ^
  \"from blade_defect.experiment.analysis import analyze_hierarchy; ^
   analyze_hierarchy('results/summary.csv', 'runs', 'results/hierarchy')\"
`

## 待验证假设

1. 表面腐蚀（0–3）和表面缺陷（6–9）可能由于视觉特征相似而导致跨大类混淆。
2. 维修痕迹（10）样本量小，可能被误分类到背景或其他大类。
3. 附件脱落（14）属于罕见类别，漏检率可能显著高于其他大类。
