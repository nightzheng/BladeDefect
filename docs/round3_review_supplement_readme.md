# Round 3 复核补件说明（修订版）

> 发件：张劲焱　　收件：复核人　　日期：2026-10-24
> 本文件是 `round3_review_supplement.zip` 内 README 的**入库修订版**（已修正一处外文混入：原文该处误用了非中文词汇，现统一为中文表述）。
> 配套包：`round3_review_package_v3.zip`（结论 v3 + 终版归属核验）。

---

## 一、本次补件解决什么

复核人 2026-10-24 提出两个问题：一个程序、一个留痕。本补件为回应，另附上"漏检清单"缺失项。

## 二、程序违规（已追认，已留痕）

锚点策略（"落点全图随机" → "锚点必须落在目标图已有标注多边形内"，并加 2% 边界余量）是在 spec 冻结**之后**的改参，**未按流程先发复核人确认**。

- 复核人予以追认（动机正当、留痕完整、方向更真实、结论不依赖落点细节），**复核记录将写明"冻结后改参未事先确认"**。
- 已写入 `docs/v3_round3_conclusions.md` **§7.0**。
- **流程要求（以后无例外）**：冻结后任何参数改动，**先发消息给复核人**，半小时内会回；不得先做后报。

## 三、留痕缺口（已补齐）

原包中的 `11_split_membership_check.log` 是针对**旧版 manifest** 跑的（log 写"物理 val 目标图 34 张/34 次"）。

对**终版 manifest** 重跑后（`split_membership_check_final.log`，已并入 v3 包）：

```
违规记录数（src/dst 不在 train 索引，或任一在 val 索引）: 0
物理路径含 /images/val/ 的源图：14 张 → 全在 train 索引，无一在 val 索引
物理路径含 /images/val/ 的目标图：24 张（被选中 24 次）→ 全在 train 索引
结论：违规 0 条，split 归属成立
```

差异原因：终版 manifest 的 dst 池整体换过（24 张/24 次 vs 旧版 34 张/34 次），src 池未变（仍 14 张）。

## 四、`prediction_export_error`（已定位并修复）

`b1_metrics.json` 中记录的导出失败：

```
prediction_export_error: /data/datasets/derived/blade-v2/images/train/-3-3-DSC01666.JPG does not exist
```

**根因（代码 bug，非数据问题）**：`prediction_exporter.py` 解析 val.txt 内相对条目时，以 **data.yaml 的 `path`**（派生目录）为基准；正确基准应是 **val.txt 自身所在目录**。派生集 val 指回原数据集，于是拼成 `derived/blade-v2/...` 全数找不到。

**修复**：`index_base = val_path.parent`（diff 见包内 `prediction_exporter.fix.patch`）。

**影响范围**：mAP / recall / precision 由 ultralytics `val()` 内部计算，**不受影响**，三条判据的计算也不用重做；受影响的只有"逐图预测 JSON"与"逐实例错误分解"这一层。

## 五、漏检清单（已产出）

用 B-1 的 `best.pt` 重导 val 预测（7,244 张）后完成错误分解：

| 指标 | A0 | B-1 |
|---|---:|---:|
| ok（检出且类别正确） | 1 | **0** |
| misclassified | 3 | 4 |
| missed | 21（25 行含修复前 2 条重复） | **19** |
| false positive | **1** | **11** |

19 个漏检实例明细见 `round3_results/missed_class13_b1.csv`。

## 六、包内文件清单

| 文件 | 内容 |
|---|---|
| `round3_review_supplement.zip` 内的 `README.md` | 本说明（修订版） |
| `split_membership_check_final.log` | 终版 manifest 归属核验（0 违规，24/24） |
| `error_decomposition_b1_vs_a0.json` | 两臂逐类错误分解 |
| `missed_class13_b1.csv` / `.json` | B-1 类 13 的 19 个漏检实例明细 |
| `missed_class13_a0.csv` / `.json` | A0 同口径对照 |
| `prediction_exporter.fix.patch` | 导出器修复 diff |
