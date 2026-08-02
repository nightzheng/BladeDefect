# blade-v3-grouped Candidate Split

`blade-v3-grouped` 的目标是在不修改 `blade-v2` 冻结版的前提下，生成 train/val/test = 70%/15%/15% 的候选划分，并降低连续拍摄样本跨划分造成的数据泄漏风险。

## 分组策略

- 对每张图片计算逐字节 SHA-256；完全相同图片进入同一 `duplicate_group_id`。
- 能解析 `DSCxxxxx` 的样本使用 `capture_key + 15类缺陷签名 + 6类粗分组签名` 生成 `sequence_group`。
- 文件名前缀可解析时记录 `blade_id` 和 `flight_batch`，作为复查字段。
- 无法解析序列来源时，用文件名和缺陷类型生成稳定 fallback 组。
- 18个跨划分文件名前缀组和2,404对相邻序列候选保存在 `sequence_group_candidates.csv`，并补充 `blade_id`、`flight_batch`、`sequence_group` 和待确认状态。

## 划分策略

- 以组为最小单位分配到 train、val 或 test，不进行样本级随机划分。
- 优先保证 exact duplicate group 和 sequence group 不跨 split。
- 在上述约束下，按固定 seed 对完整组稳定排序；联合比较图片数、15类实例、6类实例、small/medium/large和“细类×大小”分布，使其接近70%/15%/15%。
- seed只影响稳定的组级并列顺序，不对单张图片随机抽签，同组样本始终一起移动。
- 如果比例和无泄漏约束冲突，以无泄漏约束优先，并在校验文件和周报中记录偏差。

## 验收文件

- `results/split_candidate/train.txt`
- `results/split_candidate/val.txt`
- `results/split_candidate/test.txt`
- `results/split_candidate/split_group_summary.csv`
- `results/split_candidate/sample_group_assignments.csv`
- `results/split_candidate/class_distribution.csv`
- `results/split_candidate/split_balance_report.json`
- `results/split_candidate/leakage_validation.json`
- `results/split_candidate/dataset_validation.json`
- `datasets/blade-v3-grouped/dataset_manifest.json`

## 判断优先级

1. 完全相同图片、同序列与合并后的完整组不得跨 split。
2. 冻结版48,291组样本必须全部保留且恰好出现一次。
3. train/val/test图片比例尽量接近70%/15%/15%。
4. 在前三项约束内，尽量均衡15类、6类和缺陷大小分布；若大型序列组导致偏差，必须在报告中保留真实偏差，不拆组掩盖。
