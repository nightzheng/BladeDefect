# Dataset Registry

本项目用 `configs/dataset_registry.yaml` 区分数据版本身份。数据版本身份由 `dataset_id`、manifest、样本清单哈希、标签哈希、来源配置和生成命令共同确定；本机绝对路径和 NTFS junction 只作为访问入口。

## 当前版本

- `blade-v2-sampled-4987`: 历史抽样实验版本，仅用于解释已有小样本实验。
- `blade-v2-full-frozen-48291`: 冻结全量来源版本，保持 `datasets/blade-v2`、`train.txt`、`val.txt` 和 manifest 不变。
- `blade-v3-grouped`: 本周新增候选分组划分，不替换正式训练配置。

## 使用原则

- 不在原始图像和冻结数据目录中删除、移动或覆盖样本。
- `blade-v3-grouped` 只生成候选 `train.txt`、`val.txt`、`test.txt` 和 manifest。
- 分组划分优先保证同一 exact SHA-256 重复组、同一序列名称和缺陷类型组合不跨 split。
- 划分器以完整组为最小单位，按固定 seed 生成稳定组级顺序，并联合平衡图片数、15类、6类、目标大小和“细类×大小”分布；不执行样本级随机抽签。
- 感知哈希近重复只作为复查候选，不自动删除，也不直接判定为泄漏。
- 候选清单和 manifest 使用仓库相对路径；NTFS junction 与本机盘符只用于本机访问，不属于数据版本身份。

## 重建命令

```powershell
python scripts\build_duplicate_groups.py `
  --dataset datasets\blade-v2 `
  --output results\dataset_registry\exact_duplicate_groups.csv `
  --inventory-output results\dataset_registry\image_sha256.csv `
  --workers 8

python scripts\build_grouped_split.py `
  --dataset datasets\blade-v2 `
  --data datasets\blade-v2\data.yaml `
  --hierarchy configs\class_hierarchy.yaml `
  --registry-output results\dataset_registry `
  --split-output results\split_candidate `
  --manifest-output datasets\blade-v3-grouped `
  --hash-inventory results\dataset_registry\image_sha256.csv `
  --workers 8 `
  --seed 42 `
  --command "见 docs/dataset_registry.md"

python scripts\validate_split_leakage.py `
  --dataset datasets\blade-v2 `
  --split-output results\split_candidate `
  --hash-inventory results\dataset_registry\image_sha256.csv `
  --leakage-output results\split_candidate\leakage_validation.json `
  --dataset-output results\split_candidate\dataset_validation.json
```

## 关键校验

- `image_sha256.csv` 是全量逐字节哈希清单，划分和泄漏复查共同复用，避免同一批图片重复计算哈希。
- `sample_group_assignments.csv` 逐样本记录来源 split、目标 split、重复组、序列组、叶片/批次字段、15类、6类与大小类别。
- `leakage_validation.json` 直接用 SHA 清单复查跨 split 重复，并检查序列组、组ID、索引重叠、重复索引和分组声明一致性。
- `dataset_validation.json` 检查冻结版全部样本是否恰好出现一次、图片与标签是否存在、标签是否合规以及15类是否在三个 split 中完整出现。
- `split_balance_report.json` 记录实际比例和15类、6类、目标大小分布偏差；分组约束与比例冲突时以无泄漏优先。
