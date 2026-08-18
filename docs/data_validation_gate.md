# 数据校验门禁与解码缓存规则（2026.08 固定）

## 背景

正式 v3 / OBB 派生数据（blade-v3-grouped-obb，48,291 张索引图片）的全量门禁
`check_obb_indexed_samples` 对每张图片执行 OpenCV 解码校验，单次约 40—50 分钟纯 CPU；
smoke、batch 校准、50/100 轮 baseline 每次启动都会重复触发，属无效等待。
本轮为全量图像解码门禁引入可审计缓存，**不降低校验强度、不改为全面抽样**。

## 缓存键（三者缺一不可）

| 组成 | 内容 | 作用 |
|---|---|---|
| 成员指纹 | `membership_sha256` + 各 split `identity_sha256`（由索引 txt 现场重算） | 判断数据成员是否变化 |
| 文件元信息 | 逐样本图片/标签的 `(size_bytes, mtime_ns)` 聚合 `file_state_sha256` | 判断文件内容是否可能被改写 |
| 校验版本 | `CACHE_SCHEMA_VERSION` + 校验参数（`num_classes`、`min_area`）+ validator 标识 | 校验逻辑或口径变化时强制全量复检 |

**禁止仅用旧时间戳或目录名判断数据未变化**：时间戳只是逐文件元信息的一个分量，
必须与成员指纹和校验版本共同命中才允许复用；目录名完全不参与判断。
（回归测试：`test_directory_name_or_timestamp_alone_cannot_satisfy_cache`。）

## 固定门禁规则

1. **全量解码**：正式数据首次构建，或成员指纹 / 文件元信息 / 校验版本任一变化时。
2. **缓存命中**：三者全部一致时复用已通过的逐样本解码结果，命中运行不再逐张解码；
   损坏图片的判定随缓存一并复用，命中运行同样报告 corrupt（测试
   `test_corrupt_image_stays_flagged_on_cache_hit`）。
3. **变化复检**：仅个别文件被修改（成员不变、元信息变化）时，只对变化样本重新解码，
   其余样本复用缓存；缓存缺键的样本立即补解码，不静默信任（测试
   `test_modified_image_triggers_targeted_recheck`、
   `test_missing_cache_entry_is_decoded_not_silently_trusted`）。
4. **每次都全量执行、不进入缓存**的部分：标签格式逐行校验、图片/标签路径存在性、
   索引重复项、跨 split 重叠与 split/membership 一致性。命中运行中标签若被破坏
   仍会被判 invalid（测试 `test_label_format_is_fully_rechecked_on_cache_hit`）。
5. **CI/smoke 抽样**：`--decode-sample N --decode-seed 42` 使用固定种子对索引成员
   抽样解码；抽样模式不读取也不写入正式缓存，报告中以 `decision=sampled` 单独标记，
   不得作为正式数据完整性结论。

## 使用方式

```powershell
# 正式门禁（冷启动全量解码并写缓存；指纹未变时下次自动命中）
python scripts/cache_dataset_validation.py --dataset datasets/blade-v3-grouped-obb `
  --report results/data_gate_cache/cold_run_report.json

# CI / smoke 抽样（固定种子，不触碰正式缓存）
python scripts/cache_dataset_validation.py --dataset datasets/blade-v3-grouped-obb `
  --decode-sample 512 --decode-seed 42

# 强制全量复检（忽略已有缓存，仍写缓存）
python scripts/cache_dataset_validation.py --dataset datasets/blade-v3-grouped-obb --no-cache
```

缓存文件：`results/data_gate_cache/validation_cache.json`（含缓存键、逐样本解码结果与
文件状态表；属本地结果，不进入 Git）。每次运行的决策、命中/复检数量与分阶段耗时经
`--report` 落盘。

## 口径与安全约束

- 缓存只复用“图片可解码性”这一昂贵判定；**正式指标读取路径、训练配置、数据成员
  完全不经缓存改写**——命中运行的 split 报告与一致性结论必须与冷启动逐字段一致
  （测试 `test_cache_hit_skips_decode_and_reproduces_report`）。
- test split 保持锁定：本门禁对 test 仅做完整性校验（`test_data_integrity_only=true`），
  与既有 `check_obb_dataset.py` 行为一致；本脚本不触发任何训练或指标评估。
- 校验逻辑变更时必须递增 `CACHE_SCHEMA_VERSION`，使历史缓存自动失效（测试
  `test_validation_version_change_invalidates_cache`）。

## 实测耗时（blade-v3-grouped-obb，48,291 张，RTX 4060 Laptop 机型，2026-08-18 真实运行）

| 场景 | 决策 | 解码张数 | 总耗时 | 报告 |
|---|---|---|---|---|
| 冷启动（首次构建） | cold | 48,291 | 3,342.3s（解码 3,303.0s + 文件元信息 19.0s + 标签全量 18.4s + 索引 1.7s） | `results/data_gate_cache/cold_run_report.json` |
| 缓存命中 | hit | 0（复用 48,291 条） | 34.9s（标签仍全量 13.6s + 元信息 19.2s；解码 0.01s） | `results/data_gate_cache/cache_hit_report.json` |
| 失效与复检场景 | cold/partial | 见报告 | — | `results/data_gate_cache/cache_invalidation_report.json` |

命中运行与 `results/obb_v3/validation_report.json` 基线逐字段一致（三 split 图片/标签/
实例数、负样本数、membership、valid 全部相同），证明缓存复用不改变校验结论。
相比冷启动 3,342s，命中运行 34.9s，单次启动等待降低约 98.96%。
