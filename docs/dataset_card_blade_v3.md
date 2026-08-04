# blade-v3-grouped-202608 数据卡

负责人：黄晨婧

状态：`released`

发布日期：2026-08-04
父版本：`blade-v2-full-frozen-48291`

## 版本内容

| split | 图片 | 标签 | 用途 |
|---|---:|---:|---|
| train | 33,804 | 33,804 | 训练与训练期统计 |
| val | 7,244 | 7,244 | 验证、阈值和best epoch选择 |
| test | 7,243 | 7,243 | 锁定，仅负责人批准后的最终评估 |

总计48,291张图片、48,291个标签文件和49,471个实例。15个细类、6个粗类以及small/medium/large在三个split中均有覆盖。5张空标签负样本均被保留，当前位于test。

## 分组与均衡规则

1. 字节SHA-256一致的图片必须属于同一组。
2. 其余样本使用文件名推断的序列信息并结合细类、粗类建立序列组。
3. 初始候选复核发现47对同细类相邻拍摄样本跨split，因此停止原候选发布。
4. 修订版对2,169条“相邻帧且细类重叠”候选应用保守闭合并组，再按图片数、15类、6类、缺陷大小和类别乘大小进行确定性组级均衡。
5. 不进行样本级随机划分。迁移表记录17,218张旧split发生变化、31,073张保持不变。

## 发布审计

- 68个跨旧split SHA组：字节一致，最终同组且同split。
- 18个文件名前缀组：未直接按业务叶片ID整组强制合并。
- 最终相邻序列：94对仍跨split，但细类重叠候选为0。
- 第二轮感知哈希：稳定抽取48对；12对字节完全一致且已由SHA组放入同一split，36对未确认重复。
- 三份清单交集为0；48,291个稳定样本ID恰好出现一次；缺失图片和标签均为0。

审计文件位于`results/dataset_release_final/`：`grouping_rule_review.csv`、`perceptual_hash_review_round2.csv`、`split_migration.csv`、`split_class_distribution.csv`、`release_validation.json`和`test_lock_audit.json`。

## 字段含义与限制

`blade_id`、`flight_batch`、`capture_key`和`sequence_group`来自文件名解析，是用于降低泄漏风险的推断字段，不是负责人确认的业务真值。后续取得真实叶片、航次或缺陷ID后，应发布新版本重新分组，不能覆盖本版本。

感知哈希适合发现全局轮廓相似候选，但会把简单叶片或塔筒轮廓判为相似，因此仅作为人工复核入口，不单独用于自动删除或并组。

## 存储与身份

正式目录仅包含`data.yaml`、`train.txt`、`val.txt`、`test.txt`和`dataset_manifest.json`。清单通过相对路径引用`../blade-v2/images/...`和对应标签，不依赖绝对盘符。NTFS junction只是本机访问方式，不属于数据身份。

关键指纹：

```text
membership sha256: 02a0251aeca30a4e8b332967c2124848955886a2550f75933cc1e38146dfc37b
release lists sha256: 67e7589c38c222d269b6cfc8d7a19a6b83e6ef6588c6546a2d112cb81016a41f
labels sha256: 065efdfffe956800c25a85f70e3a5860a02c23cf7294a144a00db633ce53168c
test sha256: 51e72b8b0d1ac963b818e8b2c9b3f006c5716e85e49a289f32e92875096dc9d2
```

## test锁定策略

禁止将test用于训练、数据增强调试、阈值选择、best epoch选择、模型选择或超参数搜索。`scripts/audit_test_lock.py`会校验test哈希、三集合无交集、全量覆盖，并扫描训练配置和run manifest；任一违规都会返回失败。本版本不提供自动解锁入口。
