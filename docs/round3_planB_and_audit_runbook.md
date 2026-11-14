# Round 3 前置工作 Runbook（标注审计 + 方案 B）

> 适用：下一阶段计划的"事项 1（标注审计）"与"事项 2（方案 B）"的前置部分。
> 新增代码已入库，但**必须在有数据的机器上跑**（本机没有 `/data`，也无法装 ultralytics）。
> 机器当前为"停机保留数据盘"状态，执行前先启动实例并挂载 `/data`。

---

## 0. 这次新增了什么

| 文件 | 作用 |
|---|---|
| `src/blade_defect/data/class_copy_paste.py` | 按类定向 copy-paste 的**离线合成**实现（标签读写、几何、粘贴、manifest） |
| `src/blade_defect/data/label_audit.py` | 标注审计的**可疑度打分**（7 类 flag + 优先级） |
| `scripts/build_class_copy_paste.py` | 合成入口（支持 `--dry-run` 先看数量） |
| `scripts/audit_small_class_labels.py` | 审计入口（产出人工核对 CSV + 结构画像 JSON） |
| `tests/test_class_copy_paste.py` | 8 个单元测试 |
| `tests/test_label_audit.py` | 6 个单元测试 |

**为什么用离线合成而不是改 dataloader**：ultralytics 原生 `copy_paste` 全类生效；
按类筛选要侵入它的变换链，而 Round 2 已经因为 dataloader 侧的 label cache 并发
竞态翻过车。离线合成把风险移到"生成一批图"这个可审计、可复现的步骤上，
训练侧只是"train 多了一些图"，与现有 preflight / 数据门禁完全兼容。

---

## 1. 开机门禁（必须先过，过了才允许开训）

```bash
# 目录：/root/BladeDefect
source /root/miniconda3/etc/profile.d/conda.sh && conda activate bladedefect
python -m pytest tests/test_class_copy_paste.py tests/test_label_audit.py -q
```

期望：`14 passed`。**任何 failed 都停下来修，不要开训。**

---

## 2. 事项 1：小样本类标注审计（零算力，先做）

### 2.1 跑

```bash
# 目录：/root/BladeDefect
python scripts/audit_small_class_labels.py \
  --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
  --classes 10,13,14 \
  --splits train,val \
  --out-csv  /data/backup/round3/annotation_audit.csv \
  --out-json /data/backup/round3/annotation_audit_summary.json \
  --top 120
```

### 2.2 产物怎么读

- **CSV**（`annotation_audit.csv`）：一实例一行，按 `priority` 降序。最后两列
  `review_verdict` / `reviewer_note` 是空的，留给人工填。
  `verdict` 建议取值：`ok` / `fix_label` / `wrong_class` / `not_defect` / `need_engineer`。
- **JSON**：每类实例数、命中各 flag 的数量、被标记比例。

### 2.3 flag 含义与权重

| flag | 含义 | 权重 |
|---|---|---|
| `ZERO_AREA` | 多边形面积≈0（点退化成线） | 5 |
| `OUT_OF_BOUNDS` | 坐标 <0 或 >1 | 4 |
| `DUPLICATE` | 同类实例互相重叠 IoU≥0.85 | 3 |
| `TINY` | 像素面积 < 32×32 | 3 |
| `THIN` | 包围盒最短边 < 8px | 2 |
| `CONFUSABLE` | 同图出现易混类（10↔7、13↔11/12、14↔11） | 2 |
| `NO_LABEL_FILE` | 有图无标签 | 2 |

### 2.4 人工核对怎么分配

- `priority ≥ 5` 的：**全部逐张看**（通常是零面积/越界，一眼能定）
- `3 ≤ priority < 5`：抽看，重点看图（tiny/thin 到底是不是可辨认的缺陷）
- `priority < 3` 的 `ok` 候选：抽 20% 复核即可

### 2.5 验收标准（事项 1 的出口）

1. CSV 里 `priority ≥ 5` 的行 **100% 有 verdict**
2. 类 10 的 66 条实例 **全部有 verdict**（这是 Round 2 结论写死的硬性前置）
3. 产出一页《标注修正清单》：`fix_label` / `wrong_class` 的条目逐条给出改法
4. 若出现 `need_engineer`，汇总成问题清单发给企业工程师（参考 2026-10-13 那份的格式）

---

## 3. 事项 2：方案 B 的合成与开训

### 3.0 前置：标签口径已经变了（2026-10-23）

2026-10-23 已删除类 13 的 3 对重复标注（`manifest.instances_removed = 3`），
**val 的类 13 GT 从 25 变为 23，全量 141 → 138**（详见 `round3_annotation_audit_report.md` §5.1）。

对 B-1 的影响，二选一，**必须在结论文档里写明走了哪条**：

| 做法 | 说明 | 代价 |
|---|---|---|
| **A. 复评 A0（推荐）** | 用 A0 的 `weights/best.pt` 在**修复后的 val** 上重新 validate，得到同口径基线，再与 B-1 比 | ~10 分钟 GPU（结果写进文档） |
| **B. 注明口径差异** | 直接与 Round 2 的 A0 数值对比，但结论里必须标注"对照臂为旧标注评估" | 0 成本，但可比性弱 |

无论选哪条，**类 13 的 Round 2 数值（0.0845 / 0.0917）都不能与修复后直接比较**——
这两个数是在含重复实例的旧 val 上算的。

### 3.1 先 dry-run 看数量（不写文件，秒级）

```bash
# 目录：/root/BladeDefect
python scripts/build_class_copy_paste.py \
  --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
  --out-dir   /data/datasets/derived/blade-v3-grouped-202608-cp10_13_14 \
  --classes 10,13,14 --copies 3 --dry-run
```

**预期参考值**（按 Round 2 的实例数）：类 10 = 66、类 13 = 25、类 14 = 556，
合计 647 个源实例 × 3 = **约 1,941 张合成图**。

`per_class_source` 若与上表差很多，先停下核对类别 ID 是否串了。

### 3.2 真正生成（CPU，几分钟；磁盘约 1–2 GB）

```bash
# 目录：/root/BladeDefect
python scripts/build_class_copy_paste.py \
  --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
  --out-dir   /data/datasets/derived/blade-v3-grouped-202608-cp10_13_14 \
  --classes 10,13,14 --copies 3 --max-synthetic 2000
```

产物：

```
/data/datasets/derived/blade-v3-grouped-202608-cp10_13_14/
├── images/train/*.jpg        合成图
├── labels/train/*.txt        目标图原标注 + 贴入实例
├── train.txt                 原 train + 合成样本
├── data.yaml                 可直接喂训练（val 指回原 val）
└── copy_paste_manifest.json  逐条审计记录（seed/源图/目标图/类别/缩放/落点）
```

### 3.3 生成后必须抽查三条

```bash
# 目录：任意
# ① 合成图数量与 manifest 一致
ls /data/datasets/derived/blade-v3-grouped-202608-cp10_13_14/images/train | wc -l

# ② val 没有被污染（data.yaml 里 val 必须指向原数据集）
grep '^val:' /data/datasets/derived/blade-v3-grouped-202608-cp10_13_14/data.yaml

# ③ 随机抽 10 张合成图人工看一眼：贴进去的实例位置是否自然、有没有糊成一片
```

③ 是唯一需要眼睛的步骤。若发现大量"贴在天空/背景上"这种不自然样本，
把 `--scale-min/--scale-max` 收窄，或降低 `--copies`。

### 3.4 开训

配置项已写在 `configs/v3_score_sweep.yaml`：
`v3_seg_cp_class10_13_14_960_e25`（唯一变量 = copy-paste，其余与 Round 2 的 A0 完全一致）。

```bash
# 目录：/root/BladeDefect
python scripts/run_v3_score_sweep.py preflight --require-cuda --task seg
```

> 派生数据集的索引条数会变（原 33,804 + 合成 ~1,941），所以 `dataset_manifest.json`
> 里的条数校验会不通过 —— 这是预期的，**派生数据集不带 manifest**，
> preflight 只校验 `errors` 为空即可。若报"清单数量与 manifest 不一致"，
> 说明该目录里残留了原始 manifest，删掉即可。

```bash
# 目录：任意
tmux new-session -d -s cp1 "cd /root/BladeDefect && source /root/miniconda3/etc/profile.d/conda.sh && conda activate bladedefect && python scripts/run_v3_score_sweep.py run --task seg --stage 1 --experiment v3_seg_cp_class10_13_14_960_e25 --device 0 2>&1 | tee /data/backup/round3_logs/cp1.log"
```

---

## 4. 已知限制（写进结论文档，别让复核人踩雷）

1. **合成样本不进入 val/test**：val 索引原样指回原数据集，test 依旧锁定。
2. **单实例单多边形**：与数据集现状一致；多段实例在合成时按单段处理。
3. **只做平移+缩放，不做旋转/翻转**：控制变量，避免引入几何增强混淆。
4. **每张合成图只贴 1 个实例**：便于把"贴入实例数"与"提升幅度"对应起来。
5. **合成样本与原图的分布差异**：贴入实例来自真实标注，但背景/光照是目标图的，
   可能出现不自然的组合；3.3 ③ 的抽查就是为这个设的。
6. **审计脚本不判断"算不算缺陷"**：只排序可疑度，最终结论必须人工 + 企业工程师裁定。
