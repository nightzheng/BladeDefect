# B-1 合成规格（写死，开机前复核用）

> 路线：**离线合成**（不做 dataloader 回调）
> 实现：`src/blade_defect/data/class_copy_paste.py` + `scripts/build_class_copy_paste.py`
> 测试：`tests/test_class_copy_paste.py`（8 项）+ `tests/test_label_audit.py`（5 项）
> 本文档一经复核即冻结，执行时不得改参数；要改必须重新报备。

---

## 1. 变量与对照

| 项 | 值 |
|---|---|
| 唯一变量 | **仅类 13 的定向 copy-paste**（其他一切与 A0 相同） |
| 对照臂 | **A0 在修复后 val 上的复评值**（`/data/runs/v3_seg_a0_reval_fixedlabels/metrics.json`），不用 Round 2 旧值 |
| 训练配置 | 与 A0 完全一致：imgsz 960 / batch 4 / epochs 25 / patience 8 / AdamW lr0 5e-4 / lrf 0.01 / cos_lr / warmup 2 / wd 5e-4 / mosaic 1.0 / close_mosaic 10 / 无原生 copy_paste / seed 42 / workers 8 / 起始权重 = 封存基线 best e89 |
| **data= 指向** | **`/data/datasets/derived/blade-v3-grouped-202608-b1cp13/data.yaml`**（派生集，**不是**原 `data.yaml`；用默认值会退化成 A0 复跑） |

---

## 2. 合成参数（逐项写死）

| 参数 | 值 | 说明 |
|---|---|---|
| 目标类别 | **仅 13**（叶片损伤--结构损伤） | 只从类 13 的实例里取源 |
| 源实例范围 | 原 **train** split 中所有类 13 实例 | 不用 val/test 的实例做源，避免验证信息泄漏进训练 |
| 每个源实例复制次数 | **3** | 类 13 在 train 的实例数 × 3 = 合成图数量上限 |
| 每张合成图贴入实例数 | **1** | 便于"贴入数量 ↔ 提升幅度"对应，不做一图多贴 |
| 来源图选取 | 随机（`random.Random(seed)`），要求 `src_image != dst_image` | 同一张图不自我复制 |
| 目标图选取 | 从**原 train split 全量**随机取（不限类别、允许重复被选） | 保持背景分布贴近训练集 |
| 缩放范围 | **U(0.8, 1.2)** | 只做轻微尺度抖动 |
| 旋转 | **无（0°）** | 故意不做，避免与 A0 的几何增强混淆 |
| 翻转 | **无** | 同上 |
| 落点 | 随机平移，**保证贴入实例完整落在图内**（不裁剪） | 避免产生越界多边形 |
| 合成图编码 | JPEG quality 92 | 与原图同格式 |
| 随机种子 | **42** | 与训练 seed 一致；同 seed 重跑得到同一批图 |
| 合成图是否进 mosaic | **进**。合成图追加进 train 索引，ultralytics 的 `mosaic 1.0` 对它们**与原始图一视同仁** | 这是刻意选择：不做特殊豁免，避免引入第二个变量 |
| **合成图总数（写死）** | **345 张** = 类 13 在修复后 train 的 115 个实例 × 3 | 生成后 `synthetic_images` 必须等于 345；**不等就停下核对**，不得带着异常数开训 |

**对训练步数的影响（实测口径）**：

| | 训练图数 | iters/epoch（batch 4） | 相对变化 |
|---|---:|---:|---:|
| A0 | 33,804 | 8,451 | — |
| B-1 | 34,149 | 8,538 | **+1.03%** |

合成量级只把每 epoch 步数抬高约 1%，不足以影响收敛判断；因此**不需要为 B-1 调整 epochs**，与 A0 保持 25 轮即可。


> 说明：mosaic 这条必须写清，因为"合成图不参与 mosaic"会成为隐藏的第二个变量；本方案选择让它参与，与 A0 的增强口径完全一致。

---

## 3. 目录与索引（不改原数据集）

```
/data/datasets/derived/blade-v3-grouped-202608-b1cp13/
├── images/train/*.jpg        合成图 **345 张**（115 个类 13 实例 × 3，写死）
├── labels/train/*.txt        目标图原标注 + 贴入的类 13 实例
├── train.txt                 原 train 索引（原样）+ 合成图绝对路径（追加）
├── data.yaml                 path=本目录 / train=本目录 train.txt / val=原 val.txt / test 原样指回
└── copy_paste_manifest.json  逐条记录：源图、目标图、类别、缩放、落点、面积、seed
```

- [ ] 原数据集 `/data/datasets/v3-index-rebuild/...` **只读**，一个字节都不改
- [ ] `val` 与 `test` 条目**原样指回原数据集**，派生集不新增、不改写
- [ ] 合成样本**只进 train**

---

## 4. 开机后、训练前的三步验收（缺一不可）

```bash
# 目录：/root/BladeDefect
source /root/miniconda3/etc/profile.d/conda.sh && conda activate bladedefect
```

**① 生成前 dry-run，核对源实例数**

```bash
python scripts/build_class_copy_paste.py --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
  --out-dir /data/datasets/derived/blade-v3-grouped-202608-b1cp13 --classes 13 --copies 3 --dry-run
```

期望 `per_class_source["13"] = 115`（修复后 train 的类 13 实例数），`planned_synthetic = 345`。

**② 正式生成 + 参数自检**

```bash
python scripts/build_class_copy_paste.py --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
  --out-dir /data/datasets/derived/blade-v3-grouped-202608-b1cp13 --classes 13 --copies 3 --seed 42 --scale-min 0.8 --scale-max 1.2
```

生成后核对 manifest 的关键字段：`target_classes=[13]`、`seed=42`、`copies_per_instance=3`、`rotation=none`、`flip=none`、`pastes_per_synthetic_image=1`、`per_class_synthetic={"13": N}`。

**③ 对合成集复跑审计（验收口径见 §5）**

```bash
python scripts/audit_small_class_labels.py \
  --data-yaml /data/datasets/derived/blade-v3-grouped-202608-b1cp13/data.yaml \
  --classes 13 --splits train --geometric-only \
  --out-csv /data/backup/round3/synthetic_audit.csv \
  --out-json /data/backup/round3/synthetic_audit.json
```

---

## 5. "0 flagged" 的验收口径（需复核人确认）

`--geometric-only` 下统计的是**几何类 flag**：`ZERO_AREA / OUT_OF_BOUNDS / DUPLICATE / TINY / THIN`。

**为什么要关掉 CONFUSABLE**：`CONFUSABLE` 是"同图出现易混类"的共现启发式。合成图的目标图本来就可能含类 11/12（它们与类 13 互为易混邻居），按默认口径会把**完全正常的合成样本**标脏，永远达不到 0。

因此验收标准定为：

- [ ] **几何类 flag = 0**（硬性）
- [ ] `CONFUSABLE` **未评估**（`--geometric-only` 下脚本关闭了该判断，`confusable_enabled=false`）

> **措辞纪律**：合成集审计报告里**不得写"CONFUSABLE = 0"**——那是"评估过且没命中"的意思，而实际上是"根本没评估"。
> 正确写法：`CONFUSABLE 未评估（脚本以 --geometric-only 运行）`。
> 若将来要用共现口径，必须单独跑一次不带 `--geometric-only` 的审计，那时才能报数字。

> 复核人 2026-10-23 说明：CONFUSABLE 是目标图固有的共现属性，不是合成引入的缺陷；
> 排除含类 11/12 的目标图会改背景分布、反而引入新变量，故不做，也无需重新报备。


---

## 6. pytest 记录

命令与结果（服务器上执行）：

```bash
python -m pytest tests/test_class_copy_paste.py tests/test_label_audit.py -q
# 13 passed in 0.40s
```

本次改动后需**重跑一次**（改动点：派生 `data.yaml` 保留 `test` 条目；manifest 增加 `rotation/flip/pastes_per_synthetic_image/dest_sampling/mosaic_participation` 字段；审计脚本增加 `--geometric-only`）。

---

## 7. 判据（写死，事后不得改）

> ① 类 13 recall Δ > 0；
> ② 类 13 漏检率 Δ ≤ −3pt（val 23 GT 下 −3pt ≈ 0.69 个实例，**按漏检数至少 −1 执行**）；
> ③ 总 mAP50-95(M) Δ ≥ −0.002。
> 对照 = A0 在修复后 val 上的复评值。

- [ ] 判据②的换算口径（≥1 个实例）原样写进结论文档
