# Round 3 B-1 开跑 Checklist（仅类 13 + A0 复评 + 3a 测速）

> 状态：等两件事 → **负责人批复**（B-1 缩为仅类 13）+ **代码 diff 复核通过**
> 前置事实：类 13 重复标注已修（2026-10-23，删 3 个，val 25→23，全量 141→138）
> 本清单按复核人 2026-10-23 口径编写，执行时逐条打勾，不得跳步。

---

## 0. 开跑前必须齐的三样

- [ ] 负责人书面批复：B-1 范围 = **仅类 13**
- [ ] 代码 diff 已复核通过（阶段 1 产物）
- [ ] 企业工程师口径答复到位或明确"本轮不动类 10"（不影响 B-1，但影响结论文档怎么写）

---

## 1. 开机自检（5 分钟）

```bash
# 目录：任意
nvidia-smi --query-gpu=index,name,memory.total --format=csv
df -h /data
cd /root/BladeDefect && source /root/miniconda3/etc/profile.d/conda.sh && conda activate bladedefect
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version())"
```

- [ ] 两张 V100 可见；`/data` 挂载 492G
- [ ] `torch 2.9.0+cu128`、`cudnn 91002`（V100 必须锁 9.10，见 ENVIRONMENT 记录）
- [ ] 若 cudnn 显示 91901 → `activate.d/zz_cudnn_volta.sh` 没生效，先修再继续

---

## 2. 【必做】清 label cache —— 不清就等于没修标注

修复改了 3 个标签文件，但 YOLO 的 `*.cache` 可能仍是旧的。**复评和训练都会直接吃旧缓存，修复白做。**

```bash
# 目录：任意
# ① 先看有哪些缓存、时间戳是否早于修复时间（2026-10-23 18:17）
find /data/datasets/v3-index-rebuild -name "*.cache" -printf "%TY-%Tm-%Td %TH:%TM  %p\n" | sort

# ② 删掉（只删 .cache，不碰图/标签）
find /data/datasets/v3-index-rebuild -name "*.cache" -delete

# ③ 确认已清空
find /data/datasets/v3-index-rebuild -name "*.cache" | wc -l     # 期望 0
```

- [ ] 缓存已清空
- [ ] 记一笔：首次 run 会重建缓存，首轮多花 3–8 分钟，属正常

> 说明：ultralytics 缓存里带标签哈希，理论上会自动失效重建，但**不要赌**——Round 2 已经在这里栽过一次（两个进程抢缓存竞态）。删掉最干净。

---

## 3. A0 复评（同口径基线，约 10 分钟）—— 可与第 4 步并行

用 Round 2 的 A0 权重，在**修复后的 val** 上重跑一次 validate，**不用旧值**。

> **前置**：复评前必须先删除 val 的 cache（第 2 节已删全部 cache；若只删了 train，补删 val）。
> **并行**：A0 复评与第 3.5 节"建合成集"互不依赖（一个吃 GPU、一个吃 CPU+IO），可以同时进行以省时间；
> 但两者都要在**开训之前**完成，且都要先把 cache 清干净。

```bash
# 目录：/root/BladeDefect
python - <<'PY'
from ultralytics import YOLO
m = YOLO("/data/runs/v3_seg_a0_res960_ft_e25/weights/best.pt")
r = m.val(data="/data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml",
          imgsz=960, batch=4, split="val", project="/data/runs", name="v3_seg_a0_reval_fixedlabels", exist_ok=True)
print("mask mAP50-95 =", r.seg.map)
print("mask recall   =", r.seg.r)
PY
```

- [ ] 产出 `/data/runs/v3_seg_a0_reval_fixedlabels/metrics.json`
- [ ] **归档**：`cp /data/runs/v3_seg_a0_reval_fixedlabels/metrics.json /data/backup/round3/a0_reval_fixedlabels_metrics.json`
- [ ] 记下 **类 13 的 AP50-95(M)** 与 **类 13 漏检数**（后面判据要用）
- [ ] 与旧值（0.0845）对比，差异应很小（只删了 2 个 val 重复实例）

> 为什么必须做：B-1 的对照臂若用旧标注下的数值，等于拿两套 GT 比，判据②（漏检率）直接失效。

---

## 3.5 建合成集 + 审计（CPU，约 10 分钟）—— 可与第 3 步并行

> 与 A0 复评互不依赖（一个吃 GPU、一个吃 CPU+IO），可同时跑；但都必须**在开训之前**完成。
> 参数已冻结，见 `round3_b1_synthesis_spec.md`；**合成图必须是 345 张**（115 实例 × 3）。

```bash
# 目录：/root/BladeDefect
source /root/miniconda3/etc/profile.d/conda.sh && conda activate bladedefect

# ① dry-run 核对源实例数（期望 13: 115，planned_synthetic: 345）
python scripts/build_class_copy_paste.py --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
  --out-dir /data/datasets/derived/blade-v3-grouped-202608-b1cp13 --classes 13 --copies 3 --dry-run

# ② 正式生成
python scripts/build_class_copy_paste.py --data-yaml /data/datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml \
  --out-dir /data/datasets/derived/blade-v3-grouped-202608-b1cp13 --classes 13 --copies 3 --seed 42 --scale-min 0.8 --scale-max 1.2

# ③ 对合成集复跑审计（几何口径）
python scripts/audit_small_class_labels.py \
  --data-yaml /data/datasets/derived/blade-v3-grouped-202608-b1cp13/data.yaml \
  --classes 13 --splits train --geometric-only \
  --out-csv /data/backup/round3/synthetic_audit.csv \
  --out-json /data/backup/round3/synthetic_audit.json

# ④ CONFUSABLE 抽样截图（不作失败条件，仅留证）
python scripts/make_audit_sheets.py \
  --data-yaml /data/datasets/derived/blade-v3-grouped-202608-b1cp13/data.yaml \
  --classes 13 --splits train --top 20 \
  --out-dir /data/backup/round3/synthetic_audit_sheets
```

- [ ] manifest：`synthetic_images == 345`、`target_classes == [13]`、`seed == 42`、`rotation=none`、`flip=none`
- [ ] `synthetic_audit.json`：**几何类 flag = 0**
- [ ] `CONFUSABLE` 计数已记录（允许 >0）
- [ ] 抽样截图已生成
- [ ] 派生 `data.yaml` 的 `val` / `test` 指回原数据集（`grep -E '^(val|test):'` 核对）

---

## 4. B-1 训练（GPU0，约 11 小时）

**配置 = A0 原样**，唯一变量 = 类 13 定向 copy-paste。

> **参数以服务器实测 `args.yaml` 为准，不以本文档为准。** 下表只是计划值；
> 开工前必须与 `/data/runs/v3_seg_a0_res960_ft_e25/args.yaml` 逐项 diff（warmup / wd 等一律读文件）。

| 参数 | 计划值 |
|---|---|
| imgsz / batch / epochs / patience | 960 / 4 / 25 / 8 |
| optimizer / lr0 / lrf | AdamW / 5e-4 / 0.01（cos） |
| mosaic / close_mosaic | 1.0 / 10 |
| seed / workers | 42 / 8 |
| 起始权重 | 封存基线 best e89 |
| copy_paste（原生） | **0**（原生全类关闭，改用定向实现） |

**⚠️ `data=` 必须显式指向派生集**，不要用入口脚本的默认值（默认会拿原 `data.yaml`，那就变成没加合成图的 A0 复跑）：

```
data=/data/datasets/derived/blade-v3-grouped-202608-b1cp13/data.yaml
```

启动（单臂，不会有缓存竞态）：

```bash
tmux new-session -d -s b1 "cd /root/BladeDefect && source /root/miniconda3/etc/profile.d/conda.sh && conda activate bladedefect && python scripts/<入口> --data /data/datasets/derived/blade-v3-grouped-202608-b1cp13/data.yaml --device 0 2>&1 | tee /data/backup/round3_logs/b1.log"
```

**对拍（开训后立刻做，结果截图发复核人）**：

```bash
# 目录：任意
diff <(grep -E '^(imgsz|batch|epochs|patience|optimizer|lr0|lrf|cos_lr|warmup_epochs|weight_decay|mosaic|close_mosaic|seed|workers|copy_paste):' /data/runs/v3_seg_a0_res960_ft_e25/args.yaml) \
     <(grep -E '^(imgsz|batch|epochs|patience|optimizer|lr0|lrf|cos_lr|warmup_epochs|weight_decay|mosaic|close_mosaic|seed|workers|copy_paste):' /data/runs/<B1_RUN>/args.yaml)
# 期望：无输出（除 data 路径外逐项一致）
```

- [ ] 训练日志里能确认"只贴类 13"（离线合成：看 manifest，见第 7 节）
- [ ] `args.yaml` 逐项 diff 无差异（`data` 除外），**截图发复核人**
- [ ] 里程碑 `milestones.log` 记一条开机时间

---

## 5. 事项 3a 测速（GPU1，与 B-1 同期）

- [ ] **与 B-1 启动间隔 3–5 分钟**（Round 2 教训：避免 IO/缓存竞态）
- [ ] 用 Round 2 的 A1（1280）权重，按顺序测：
  1. FP16 推理
  2. TensorRT（装不上 → 退化 ONNX Runtime + FP16，**并在文档注明口径差异**）
  3. 按图分档（关键图 1280 / 一般图 960）
- [ ] Go/No-Go：**≥ 150 FPS** 才提 3b；不达标 → 关闭 1280 生产路线，不提 3b
- [ ] 测速结果存 `/data/backup/round3_logs/speed_*.json`

---

## 6. 判据（写死，事后不得改）

> B-1 范围按审计修正为仅类 13，判据写死：
> ① 类 13 recall Δ > 0；
> ② 类 13 漏检率 Δ ≤ −3pt（val 23 GT 下 −3pt ≈ 0.69 个实例，**按漏检数至少 −1 执行**）；
> ③ 总 mAP50-95(M) Δ ≥ −0.002。
> 对照 = **A0 在修复后 val 上的复评值**，不用旧值。

- [ ] 三条逐条计算并写进结论文档
- [ ] 判据②的"至少 −1 个实例"这个换算口径要**原样写进结论**，别只写百分比

---

## 7. 留痕与交付

**开训前一次性发复核人的四样**：

- [ ] `pytest` 13 项运行记录
- [ ] `a0_reval_fixedlabels_metrics.json`（已归档 `/data/backup/round3/`）
- [ ] `copy_paste_manifest.json`（含 `synthetic_images == 345` 等冻结字段）
- [ ] `synthetic_audit.json` + `synthetic_audit_sheets/`（几何 0 flagged + CONFUSABLE 计数与抽样图）

**开训后立刻补发**：

- [ ] `runs/<B1_RUN>/args.yaml` 与 A0 `args.yaml` 的逐项 diff **截图**（以实测文件为准，不用文档值）
- [ ] 确认实际生效的 `data` 指向 `blade-v3-grouped-202608-b1cp13/data.yaml`

**训练与测速完成后**：

- [ ] `milestones.log`：开机 / 清缓存 / A0 复评完成 / 建集合完成 / B-1 起跑 / 3a 起跑 / 结束
- [ ] B-1 的 `metrics.json` + `per_class_metrics.csv`
- [ ] 3a 的测速 json + 结论（达标与否）
- [ ] 结论文档按 Round 2 格式（错误分解 + 口径说明 + 双人复核）

---

## 8. Round 2 踩过、这里必须避开的坑

| 坑 | 规避 |
|---|---|
| 两进程同时启动 → `labels/*.cache` 竞态 | 本次 B-1 单臂训练；3a 测速与训练间隔 3–5 分钟 |
| cuDNN 9.19 不支持 V100 | `activate.d` 已固化，开机先验 `cudnn 91002` |
| 字体下载卡住 | `Arial.ttf` / `Arial.Unicode.ttf` 已预置在 `runs/.ultralytics-config` |
| 欠费停机丢 6 小时 | 开机前把账户余额充够（≥ ¥300），或开自动充值 |
| 公网 IP 变动 | 建议转弹性 EIP |
| 旧 cache 掩盖标注修复 | 第 2 节强制清缓存 |
