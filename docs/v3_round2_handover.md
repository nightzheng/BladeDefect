# v3 Round 2 实验交接文档（方案 A 分辨率隔离实验）

> 交接日期：2026-10-12。上一棒：第一轮 score sweep（四组实验全部 reject，结论已复核归档）。
> **当前资源状态：实例与数据盘均已释放，仅存数据盘快照一个**（含 /data 全部内容：datasets 193G + backup 全部成果）。OSS bucket `belong-temp` 另有数据原件+封存基线归档。现役成本仅快照存储 ~¥25/月。
> 本文档分两部分：**一、前任需交付的物品清单**；**二、接手人执行步骤**。
> 所有文档以 GitHub 仓库 `experiments/v3-score-sweep` 分支（PR #1，合并后以 main 为准）上的版本为准。

## 一、前任需交付给接手人的东西

### 1. 访问权限与资源
- **阿里云控制台**：杭州 I 区操作权限（新建抢占式实例、从快照创建云盘、绑定 EIP）；RAM 角色 `belong-temp`（OSS 只读）
- **数据盘快照 ID**：【前任填写快照 ID】← 接手人第一步要用它创建数据盘
- **OSS**：bucket `belong-temp`（数据原件 + 封存基线归档，只读）
- **GitHub**：仓库 `nightzheng/BladeDefect` 读写权限；PR #1（https://github.com/nightzheng/BladeDefect/pull/1 ）
- **沟通渠道**：项目群 + 负责人 + 复核人（第二方复核传统需延续）

### 2. 必读文档（按顺序读）
| 顺序 | 文档 | 作用 |
|---|---|---|
| 1 | `docs/v3_next_round_proposals.md` | 方案 A 全部设计 + **开机前核对清单**（变量受控表/判定标准/预算上限） |
| 2 | `docs/v3_score_sweep_round1_conclusions.md` | Round 1 证据与口径定义（漏检=未达匹配阈值） |
| 3 | 本文档 §三 红线 | 不可违反事项 |
| 4 | 快照内 `/data/backup/system_rebuild/DISASTER_RECOVERY.md` | 系统环境重建手册（本轮必经，已实战验证 ~40 分钟） |

### 3. 关键事实告知
- **预算**：方案 A 已获原则同意；申报上限 **¥500（超出即停上报）**；双卡并行预计 ~15h ≈ ¥280，另加重建环境 ~40 分钟
- **封存基线（禁止改动）**：重建后位于 `/data/runs/v3_yolo11s_seg_960_e200`（best e89）与 `v3_yolo11s_obb_960_e200`（best e81）
- **初始权重**：两臂统一从 SEG 基线 best e89 微调（与 Round 1 各实验同起点，保证可比）
- **环境基准**：重建以快照内 `/data/backup/system_rebuild/pip_freeze.txt` 为准（`bladedefect_env.tar` 尾部损坏勿用）；目标环境 = conda `bladedefect`（Python 3.10.22 / torch 2.9.0+cu128 / ultralytics 8.4.90 / shapely 2.1.2）
- **代码仓库**：以 GitHub 远端为准克隆（本地无仓库——原 `/root/BladeDefect` 已随实例释放）；分支 `experiments/v3-score-sweep`
- **网络**：实例直连 GitHub push 不稳定；`gh-proxy.com` 前缀只能 fetch/clone，不能 push；pip 走阿里云内网镜像

## 二、接手人执行步骤

### 第 0 步：报到与复核安排
- 群内报到，确认负责人已批复方案 A 核对清单（变量表/判定标准/预算上限）
- 与第二方复核人约定复核节点（结果出来后）

### 第 1 步：重建环境（~1 小时，全部为按手册操作）
1. 控制台新建抢占式实例：**杭州 I 区，16 vCPU + 2×Tesla V100 16G（gn6v-c8g1.4xlarge）**，镜像按 DISASTER_RECOVERY.md 指定；绑定新 EIP
2. **从数据盘快照创建 ESSD 云盘**（500G），挂载到实例并 `mount` 到 `/data`，写入 `/etc/fstab`
3. 按 `/data/backup/system_rebuild/DISASTER_RECOVERY.md` 七步重建系统环境（conda env、pip 依赖以 `pip_freeze.txt` 为准）
4. 核验：
```bash
nvidia-smi                        # 两张 V100 可见
df -h /data                       # 数据盘已挂载，datasets/runs/backup 齐全
conda activate bladedefect
git clone https://gh-proxy.com/https://github.com/nightzheng/BladeDefect.git /root/BladeDefect
cd /root/BladeDefect && git checkout experiments/v3-score-sweep
python -m pytest tests/ -q        # 应 243 全过
```

### 第 2 步：准备实验配置（不开 GPU）
- 参照 `configs/experiments/v3_yolo11s_seg_960_e200.yaml` 新建两臂配置，**逐项对照提案文档中的变量受控表**：
  - A0：`imgsz 960, batch 4, epochs 25, patience 8, lr0 5e-4 AdamW+cos, mosaic 1.0, close_mosaic 10, 无 copy_paste, seed 42`
  - A1：仅 `imgsz 1280`，其余与 A0 完全一致
  - 初始权重均指向封存基线 best e89 的 `.pt`（**只读引用，不移动不覆盖**）
- workers 不写进基线 YAML；运行时用 CLI `--workers 8` 覆盖

### 第 3 步：开跑（双卡并行）
```bash
tmux new -s a0   # GPU0 跑 A0
tmux new -s a1   # GPU1 跑 A1
```
- A0 前 2 epoch 顺带做 workers 8/12/16 吞吐记录（提案文档遥测项）
- 跑前在群里发开机通知（时间、两臂配置 hash、预计费用）

### 第 4 步：监控
- 里程碑事件记入 `results/v3_score_sweep/milestones.log`
- **OOM 或抢占中断：先上报再决策**；中断恢复按现场笔记，失败现场不静默重跑（留 `.aborted-*` 标记）
- 费用每半天对一次控制台账单，逼近 ¥500 上限立即停机上报

### 第 5 步：评估与判定（两臂都结束后）
```bash
# 导出两臂全量 val 预测（conf=0.25，沿用 Round 1 导出脚本）
# 然后统一跑错误分解：
python scripts/decompose_val_errors.py out.json a0=runs/<A0>/validation_predictions.json a1=runs/<A1>/validation_predictions.json
```
- 按核对清单的三条判定标准下结论（①类 8/9/14 AP50-95(M) 全正 ②漏检率降 ≥3pt ③总 mAP50-95(M) Δ≥-0.002），**不得临时修改标准**
- 填写提案文档方案 A 的"实际结论"，撰写 Round 2 结论文档（沿用 Round 1 格式：错误分解 + 口径说明 + 双人复核）

### 第 6 步：收尾
1. 成果同步 GitHub（runs/results/数据集/.pt 不入 Git，只交文档和代码）
2. 复核人复核通过后发负责人
3. 新成果备份到 /data/backup（并视负责人指示传 OSS）后：**负责人确认归档前只停机不释放**；确认后做新数据盘快照 → 释放实例与数据盘
4. 参考成本：停机保留数据盘 ~¥10/天；快照归档后 ~¥25/月；全释放归零（需先确认 OSS 归档）

## 三、红线（全程有效，不可违反）

1. test 集始终锁定（任何文档注明 test_used=false）
2. 不动封存基线目录 `/data/runs/v3_yolo11s_*_e200`
3. 失败实验不用 --force 静默重跑，现场留 `.aborted-*`
4. 改参须两人 + 负责人确认；每轮实验先有书面假设再开机
5. 数据集 / .pt / 完整 runs/results 不进 Git
6. OOM 先上报再决策

### 第 2 步：准备实验配置（不开 GPU）
- 参照 `configs/experiments/v3_yolo11s_seg_960_e200.yaml` 新建两臂配置，**逐项对照提案文档中的变量受控表**：
  - A0：`imgsz 960, batch 4, epochs 25, patience 8, lr0 5e-4 AdamW+cos, mosaic 1.0, close_mosaic 10, 无 copy_paste, seed 42`
  - A1：仅 `imgsz 1280`，其余与 A0 完全一致
  - 初始权重均指向封存基线 best e89 的 `.pt`（**只读引用，不移动不覆盖**）
- workers 不写进基线 YAML；运行时用 CLI `--workers 8` 覆盖

### 第 3 步：开跑（双卡并行）
```bash
tmux new -s a0   # GPU0 跑 A0
tmux new -s a1   # GPU1 跑 A1
```
- A0 前 2 epoch 顺带做 workers 8/12/16 吞吐记录（提案文档遥测项）
- 跑前在群里发开机通知（时间、两臂配置 hash、预计费用）

### 第 4 步：监控
- 里程碑事件记入 `results/v3_score_sweep/milestones.log`
- **OOM 或抢占中断：先上报再决策**；中断恢复按现场笔记，失败现场不静默重跑（留 `.aborted-*` 标记）
- 费用每半天对一次控制台账单，逼近 ¥500 上限立即停机上报

### 第 5 步：评估与判定（两臂都结束后）
```bash
# 导出两臂全量 val 预测（conf=0.25，沿用 Round 1 导出脚本）
# 然后统一跑错误分解：
python scripts/decompose_val_errors.py out.json a0=runs/<A0>/validation_predictions.json a1=runs/<A1>/validation_predictions.json
```
- 按核对清单的三条判定标准下结论（①类 8/9/14 AP50-95(M) 全正 ②漏检率降 ≥3pt ③总 mAP50-95(M) Δ≥-0.002），**不得临时修改标准**
- 填写提案文档方案 A 的"实际结论"，撰写 Round 2 结论文档（沿用 Round 1 格式：错误分解 + 口径说明 + 双人复核）

### 第 6 步：收尾
1. 成果同步 GitHub（runs/results/数据集/.pt 不入 Git，只交文档和代码）
2. 复核人复核通过后发负责人
3. **负责人确认成果归档前：只停机，不释放实例/数据盘**；确认后：核 OSS bucket → 数据盘快照 → 释放实例 → 快照完成后释放数据盘
4. 停机状态下成本：数据盘 ~¥10/天；快照归档后 ~¥25/月

## 三、红线（全程有效，不可违反）

1. test 集始终锁定（任何文档注明 test_used=false）
2. 不动封存基线目录 `/data/runs/v3_yolo11s_*_e200`
3. 失败实验不用 --force 静默重跑，现场留 `.aborted-*`
4. 改参须两人 + 负责人确认；每轮实验先有书面假设再开机
5. 数据集 / .pt / 完整 runs/results 不进 Git
6. OOM 先上报再决策
