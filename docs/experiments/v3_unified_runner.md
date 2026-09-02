# v3 baseline 统一训练入口

统一入口为 `scripts/run_v3_baselines.py`，固定读取
`configs/v3_baselines.yaml`。它调度原有 seg/OBB runner，不复制训练逻辑。

所有命令都从项目根目录执行，并使用当前已激活 Conda 环境中的 `python`；不要写死
Miniconda/Anaconda 的安装目录或某台机器上的解释器绝对路径：

```powershell
conda activate bladedefect
python -c "import sys; print(sys.executable)"
```

第二条命令只用于确认当前解释器确实来自目标环境。Linux、服务器和自定义 Conda 安装位置
同样使用后续命令；项目入口不依赖 Windows 盘符。

## 状态检查

```powershell
python scripts/run_v3_baselines.py status
```

状态检查不启动训练，输出版本匹配、数据可用性、实验状态与 `last.pt` 恢复能力。

## 首次训练与中断恢复

```powershell
# 15 类 YOLO11s-seg @ 960，目标总轮次 200
python scripts/run_v3_baselines.py run --task seg --device 0

# 15 类 YOLO11s-obb @ 960，目标总轮次 200
python scripts/run_v3_baselines.py run --task obb --device 0

# 顺序处理两项；已完成项跳过，未完成项自动恢复
python scripts/run_v3_baselines.py run --task all --device 0
```

意外中断时重新执行同一条 `run` 命令。只要当前阶段存在 `weights/last.pt`，就恢复
optimizer、scheduler、best fitness 和下一 epoch，继续到该阶段原定目标轮次。

## 完成后增加训练轮次

```powershell
# seg e200 完成后，从 best.pt 新建 50 epoch 微调阶段（累计目标 e250）
python scripts/run_v3_baselines.py extend `
  --task seg --add-epochs 50 --device 0

# OBB 同理；若要从最后一轮而非最佳轮权重开始
python scripts/run_v3_baselines.py extend `
  --task obb --add-epochs 50 --weights last --device 0
```

默认选择该任务累计轮次最大的已完成阶段作为父阶段。输出使用新目录，例如：

```text
runs/v3_yolo11s_seg_960_e200/       # 200 epoch 正式阶段，保持不变
runs/v3_yolo11s_seg_960_e250_ft1/   # 从 e200 best.pt 新增 50 epochs
runs/v3_yolo11s_seg_960_e300_ft2/   # 再新增 50 epochs
```

每个扩轮阶段在 manifest 中记录父 run、起始权重路径与 SHA-256、阶段轮次、累计目标
轮次和优化器策略。扩轮阶段若中断，重新执行同一条 `extend` 命令会命中同名目录并从
该阶段自己的 `last.pt` 完整恢复。

完成后的 Ultralytics checkpoint 不保留 optimizer，因此跨已完成阶段采用“继承模型
权重 + 新 optimizer/scheduler”的 fine-tune；这与意外中断时的完整 resume 是两种不同
语义。默认从 `best.pt` 扩轮，避免从性能较差的最后一轮继续；可用 `--weights last`
显式改为最后一轮。

## 固定环境

正式版本要求定义在 `configs/v3_baselines.yaml`，关键 Python 包锁定见
`requirements-v3.txt`。版本不匹配时正式训练拒绝启动；`--allow-version-mismatch`
只用于明确的非正式试验，并会由原 runner 记录实际环境。

新机器首次创建环境或同步已有环境：

```powershell
# 首次创建；environment.yml 中的 bladedefect 只是项目默认环境名
conda env create -f environment.yml

# 已有环境同步依赖（先退出正在使用的同名环境更稳妥）
conda env update --name bladedefect --file environment.yml

conda activate bladedefect
python -m pip check
python scripts/run_v3_baselines.py status
```

如需使用其他环境名，创建时可传 `--name <环境名>`，之后激活该环境并继续使用相同的
`python scripts/run_v3_baselines.py ...` 命令。统一入口只校验 Python/Ultralytics 版本，
不校验环境名称或 Conda 安装路径。

## 数据与磁盘保护

正式 seg 数据使用已验收的隔离重建版本
`datasets/v3-index-rebuild/blade-v3-grouped-202608/data.yaml`；正式 OBB 派生数据位于
`datasets/blade-v3-grouped-obb/data.yaml`。若任一数据缺失，`status` 会报告
`data_available=false`，因此不会误启动对应训练。

“索引数据集”只表示 split 和样本成员由索引文件确定，并不天然满足 Ultralytics 的
`images/labels` 目录布局。正式 OBB 激活入口因此只允许硬链接；如果权限或文件系统
不支持硬链接，立即失败，不再自动回退为全量图片复制。修复硬链接条件后再单独执行
OBB 激活流程。
