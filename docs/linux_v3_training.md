# v3 Linux 算力节点部署与校验

本页用于把 v3 封存基线和提分实验迁移到 Linux 工作站、云 GPU 或学校算力节点。
训练、模型选择和自动分析只读取 train/val；test 始终锁定。

## 1. 必须同步的目录

保持仓库内的相对目录结构，不要把 Windows 盘符写进 YAML：

```text
configs/
scripts/
src/
datasets/v3-index-rebuild/blade-v3-grouped-202608/
datasets/v3-index-rebuild/blade-v2/
datasets/blade-v3-grouped-obb/
runs/v3_yolo11s_seg_960_e200/weights/best.pt
runs/v3_yolo11s_obb_960_e200/weights/best.pt
```

seg 的 `train.txt`/`val.txt` 引用同级兼容目录 `datasets/v3-index-rebuild/blade-v2/`；
上传时必须复制该目录本身，不能只上传 data.yaml。OBB 数据中的图片和标签也必须保留
原有大小写。Linux 文件系统区分大小写，预检会报告清单中无法找到的引用。

若使用 `rsync`，建议保留硬链接以减少 OBB 数据占用：

```bash
rsync -aH --info=progress2 BladeDefect/ user@server:/workspace/BladeDefect/
```

不要跨机器复用 `normalized_data.yaml`、`normalized_data.*.txt` 或
`ultralytics_cache_views/`；这些是包含节点本地绝对路径的运行时文件，会在新节点自动重建。

## 2. 创建环境

先按照服务器驱动安装匹配的 CUDA 版 PyTorch，再安装项目依赖：

```bash
cd /workspace/BladeDefect
conda env create -f environment.yml
conda activate bladedefect
python -m pip install -e .
python -m pip check
```

正式入口要求 Python 3.10 和 Ultralytics 8.4.90。PyTorch/CUDA 版本由目标节点驱动决定，
不要复制 Windows 机器上的解释器路径或 CUDA wheel。

## 3. 开跑前强制预检

```bash
python scripts/run_v3_score_sweep.py preflight --require-cuda
```

默认全量检查两个任务的 82,096 个 train/val 清单引用，同时检查：

- Python、Ultralytics、PyTorch 和 CUDA；
- 项目根目录、基础配置和两个基线 `best.pt`；
- train/val 数据清单、manifest 数量/SHA-256 及所有图片引用；
- Windows 盘符、POSIX 绝对路径混用和不可移植实验 ID；
- test 未被读取。

只有输出 `"ok": true` 才能开始付费训练。临时排查可用 `--sample-data-check` 只检查
每份清单前 100 个引用，但正式开跑不得用抽样结果替代全量预检。

## 4. 训练与多节点拆分

单节点顺序运行：

```bash
python scripts/run_v3_score_sweep.py plan --stage all
python scripts/run_v3_score_sweep.py run --task all --stage 1 --device 0
python scripts/run_v3_score_sweep.py analyze
```

两节点并行时，每个节点只能写不同的 experiment_id：

```bash
# 节点 A
python scripts/run_v3_score_sweep.py run --experiment v3_seg_ft_cos_adamw_960_e30 --device 0
python scripts/run_v3_score_sweep.py run --experiment v3_seg_ft_smallobj_1280_e25 --device 0

# 节点 B
python scripts/run_v3_score_sweep.py run --experiment v3_obb_ft_cos_adamw_960_e30 --device 0
python scripts/run_v3_score_sweep.py run --experiment v3_obb_ft_smallobj_1280_e25 --device 0
```

集中结果时复制每个完整的 `runs/<experiment_id>/`，不要只复制权重。合并后在一台机器运行
`analyze`，统一生成排名和人工复核表。禁止两个节点同时写同一个实验目录，也不要使用
`--force` 覆盖已经完成的实验。

## 5. Linux 代码校验

```bash
python -m compileall -q src scripts
python -m pytest -q
python scripts/run_v3_score_sweep.py preflight --require-cuda
git diff --check
```

无 GPU 的 CI 节点可以省略 `--require-cuda`，但正式训练节点必须保留。项目使用
`pathlib` 和正斜杠 YAML；Windows 的 junction 回退只在 `os.name == "nt"` 时执行，
Linux 使用符号链接或原始绝对清单，不调用 `cmd`/`mklink`。
