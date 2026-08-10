# OBB 工程交接指南

本文描述当前代码实际支持的 OBB 执行链路，以及激活 `blade-v3-obb` 前必须补齐的接口。所有命令均从仓库根目录 `D:\Program\BladeDefect` 执行。

## 1. 目标链路

```text
YOLO-seg polygon label
        |
        v
scripts/convert_seg_to_obb.py
        |
        v
datasets/blade-v3-obb
        |
        v
scripts/check_obb_dataset.py
        |
        v
YOLO11s-obb smoke（1 epoch）
```

当前 v2 目录型数据已走通“转换 → validator → 预览”，但 v3 是索引型发布版，尚不能直接进入转换器的写出模式。执行 v3 正式转换前，必须先解决第 6 节的两个阻塞点；不要用重新随机划分或把 test 混入 train/val 的方式绕过。

## 2. 环境与只读预检

### 输入

- Python 环境：项目依赖已安装，且 `src` 可被当前环境导入。
- v3 索引配置：`configs/data.yaml`，其数据根目录指向 `datasets/blade-v3-grouped-202608`。
- v3 发布文件：`data.yaml`、`train.txt`、`val.txt`、`test.txt`、`dataset_manifest.json`，以及清单引用的图像和 polygon 标签。

### 命令

```powershell
python scripts/convert_seg_to_obb.py `
  --source-data configs/data.yaml `
  --dry-run-index-check
```

### 输出

- stdout JSON：`splits`、`total_samples`、`source_instances`、`converted_instances`、`membership_sha256`、`errors`、`valid`。
- 不生成 OBB 数据集，不修改标签。
- `valid` 为 false 时退出码为 1；必须先处理数据问题，不能继续正式转换。

注意：当前本机盘点时 `datasets/blade-v3-grouped-202608` 不存在，因此该命令需要在 v3 发布数据挂载/恢复后运行。

## 3. polygon → OBB 正式转换

### 当前写出模式要求的输入布局

`--source` 必须是已经按 v3 固定 membership 物化好的目录型 seg 数据集根目录，下文以 `<V3_SEG_DIRECTORY_ROOT>` 表示：

```text
<V3_SEG_DIRECTORY_ROOT>/
  images/train/**
  images/val/**
  labels/train/**/*.txt   # YOLO-seg: class x1 y1 x2 y2 ...
  labels/val/**/*.txt
```

该物化目录必须来自 v3 的 train/val 清单，且成员关系可与 v3 membership hash 对账。当前仓库没有生成该目录的受支持步骤，不能自行选择一个近似目录代替。

### 目标命令

在上述前置条件满足后执行：

```powershell
python scripts/convert_seg_to_obb.py `
  --source <V3_SEG_DIRECTORY_ROOT> `
  --output datasets/blade-v3-obb `
  --results results/obb_v3_activation
```

如果 `datasets/blade-v3-obb` 已存在，脚本会拒绝覆盖。只有确认旧目录可丢弃时才可显式追加 `--overwrite`；该选项会递归删除并重建目标目录。

### 生成文件

```text
datasets/blade-v3-obb/
  data.yaml
  images/train/**         # 从源目录复制
  images/val/**
  labels/train/**/*.txt   # class x1 y1 x2 y2 x3 y3 x4 y4
  labels/val/**/*.txt

results/obb_v3_activation/
  conversion_summary.json
  invalid_obb_labels.csv
```

`conversion_summary.json` 给出各 split 的图像、标签、源实例、转换实例、失败实例、修复告警和空标签统计。`invalid_obb_labels.csv` 同时记录 warning 与 error。转换失败策略是只丢弃失败实例，保留图像和同图其他有效实例，因此正式训练前必须人工确认审计结果，而不能只看脚本退出成功。

## 4. OBB validator 与可视化预览

### validator

输入目录：`datasets/blade-v3-obb/images/{train,val}` 与 `datasets/blade-v3-obb/labels/{train,val}`。

```powershell
python scripts/check_obb_dataset.py `
  --dataset datasets/blade-v3-obb `
  --report results/obb_v3_activation/obb_validation_report.json
```

生成 `results/obb_v3_activation/obb_validation_report.json`，并将同一 JSON 打印到 stdout。只有顶层 `valid: true` 才允许进入训练；任一 missing/orphan label、坏图、字段数、类别、数值、坐标范围、重复角点、面积或角点顺序错误都会使命令以 1 退出。

### 并排预览

`<V3_SEG_DIRECTORY_ROOT>` 与正式转换使用同一物化源目录：

```powershell
python scripts/visualize_obb_labels.py `
  --source-seg <V3_SEG_DIRECTORY_ROOT> `
  --source-obb datasets/blade-v3-obb `
  --output results/obb_v3_activation/conversion_preview `
  --count 100 `
  --seed 42
```

生成：

- `results/obb_v3_activation/conversion_preview/train/*.jpg`
- `results/obb_v3_activation/conversion_preview/val/*.jpg`
- `results/obb_v3_activation/conversion_preview/preview_summary.json`

预览左侧为 polygon，右侧为 OBB；OBB 四角标为 1–4。人工抽查应覆盖长条、近正方形、贴边、多实例与不同缺陷类别。

## 5. YOLO11s-obb smoke

当前入口：

```powershell
python scripts/run_obb_smoke.py `
  --config configs/experiments/obb_smoke_yolo11s_960.yaml
```

但当前配置的 `data` 是 `datasets/blade-v2-obb/data.yaml`。因此这条原样命令只验证 v2，不属于 v3 训练。v3 激活时应先由负责人新增/批准一个只改变数据与输出命名、保持现有训练参数不变的配置，例如 `<V3_OBB_SMOKE_CONFIG>`，再执行：

```powershell
python scripts/run_obb_smoke.py --config <V3_OBB_SMOKE_CONFIG>
```

必须保持的现有 smoke 训练参数为：

| 参数 | 值 |
|---|---:|
| task | `obb` |
| model | `yolo11s-obb.pt` |
| imgsz | 960 |
| epochs | 1 |
| batch | 4 |
| workers | 2 |
| device | 0 |
| seed | 42 |
| pretrained | true |
| cache | false |
| amp | true |

入口内部执行顺序：

1. 对 train/val 调用 `check_obb_dataset()`，失败即停止。
2. `YOLO("yolo11s-obb.pt").train(data=..., task="obb", ...)`。
3. 检查 `<save_dir>/weights/best.pt` 是否存在。
4. 用 best 权重执行 `val(task="obb")`。
5. 对第一张 val 图执行 `predict(task="obb")`。
6. 写 Markdown 与 JSON smoke 报告。

按现有路径约定，训练和报告类生成物为：

```text
runs/obb/<experiment-name>/
  weights/best.pt
  ...                    # Ultralytics 训练产物

results/obb_v3_activation/
  obb_smoke_report.md
  obb_smoke_report.json
  smoke_prediction/**
```

`yolo11s-obb.pt` 若本地不存在，Ultralytics 会尝试联网下载。历史 v2 smoke 正是在该下载步骤失败；离线环境需预先按项目模型解析规则提供该权重，但不要替换模型规格。

## 6. v3 激活阻塞点与交接决策

### A. 索引型 v3 不能直接写出 OBB

正式 v3 只保存 train/val/test txt 清单；当前转换器写出模式只接受实体 train/val 目录。后续实现应满足：

- 精确读取 v3 清单，不重新划分；
- 保持 `membership_sha256` 可核对；
- 明确 test 锁定策略；
- 不改变 `convert_polygon_to_obb()` 与 `order_obb_corners()`；
- 为索引写出补测试后再激活。

本次不实现该能力。

### B. smoke 配置仍绑定 v2

当前配置直接运行会训练 v2-obb。应新增 v3 专用配置或给入口增加经批准的数据路径覆盖方式；训练超参数保持原值。本次不修改配置或训练入口。

### C. test 尚未进入 OBB 转换契约

转换器、validator CLI 和 smoke 均只处理 train/val，而 v3 有锁定 test。是否生成 `images/test`/`labels/test`、何时解锁评估必须由数据负责人决定。未经批准不得把 test 用于训练、调参、阈值选择或 smoke 样本选择。

## 7. 每次交付前验证

```powershell
pytest
python -m compileall src scripts
git diff --check
```

再检查改动边界：

```powershell
git status --short
git diff -- scripts/convert_seg_to_obb.py `
  src/blade_defect/data/obb_check.py `
  configs/experiments/obb_smoke_yolo11s_960.yaml
```

本次交接的预期是第二条 `git diff` 无输出；仅文档发生变化。
