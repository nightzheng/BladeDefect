# blade-v2 正式数据集版本说明

> 负责人：黄晨婧  
> 冻结时间：2026-07-23 18:36:29（Asia/Shanghai）  
> 数据集目录：`E:\BladeDefect\BladeDefect-code\datasets\blade-v2`

## 1. 版本结论

`blade-v2` 保留原有 `train`、`val` 划分，本周不随机创建 `test`。最终训练清单包含48,291张图片，其中train 43,463张、val 4,828张；对应48,291个标签文件和49,471个缺陷实例。5张确认无缺陷图片保留为空标签负样本。

严格校验已经通过：清单内图片完整像素解码失败0张、图片标签缺失0、孤立标签0、非法类别ID 0、polygon点数异常0、越界坐标0。详细结果见 `results/dataset_review/dataset_validation.json`。

## 2. 数据来源与存储方式

- 原始图片：`D:\images`，保持只读；
- 原始标签：`D:\labels`，保持只读；
- 新标签：实际复制到 `datasets/blade-v2/labels/train|val`；
- 新图片入口：`datasets/blade-v2/images/train|val` 是指向原始只读图片目录的NTFS目录联接；
- 有效样本由 `train.txt`、`val.txt` 和 `configs/dataset_filter.yaml` 共同固定。

原图总体积约180.15GB，而生成时E盘可用空间约133.98GB，因此不能再复制一份完整图片。目录联接只提供只读访问入口，不会修改或覆盖D盘图片。训练和统计必须使用 `train.txt`、`val.txt`，不能直接枚举联接目录中的全部文件，否则会把已筛选图片重新计入。

## 3. 筛选与修复

`dataset_filter.yaml` 共记录30张图片：

- 25张排除：9张原缺标签问题图片、16张损坏或截断图片；
- 5张保留为负样本；
- 16张损坏图片中，第一周发现10张（6张0字节、4张文件头异常），第二周完整像素解码又发现6张截断图片。

第二周对131张hard error图片逐张查看原图和异常多边形局部放大图。131张均显示多边形与可见缺陷及类别一致，越界顶点位于图像边缘外侧，因此全部记录为 `repair_confirmed`，没有统一删除图片。新标签中共修复：

- 747个soft error坐标；
- 202个经人工复核确认的hard error行内坐标；
- 修复后越界坐标为0。

逐张决策见 `results/dataset_review/hard_error_review.csv`，修改前后坐标见 `hard_error_repair_log.csv` 和 `soft_error_repair_log.csv`。原始标签没有修改。

## 4. 版本指纹

以 `dataset_manifest.json` 为准：

```text
filter_config sha256:
4039add7140c37d03d2fa66eeb2a10adbf1f1b008d475cb697a8a870ccc92017

hard_error_review sha256:
e3f97c142da9f008884d646e58098ce7a9d0f371c9bcd965001974dfa6c71ad3

train/val sample lists sha256:
bea593052c575ab677dd7e697a5876f109497948f106ef2da7fcbae1b5c80c8f

frozen labels aggregate sha256:
065efdfffe956800c25a85f70e3a5860a02c23cf7294a144a00db633ce53168c
```

## 5. 复现命令

```powershell
cd E:\BladeDefect\BladeDefect-code

python scripts\review_hard_errors.py `
  --hard-csv results\dataset\hard_error_coordinates.csv `
  --images D:\images `
  --output results\dataset_review `
  --decision-config configs\hard_error_review.json

python scripts\freeze_dataset.py `
  --images D:\images `
  --labels D:\labels `
  --filter-config configs\dataset_filter.yaml `
  --hard-review results\dataset_review\hard_error_review.csv `
  --coordinate-csv results\dataset\out_of_bounds_coordinates.csv `
  --output datasets\blade-v2 `
  --review-output results\dataset_review `
  --seed 42 `
  --workers 16 `
  --command "见本文件中的freeze_dataset命令"
```

从空目录首次生成时不要加 `--resume`。若在标签已完整生成、仅全量验证或manifest阶段被中断时续跑，才使用 `--resume`。图片目录联接需在生成后建立：

```powershell
New-Item -ItemType Junction `
  -Path datasets\blade-v2\images\train `
  -Target D:\images\train

New-Item -ItemType Junction `
  -Path datasets\blade-v2\images\val `
  -Target D:\images\val
```

重新统计正式版本：

```powershell
python scripts\analyze_dataset.py `
  --images datasets\blade-v2\images `
  --labels datasets\blade-v2\labels `
  --data datasets\blade-v2\data.yaml `
  --output results\dataset_v2 `
  --filter-config configs\dataset_filter.yaml `
  --workers 16
```

## 6. 当前限制

泄漏风险检查发现18个文件名前缀组同时出现在train和val，并生成2,404对相邻航拍序列候选、790对感知哈希完全相同候选和26,122对低距离近重复候选。这些是待复核风险，不等同于已确认重复。`blade-v2`当前保留原划分，后续如负责人决定按叶片或航拍序列重新分组，应生成新版本而不是覆盖本版本。
