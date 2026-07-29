# 实验 Registry

实验运行器会为每次训练生成可追溯元数据，解决模型、数据集、代码版本和指标无法对应的问题。

## 输出文件

每个实验目录会包含：

```text
runs/<experiment_id>/
├── run_manifest.json
├── environment.json
├── metrics.json
└── ...
```

`run_manifest.json` 在模型初始化前写入，实验结束后更新为 `completed` 或 `failed`。如果进程被强制中止，文件会保留 `running` 状态，便于识别未正常结束的实验。

清单记录以下信息：

- 唯一 `run_id`、实验 ID、开始和结束时间、运行状态；
- 数据集 ID、数据哈希、数据清单及其来源；
- 模型、输入尺寸、轮数、批量大小、随机种子和运行设备；
- 配置文件路径、配置哈希及最终生效配置；
- Git commit、分支、当前 commit 对应的全部 tag、`git describe`、dirty 状态和 dirty 文件；
- 指标摘要、权重和预测文件路径及其 SHA-256；
- 运行警告和结构化失败信息。

`environment.json` 记录 Python、操作系统、PyTorch、TorchVision、Ultralytics、NumPy、OpenCV、CUDA、cuDNN 和 GPU 信息。

## 数据集哈希规则

若数据集包含 `dataset_manifest.json`，优先根据冻结清单中的样本列表哈希、标签哈希和筛选配置哈希生成数据集内容哈希。若没有冻结清单，则对 `data.yaml`、标签文件和训练/验证列表文件计算后备哈希。后备哈希不读取大体积图片内容。

正式实验应使用不会重复覆盖的实验 ID，例如 `exp_full_yolo11s_960_v1`。`run_id` 仍会为每次启动追加时间戳和 commit 短哈希，以便交叉核对日志和指标。
