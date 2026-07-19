# BladeDefect 跨平台环境指南

项目推荐使用名为 `bladedefect` 的独立 Conda 环境，以隔离 Python、科学计算库和训练框架依赖。
项目路径和配置不绑定任何一台电脑，可在 Windows 工作站、Linux 服务器或学校算力服务器上使用。

## Windows 本地开发

在项目根目录打开 PowerShell：

```powershell
conda env create -f environment.yml
conda activate bladedefect
python -m pip install -e .
```

如果 PowerShell 禁止运行 Conda 激活脚本，可为当前用户启用已签名或本地脚本：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

这会修改当前 Windows 用户的 PowerShell 执行策略，请先确认符合所在单位的安全要求。

VSCode 有时会自动激活项目中的 `.venv`。请通过“Python: Select Interpreter”选择 Conda 的
`bladedefect` 解释器。典型路径示例为：

```text
D:\miniconda\envs\bladedefect\python.exe
```

实际位置取决于 Miniconda/Anaconda 的安装目录，可用 `conda env list` 查看，不要把该示例路径写入
项目配置。

## Linux 与算力服务器

克隆或上传项目后，在项目根目录执行：

```bash
conda env create -f environment.yml
conda activate bladedefect
python -m pip install -e .
```

建议始终从项目根目录启动命令。配置中的 `project_root`、数据集目录和输出目录均可使用相对路径或
绝对路径；项目内部统一使用 `pathlib.Path` 解析，不依赖 Windows 盘符或路径分隔符。

如果服务器使用模块系统，应先按管理员说明加载 CUDA/驱动相关模块，再激活 Conda 环境。训练任务
提交到调度系统时，也应在作业脚本中显式执行 `conda activate bladedefect`。

## PyTorch 与 CUDA

`environment.yml` 不强行锁死 CUDA 版 PyTorch。不同电脑和服务器的 NVIDIA 驱动支持不同 CUDA
版本，GPU 用户应先根据实际驱动安装匹配的 PyTorch，再安装本项目。

Windows 本地可尝试：

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
```

如果 NVIDIA 驱动支持 CUDA 12.8，也可通过阿里云 PyTorch Wheels 镜像安装固定版本，以改善国内网络
环境下的下载稳定性：

```powershell
python -m pip install torch==2.9.0+cu128 torchvision==0.24.0+cu128 torchaudio==2.9.0+cu128 --find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/ --no-index --timeout 1000 --retries 20 --no-cache-dir
```

该命令仅从指定镜像查找安装包，并固定为 CUDA 12.8 版本；执行前应确认 NVIDIA 驱动兼容 CUDA
12.8。CPU 环境或使用其他 CUDA 版本的机器不应直接套用此命令。

如果 CUDA 12.4 组合不稳定，可尝试：

```powershell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Linux 服务器同理，应按服务器驱动和 PyTorch 官方兼容矩阵选择 CUDA 版本；不要因为本地电脑使用
12.4 就假定学校服务器也使用相同版本。

## 设备选择

所有训练和推理入口共享同一套设备规则：

- `auto`：`torch.cuda.is_available()` 为真时使用 GPU `0`，否则使用 `cpu`。
- `0`：显式使用首张 GPU。
- `cpu`：显式禁用 GPU。

命令行示例：

```bash
blade-defect train --config configs/train.yaml --device cpu
blade-defect predict --weights runs/segment/train/weights/best.pt --source assets --device cpu
blade-defect evaluate --model runs/segment/train/weights/best.pt --data configs/data.yaml --device cpu
blade-defect ablation --config configs/ablation.yaml --device cpu
```

配置文件中可使用：

```yaml
device: cpu
```

## 不使用 Conda 的备选方式

如果环境由服务器管理员或容器统一管理，可使用现有依赖清单：

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 验证环境

```bash
python -m pytest
python -m compileall src
blade-defect --help
blade-defect evaluate --help
blade-defect eval --help
```
