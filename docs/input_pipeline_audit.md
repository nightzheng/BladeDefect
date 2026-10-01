# BladeDefect 训练输入流水线专项审计（2026-10-01）

## 1. 当前瓶颈分析

主要嫌疑是 worker 内的大图 JPEG 解码、resize、CPU augmentation，以及等待 worker 队列。用户提供的 278ms/图对应单进程约 3.6 图/秒；提高 workers 有助于并行，但不能按 vCPU 数线性保证加速。Mosaic 会请求多张图，进程内有小型图像 buffer；磁盘/page cache、CPU 争用、mask rasterization 都影响实际吞吐。21h/34h 是用户提供的整组耗时，本次没有将它们当作实测 epoch 时间或推导加速比例。

```text
原图磁盘 → worker: 读取字节 → OpenCV JPEG decode → resize
       → YOLODataset.get_image_and_label → CPU augmentation / 分割 mask / 格式化
       → worker collate → InfiniteDataLoader 队列与主进程 pin-memory
       → Trainer.preprocess_batch: non_blocking CPU→CUDA + float / 255
       → GPU model / loss / backward
```

最可能等待点：主进程从 iterator 取下一个 batch。GPU 利用率低本身不能区分解码、I/O、增强热点，必须联合 fetch gap、CPU、iowait、磁盘吞吐分析。预取能隐藏延迟，不能提高饱和 CPU 的长期生产速度。

## 2. 当前 DataLoader 实现

审计依据包括仓库源码、项目专用 Conda 环境的 Ultralytics **8.4.90** / PyTorch **2.9.0+cu128** 源码；系统默认 Python 另装 8.4.81 / CPU Torch，不能混用。requirements-v3.txt 固定 8.4.90，pyproject.toml / requirements.txt 只有下限，服务器务必记录实际版本。

| 检查项 | 当前行为 |
|---|---|
| `blade-defect train` | cli._train → 项目 SegmentationTrainer.from_config/train → YOLO.train(task='segment') |
| Trainer | YOLO task_map 选择官方 segmentation.SegmentationTrainer，继承 DetectionTrainer/BaseTrainer |
| Dataset | build_yolo_dataset → YOLODataset(task='segment') → BaseDataset |
| DataLoader | DetectionTrainer.get_dataloader → build_dataloader → InfiniteDataLoader |
| batch/workers | 平铺 YAML kwargs；configs/train.yaml 是 batch=8、workers=4。官方未指定时 workers=8 |
| run-all 参数优先级 | batch/workers：显式 CLI/API 覆盖 > 注册表 > YAML/官方默认；注册表当前不定义workers；cache/profile仍从YAML继承 |
| pin_memory | build_dataloader 默认 True，检测到 CUDA 设备时开启 |
| persistent_workers | PyTorch 标志默认 False，但 InfiniteDataLoader 保存 iterator，跨 epoch 实际复用 worker |
| prefetch_factor | 官方工厂 workers>0 时固定为 4，workers=0 时 None |
| 有效 workers | 8.4.90 取请求值、CPU数/可见CUDA卡数、batch数量的最小值；单 batch loader 强制 0。CPU/MPS 训练还会将 workers=0 |
| val workers | 请求 train workers × 2，再受工厂上限约束；验证 loader 可在训练期间保留独立 worker 池 |
| CPU→GPU | 官方 preprocess_batch 对 tensor 使用 CUDA non_blocking；无需新增传输代码 |
| 图像库 | Ultralytics imread 包装：读取字节 + cv2.imdecode；PIL 用于冷启动 verify/shape/EXIF |
| augmentation | CPU worker 执行 mosaic、仿射、HSV、翻转、polygon/mask、格式化；normalize 及已配置的 multi_scale 在 GPU 预处理 |
| OpenCV 线程 | ultralytics.utils 导入时已有 cv2.setNumThreads(0)，本机 getNumThreads()=1。不能据此声称存在 OpenCV 多线程嵌套 |
| 其他 CPU 线程 | Torch/BLAS、训练主进程、验证池、后台系统线程仍可能争用；C 组限制 OMP/MKL/OpenBLAS 并记录主进程 Torch 线程 |
| 每 epoch 初始化 | 正常不重建 Dataset、不重新扫描；close_mosaic 时 reset iterator；OOM/NaN 恢复等异常路径另论 |

Dataset 每次训练初始化都会枚举图片（目录递归或读取 txt）、排序、少量随机文件速度探测；get_labels 计算原生 hash，hash 是路径及文件总大小，不是逐图 SHA256/mtime 检查。标签 `.cache` 命中跳过全量 PIL 验证与 shape probing；缺失/版本或 hash 不符则用 ThreadPool(NUM_THREADS，最多8)检查全部标签及图片，存储 shapes。标签 cache 路径取首个标签父目录的 `.cache`，临时 data YAML/txt 不会自动导致随机 cache 路径。不同 split/清单若共用同一标签父目录，会争用同一路径并交替失效；本次不改第三方缓存规则。

补充入口检查：README当前正式v3入口是 `scripts/run_v3_baselines.py`，seg分支调用run_full_primary→相同项目SegmentationTrainer，因而可使用新增profile；OBB分支经run_v3_obb_baseline直接调用官方YOLO.train，不经过此分割封装，不能直接给其YAML添加pipeline_profile。本次没有改变OBB入口或算法。已有run_throughput_calibration.py支持workers/batch短测，但硬编码部分参数并将总epochs直接设短；保留它兼容既有用途，新工具用于保留原配置训练前缀的专项比较。

`cache: false` 控制图片全量 cache，**不关闭标签 cache**，也仍有每 worker 的 mosaic 小 buffer。正式 run-all 的 strict label gate 每次命令在实验循环前执行一次，未逐 epoch 执行；它扫描 train/val 和标签但不全量解码。dataset metadata 也是每命令一次，有冻结 manifest 时直接使用；无有效 manifest 时 fallback 对标签/清单内容做 hash，未对 180GB 原图逐一内容 hash。scripts/cache_dataset_validation.py 是独立数据门禁工具，不在这两个训练入口里自动调用。

## 3. 已实施优化

1. 统一 CLI 增加 `train --workers N` 和 `experiment run-all --workers N`，直接使用官方参数；未指定时保持 YAML/旧默认。run-all 的有效 workers 写入既有 manifest。
2. 增加平铺 YAML `pipeline_profile: true`，默认关闭。通过官方 callback 统计 epoch/train wall time、平均 host iteration、fetch gap、GPU allocated/reserved memory、实际 batch/workers/imgsz、版本和线程配置；每 epoch 写一行 `pipeline_performance.jsonl`，不 CUDA synchronize、不改 Dataset、不采样或消耗随机数。结束/异常清理自身 callback，避免后续训练重复注册。
3. 独立 benchmark 工具支持 A/B/C/D、单独新进程、独立输出配置/log、原配置 SHA256、JSON plan、汇总 CSV。保留原计划总 epochs，用 stop callback 运行前3～5 epoch，保留原学习率/optimizer选择/close_mosaic计划。每组重新加载相同初始模型，不 resume。
4. C 组验证原生优化并限制 CPU 线程预算，而非强开 persistent_workers/prefetch。B→C 唯一实验因素是 CPU 线程预算。A 固定到任务要求的 imgsz=960、batch=8；源码 YAML 当前为640，因此 A 是960条件下的现有加载配置。
5. benchmark 执行前只读检查所有 train/val 图片：缺少 JPEG EOI 时拒绝进入官方可能重编码的冷验证；检测已有同名 `.npy` 时拒绝隐式 image cache 干扰。此预检不写入数据、不建立图片缓存、不计入 epoch 吞吐。

## 4. 未实施但可考虑的优化

- 没有添加 `pin_memory` / `persistent_workers` / `prefetch_factor` YAML 配置：8.4.90 官方训练配置没有这些键，工厂虽支持 pin_memory 参数，Trainer 未暴露；其他关键行为已有。未 monkey patch、未改 site-packages、未增加复制第三方 factory 的 override。
- 没有再次设置 OpenCV 线程；当前已限制。服务器 profile 记录运行时状态，如果外部程序更改 OpenCV 设置，需再调查。
- RAM cache：官方保存训练尺寸的 resized images，180GB压缩原图大小不能直接等同 RAM 需求。需计入训练/验证、增强临时数组、worker复制、预取、page cache；有框架非确定性提醒，保留 false，不推荐60GB机器直接开启。
- disk cache：保存完整尺寸、OpenCV 解码后的 uint8 BGR `.npy`，np.load 跳过 JPEG 解码，仍照常 resize/增强。不属于缩图或二次 JPEG 编码，但会新增旁边文件。
  - 按 3632×5456×3×48000 = **2,853,531,648,000 bytes（2.85TB / 2.60TiB）**，外加原图180GB+、系统、模型、日志；官方默认还检查50%安全裕量。500GB ESSD 不适合全量。
  - 仓库数据卡写明train=33,804、val=7,244、test=7,243。正常训练只缓存train/val；若两者均为该尺寸，仍约2.44TB，依然远超500GB。实际以配置指向的图像shape/数量为准，test未参与本次工具。
  - 解码下限估计：48000×0.278s≈3.7 CPU小时，加上 TB 级写入与竞争；实际创建时间必须在目标机测，不能用该估计替代墙钟测量。
  - 原生已有 `.npy` 会优先于 JPEG 被读取，**即使 cache=false**；不检查源 JPEG 内容 hash/mtime。旧源文件、色彩通道变化、同名不同后缀图片等存在陈旧/碰撞风险，不能直接认为可复现。
  - 有更大数据盘后，可单独批准无损缓存方案，再核对 JPEG 解码数组和 `.npy` 逐像素一致性及容量；本次没有建立任何全量缓存。
- 不换 JPEG decoder、不迁移增强到 GPU、不改变 mosaic、mask、插值、imgsz、seed，不生成1920px派生图。此类方案有训练内容或随机增强风险，只列候选。
- 不取消 gate/hash/标签 cache 有效性检查；启动扫描不能解释所有 epoch 的长期等待，应先量化再讨论减少跨实验重复开销。

## 5. 修改文件、原因和回滚

| 文件 | 原因 |
|---|---|
| src/blade_defect/cli.py | 可兼容地覆盖 workers，拒绝负值 |
| src/blade_defect/experiment/runner.py | 接收覆盖并记录有效配置 |
| src/blade_defect/models/trainer.py | 消费项目 profile 参数，不传入官方配置；使用 scoped callbacks |
| src/blade_defect/models/performance.py | 默认关闭的低侵入 host 诊断 |
| scripts/benchmark_input_pipeline.py | 可执行 A/B/C/D 计划、只读预检、短测及 CSV 汇总 |
| configs/train.yaml | 仅注释说明新能力，数值全部保持 |
| tests/test_input_pipeline.py | 配置口径、callback、预检安全与 CLI 回归 |
| tests/test_experiment.py | 原配置/覆盖 workers 的 manifest 与调用回归 |
| docs/input_pipeline_audit.md | 审计、风险、命令与建议 |

回滚运行行为：删除/关闭 pipeline_profile，不传 --workers，恢复原线程环境；正式 YAML 的默认值从未变动。代码回滚：审阅本任务 diff 后撤销上述已跟踪文件的本次补丁，移除三个新增 Python/测试文件与本报告；不要盲目覆盖后来修改。生成结果仅在 runs 下，均可保留；没有正式图片/标签修改要回滚。

风险：增加 workers 会改变 worker 随机流的样本归属，同 seed 不保证逐次增强像素及权重完全一致，但未改变增强定义/概率/算法；更多 worker/prefetch 会增加 RAM 与 /dev/shm、进程数和磁盘竞争；C组主线程限制可能反而降速。host fetch gap 含 loop/callback/调度开销，不能视作纯解码时间；异步 GPU 使 host step 不等于 GPU kernel 时间。GPU utilization/CPU/iowait由外部低频采集，精确拆分 decode/augmentation需要另行短期 profiler，未在正式路径持续侵入。

## 6. 测试结果

本机项目专用环境可用 RTX 5070 Ti，目标机器仍是 Ubuntu/A10/8或16vCPU。单测覆盖 scoped callback、失败清理、关闭诊断无注册、纯项目参数不泄漏给 YOLO、batch=12分离、算法超参保持、拒绝JPEG自动重编码与隐式npy、旧CLI和run-all manifest兼容。另使用 runs 下全新合成小图做官方 CUDA Trainer/多进程端到端 smoke；这不是正式数据缩图，亦不是吞吐 benchmark。

最终全套单测 **226 passed**，2条既有Matplotlib布局warning；git diff --check通过。官方CUDA smoke完成3epoch，实际workers=2、pin_memory=true、prefetch_factor=4，诊断3行无重复，合成图片/标签hash一致。本次没有在正式180GB数据上运行性能比较，**没有可宣称的加速倍率/GPU提升结果**。

## 7. 云服务器 benchmark 命令

在项目目录及原训练环境执行，先确认版本与实际 CPU 配额。通用依赖允许升级，因此固定与正式训练相同的环境，不在此任务中自动升级。

```bash
python -c "import torch, cv2; from importlib.metadata import version; print('torch', torch.__version__, 'ultralytics', version('ultralytics'), 'cuda', torch.cuda.is_available())"
python -m pip install -e . --no-deps

# 仅生成配置/计划；不训练、不建立图片cache。
python scripts/benchmark_input_pipeline.py --config configs/train.yaml --workers 4 8 12 16 --epochs 5 --output runs/pipeline-plan

# 16vCPU：A=现有workers，B=worker扫描，C=相同workers+线程预算1。
python scripts/benchmark_input_pipeline.py --config configs/train.yaml --workers 4 8 12 16 --epochs 5 --device 0 --execute --output runs/pipeline-16c
python scripts/benchmark_input_pipeline.py --summarize runs/pipeline-16c

# 8vCPU：先用4/6/8，不让16个worker与实际CPU配额竞争。
python scripts/benchmark_input_pipeline.py --config configs/train.yaml --workers 4 6 8 --epochs 5 --device 0 --execute --output runs/pipeline-8c
python scripts/benchmark_input_pipeline.py --summarize runs/pipeline-8c
```

输出目录必须不存在。各 case日志、展开的 YAML、计划在输出根目录，JSONL/官方 args.yaml/results.csv/checkpoints 在 training/case_name。训练失败立即停止后续 case，日志供排错。先运行 A/B 筛选，再用 --only 针对最佳 B/C 做重复；示例：`--workers 8 --only B_workers_8 C_workers_8_threads1`，更换新 output 重复2～3次，交替顺序降低 page cache、温度和共享资源偏差；--only的输入顺序不会改变生成的执行顺序，可分别调用两个命令交替运行。不删除内核page cache、不改数据。

从另两个终端低频观察并保存：

```bash
mkdir -p runs/pipeline-monitor
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,power.draw --format=csv -l 1 > runs/pipeline-monitor/gpu.csv
# Ctrl+C 结束采集；日志包含时间，可与训练log对应。
mpstat -P ALL 1 > runs/pipeline-monitor/cpu.txt
iostat -xz 1 > runs/pipeline-monitor/io.txt
# 查看主进程及loader进程/线程；pidstat和mpstat/iostat来自服务器已有sysstat。
pidstat -u -r -d -w -t -C python 1 > runs/pipeline-monitor/process.txt
df -h /dev/shm
```

不向训练进程附加每步CUDA同步。若缺少sysstat，可先用 `top -H` / `vmstat 1` / nvidia-smi 观察。关注全核busy还是iowait、磁盘await/吞吐/队列、swap、RAM、worker线程数量，GPU低谷是否与fetch gap一致。GPU显存看持续日志，JSONL只给epoch末allocated/reserved，并非峰值或全部NVML占用。

汇总默认5epoch比较4/5epoch的median，3epoch只能给暖机中粗略结果；若改过warmup_epochs需手工选择相应稳定区间。先比较train_wall，再比较包含val/checkpoint的epoch_wall。确认实际batch未因OOM恢复而降低、workers_effective符合预期；CPU配额可能小于os.cpu_count，容器需结合cpuset/quota看。综合稳定images/s、fetch gap、GPU低谷和内存，而非只取最高瞬时GPU利用率。保留第一次启动/标签cache冷建成本，随后使用相同cache状态比较稳态，不能将A的冷建惩罚算作B加速。

## 8. 推荐配置

保持项目平铺风格，以下是**待测起点**，只替换 workers/cache，不改既有 batch/imgsz/seed/增强。benchmark固定960，正式实验保留既定尺寸。

```yaml
# 8 vCPU：现有4作为基线，候选6，另测8
workers: 6
cache: false
pipeline_profile: false
```

```yaml
# 16 vCPU：先8，另测4/12/16；有争用时8或12优于16
workers: 8
cache: false
pipeline_profile: false
```

C获胜再在训练进程启动前设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`；工具C还显式torch.set_num_threads(1)，正式运行用线程环境后开启一次profile确认Torch实际值，勿假定环境一定生效。不推荐把pin_memory/persistent_workers/prefetch_factor直接写入8.4.90训练YAML。16 workers最多约64个预取batch，batch8/imgsz960仅图像uint8部分约1.4GB，尚未计入原图解码峰值、mask、mosaic buffer、IPC副本、验证池；需要实测RAM/共享内存。

## 9. batch=8 与 batch=12 的测试方案

A/B/C都固定batch8、imgsz960、原seed、同模型同数据同增强同其他超参；D仅在选出最佳workers/线程预算后测试batch12。下例假设B_workers_8最佳：

```bash
python scripts/benchmark_input_pipeline.py --config configs/train.yaml --workers 8 --batch12-workers 8 --only B_workers_8 D_batch12_workers_8 --epochs 5 --device 0 --execute --output runs/pipeline-batch12
python scripts/benchmark_input_pipeline.py --summarize runs/pipeline-batch12
```

若C_workers_8_threads1最佳，使用 `--only C_workers_8_threads1 D_batch12_workers_8 --threads 1`，保证batch8/12线程预算一致。比较images/s、epoch wall、fetch gap、GPU显存及OOM，不能只比iteration/s（batch12每步样本更多）。默认960结果不能保证1280不会OOM；1280须另行保持既定尺寸验证。batch改变可能影响积累/优化轨迹，是单独生产候选，不混入DataLoader收益。

兼容性补充：run-all现支持 `blade-defect experiment run-all --config configs/train.yaml --device 0 --workers 8 --batch 12`。显式batch覆盖所有选中实验的注册表值，manifest顶层batch、config.effective.batch及metrics.batch均记录12；启动日志显示effective batch/workers、imgsz、device。不传--batch时仍由注册表覆盖YAML，正式注册表默认值不变。`blade-defect train`的batch来自YAML（未配置则官方默认），workers为CLI --workers > YAML > 官方默认；它不使用实验注册表。workers日志是最终配置值，实际进程数仍受CPU/GPU及框架限制。

## 10. 是否仍建议升级16C A10

建议将16vCPU作为优先试租/验证候选，本次证据支持增加CPU解码并行度；A10 24GB是否合适取决于现有GPU与batch12实测，显存更大不能自动解决供数瓶颈。先在8C测4/6/8的CPU饱和与I/O等待，再在16C同口径测4/8/12/16。CPU已满且ESSD等待低、升16C后稳定吞吐提升才支持升级收益；若主要iowait，应先解决存储；若loader已经供给充足，再评估GPU算力。不承诺21h/34h缩短到某个时间。

## 官方参考

核查项目安装版本，在线main会变化：
- [8.4.90 build.py](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/data/build.py)
- [8.4.90 Trainer](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/models/yolo/detect/train.py)
- [8.4.90 BaseDataset](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/data/base.py)
- [8.4.90 utils与线程设置](https://github.com/ultralytics/ultralytics/blob/v8.4.90/ultralytics/utils/__init__.py)

本次没有修改/生成缩小版或二次压缩的正式数据集。
