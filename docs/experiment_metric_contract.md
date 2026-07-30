# 实验指标口径约定（metric contract）

本文档固定实验交付物的指标字段与测量口径，供训练侧（杨静怡）与分析侧
（张劲焱）共同遵守。**同名但口径不同的旧结果不得被覆盖**：任何新结果
必须携带 `dataset_id + metric_branch + fps_method`，与旧结果不一致时
另存新文件或新字段，不回写历史 metrics.json 的数值。

## 1. 数据身份

- `dataset_id + dataset_manifest_sha256 + split` 是实验数据的唯一身份；
  目录名和本机绝对路径不是数据身份。
- `blade-v2-sampled-4987`：采样版阶段性实验（exp001—exp008）。
- `blade-v2`：全量冻结版（train 43,463 / val 4,828，49,471 实例），
  即 registry 中的 `blade-v2-full-frozen-48291`。
- `blade-v2-sampled-4987-6class`：采样版 remap 的 6 类数据集，仅用于
  6 类 smoke 与层级可行性实验。

## 2. 指标字段（metrics.json）

seg 模型同时输出 Mask 与 Box 两套指标，分字段记录，不得混用：

| 字段 | 口径 |
|---|---|
| `mask_precision` / `mask_recall` / `mask_mAP50` / `mask_mAP50-95` / `mask_f1` | Ultralytics `result.seg`（mask 匹配） |
| `box_precision` / `box_recall` / `box_mAP50` / `box_mAP50-95` / `box_f1` | Ultralytics `result.box`（bbox 匹配） |
| `precision` / `recall` / `mAP50` / `mAP50-95` | 兼容字段，**固定等于 Mask 口径**（历史 exp001—008 即此口径） |
| `fps` | 见第 3 节 |
| `fps_method` | FPS 测量方法标识，必填 |

历史 exp001—exp008 的 `precision/recall/mAP50/mAP50-95` 为 Mask 口径，
`fps` 口径为 `ultralytics_val_speed_inference_ms`，补齐时保持原值不动。

## 3. FPS 口径

- 统一方法标识：`ultralytics_val_speed_inference_ms`
- 定义：验证阶段 `result.speed["inference"]`（单图纯推理毫秒数，不含
  preprocess/postprocess/NMS 之外的管线开销），`fps = 1000 / inference_ms`。
- 如需其他口径（如端到端视频流 FPS），必须使用新的 `fps_method` 值并
  另存字段（如 `fps_e2e`），不得覆盖 `fps`。

## 4. 逐类指标（per_class_metrics.csv）

每行一个 `(metric_branch, class_id)`：

```text
metric_branch,class_id,class_name,precision,recall,ap50,ap50_95
mask,0,表面腐蚀--保护膜损伤,...
box,0,表面腐蚀--保护膜损伤,...
```

`metric_branch ∈ {mask, box}`；`class_name` 来自训练所用 data.yaml 的
`names`（15 类或 6 类），缺失时记 `unknown_<id>` 并视为数据问题上报。

## 5. 逐样本/逐实例预测（validation_predictions.json）

训练侧只导出原始字段，不在导出侧判定最终错误类型：

- 样本级：`image_path, split, image_width, image_height, true_classes,
  predicted_classes, ground_truth[], predictions[]`
- GT 实例：`class_id, class_name, polygon`（归一化坐标）
- 预测实例：`class_id, class_name, confidence, bbox(xyxy 像素),
  mask_polygon(像素坐标，可空)`

实例匹配、IoU、FN/FP 候选与最终 `error_type` 由分析侧计算并经人工复核
确认；自动候选数量不得直接当作最终实例错误统计。

## 6. run_manifest.json

每组实验必须有 run_manifest，字段：`experiment_id, dataset_id,
dataset_manifest_sha256, code_commit, hardware{gpu_name, torch_version,
cuda_version, ultralytics_version, python_version, platform}, config,
command, started_at, finished_at, status, resume_supported`。

## 7. 汇总与混排规则

- 15 类与 6 类结果分别汇总（`label_level` 区分），不放入同一排名。
- 采样版（sampled_phase）与全量正式结果分开标记，不直接横向排名。
- 15 类预测聚合到 6 类属于诊断分析，独立 6 类训练属于模型实验，
  两者不能互相替代。
