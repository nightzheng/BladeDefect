# 失败案例索引

失败案例使用 `results/failure_cases/cases.csv` 建立轻量索引，不批量复制原图。当前仓库没有
完整预测结果，因此只提供可验证的结构与导出流程，不伪造案例。

## 字段

`image_path, split, true_class, predicted_class, confidence, iou, error_type, experiment_id`

`confidence` 和 `iou` 可留空；填写时必须位于 `[0, 1]`。`error_type` 可暂时留空，由后续人工
复核填写。支持的类别为：

- `small_object_miss`、`low_confidence_miss`、`class_confusion`
- `background_false_positive`、`blade_edge_false_positive`
- `overexposure`、`shadow`、`blur`
- `mask_boundary_error`、`multiple_defects`、`possible_label_error`

## 代表图与类别映射

培训材料中的 16 张代表图已经整理到
[`defect_reference_gallery.md`](defect_reference_gallery.md)，机器可读清单位于
[`assets/failure_cases/reference_catalog.csv`](assets/failure_cases/reference_catalog.csv)。

清单明确区分当前 15 类体系中的直接映射、相关参考和体系外补充类别。代表图只用于人工
复核，不作为模型失败案例，也不参与指标计算。

## 导出流程

先由验证/推理流程生成包含上述字段的临时 CSV，再运行：

```powershell
python scripts/export_failure_cases.py `
  --source runs/evaluation/failure_candidates.csv `
  --output results/failure_cases/cases.csv
```

脚本会校验字段、数值范围和错误类型。典型漏检、误检和 mask 边界问题应由人工结合原图、
标注和预测图复核。可参照代表图库区分掉漆/胶衣脱落、油污/腐蚀、表层裂纹/结构开裂等
高风险混淆组合。后续有真实预测数据时，可按 `error_type` 筛选索引并制作真实案例图。
