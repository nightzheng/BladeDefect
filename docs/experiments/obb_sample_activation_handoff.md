# v3 OBB 抽样激活交接

本交接入口用于在不执行 48,291 张全量 OBB 转换的情况下，验证正式 v3 txt 索引、polygon→OBB、indexed validator 和并排预览链路。

## 固定身份

- parent dataset：`blade-v3-grouped-202608`
- parent path：`D:/Program/BladeDefect/datasets/v3-index-rebuild/blade-v3-grouped-202608`
- parent membership：`02a0251aeca30a4e8b332967c2124848955886a2550f75933cc1e38146dfc37b`
- sample seed：42
- sample split：train/val/test 各 40 张
- sample membership：`f311a9587aa035dd8a222f51f485901d84595187cddfe04723d048b5279ab4b7`

## 执行顺序

```text
formal v3 train/val/test indexes
        -> build_obb_activation_sample.py
        -> sampled parent indexes (40/40/40)
        -> convert_seg_to_obb.py
        -> sampled indexed OBB
        -> check_obb_dataset.py
        -> visualize_obb_labels.py
```

抽样器只从每个正式 split 内选择样本，不产生 split 分配，也不修改 parent。选择评分优先覆盖 15 类、12/13/14、small/medium/large、倾斜细长、多实例、边缘和不规则 polygon。

## 数据入口

- sampled seg：`results/obb_v3_activation/sample_activation/parent_sample/data.yaml`
- sampled OBB：`results/obb_v3_activation/sample_activation/derived_obb/data.yaml`
- 验收结论：`results/obb_v3_activation/sample_activation/activation_acceptance.md`
- 抽样 smoke 配置：`configs/experiments/obb_yolo11s_960_v3_sample_activation_smoke.yaml`

正式转换结果为 120/120 张、316/316 实例成功，validator PASS，三 split identity 完全继承，120 张预览全部生成。test 仅参与转换、完整性校验和标签预览，不用于训练、预测、模型或阈值选择。

## Smoke 边界

当前可以进入“抽样数据上的真实 1 epoch smoke”数据 gate，但不能立即运行：官方 `yolo11s-obb.pt` 仍缺失且 `weight_provenance.json` 为 `verified: false`。取得并验证官方权重后，仍需负责人明确批准再运行；本交接没有启动训练。

本抽样 PASS 是历史激活阶段结论，不代表全量 OBB 数据已经生成，也不能替代正式全量转换验收。
旧“50 epochs 实验”计划已由 `v3_yolo11s_obb_960_e200` 取代；后者以 200 轮为预算上限，
实际在 e82 平台期封存（最佳 e81）。未来同口径从头训练建议 90 epochs。
