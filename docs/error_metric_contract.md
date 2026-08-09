# Error Metric Contract

## Three-Layer Error Hierarchy

| Layer | Category | Definition | Denominator |
|---|---|---|---|
| L1: Localization | FN (miss) | GT instance not matched to any prediction | Total GT instances |
| L1: Localization | FP (false alarm) | Prediction not matched to any GT | Total predictions |
| L2: Classification | coarse_group_error | Matched GT-pred pair with different 6-class groups | Total matched pairs |
| L2: Classification | within_group_confusion | Matched GT-pred pair with same 6-class group but different 15-class ID | Total matched pairs |
| L3: Mask Quality | mask_quality_error | Matched pair with mask IoU below threshold | Total matched pairs |

## Current Numbers (full_primary_yolo11s_seg_960, blade-v2)

- Total GT instances: 4,943
- Total predictions: 2,033
- Matched GT-pred pairs: 1,491
- FN (miss): 3,382 (68.4% of GT)
- FP (false alarm): 463 (22.8% of predictions)
- Coarse group error: 129 (8.7% of matched)
- Within-group confusion: 750 (50.3% of matched)
- Note: 129 and 750 come from different counting methods. 129 = matched instances where coarse group differs. 750 = fine-class pairs where same coarse group but different fine class. These sums may overlap.

## Principles
1. FN is a localization/recall failure, NOT a classification error
2. Coarse_group_error and within_group_confusion only count MATCHED pairs
3. Old split (split_leakage_known=true) metrics are not directly comparable to v3
