# Step31d Late-Fusion-Informed Learned Fusion Probe

Step31d tests one narrow late-fusion-informed objective for the learned
multi-view encoder. It keeps the Step31 observation family unchanged and does
not train or adapt the backend.

Artifacts:

- Training script: `train/train_step31d_late_fusion_distilled_encoder.py`
- Evaluator: `train/eval_step31c_fusion_gap_probe.py`
- Checkpoint: `checkpoints/step31d_late_fusion_distilled_encoder/best.pt`
- Summary: `artifacts/step31d_late_fusion_distilled_probe/summary.json`
- Recovery: `artifacts/step31d_late_fusion_distilled_probe/recovery_summary.csv`
- Backend transfer: `artifacts/step31d_late_fusion_distilled_probe/backend_summary.csv`

## Mechanism

The Step31d probe starts from the current learned multi-view encoder and freezes
everything except the edge head.

It uses simple late fusion as a stability teacher only for edge logits:

- keep GT edge BCE as the main recovery supervision
- in view-disagreement regions, penalize learned edge logits only when they are
  more positive than the late-fusion teacher
- do not distill node type or state
- do not train backend components

Training settings:

- epochs: `8`
- best epoch: `7`
- trainable parameters: `68,097`
- edge loss weight: `1.0`
- teacher positive-excess loss weight: `0.8`
- disagreement gate: relation/support std ramp from `0.08` to `0.15`

## Recovery Comparison

| Row | Overall P | Overall R | Overall F1 | Clean F1 | Noisy P | Noisy R | Noisy F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_view_baseline | 0.7395 | 0.8211 | 0.7782 | 0.8648 | 0.6432 | 0.7616 | 0.6974 |
| simple_late_fusion_baseline | 0.9320 | 0.9078 | 0.9198 | 0.9732 | 0.8821 | 0.8501 | 0.8658 |
| step31_multi_view_encoder | 0.8608 | 0.9504 | 0.9034 | 0.9707 | 0.7777 | 0.9141 | 0.8404 |
| step31c_agreement_damped_encoder | 0.9105 | 0.9250 | 0.9177 | 0.9767 | 0.8449 | 0.8755 | 0.8599 |
| step31d_late_fusion_distilled_encoder | 0.9592 | 0.8510 | 0.9019 | 0.9668 | 0.9235 | 0.7571 | 0.8321 |

Step31d internalizes stability, but too strongly. It improves noisy precision
over both learned fusion and late fusion, but loses too much recall.

## Disagreement Bucket

Noisy split.

| Row | Pred+ rate | Precision | Recall | F1 | FP score |
| --- | ---: | ---: | ---: | ---: | ---: |
| simple_late_fusion_baseline | 0.2635 | 0.8305 | 0.7933 | 0.8115 | 0.6309 |
| step31_multi_view_encoder | 0.3666 | 0.6771 | 0.8997 | 0.7727 | 0.7763 |
| step31c_agreement_damped_encoder | 0.2597 | 0.8354 | 0.7863 | 0.8101 | 0.6327 |
| step31d_late_fusion_distilled_encoder | 0.2274 | 0.8537 | 0.7037 | 0.7715 | 0.6793 |

The learned-vs-late diagnosis still holds: learned fusion over-admits
disagreement edges. Step31d fixes that over-admission, but becomes more
conservative than late fusion in the same bucket.

## Backend Transfer

### W012 Noisy

| Input | Full | Context | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step30 rev6 | 0.8099 | 0.8197 | 0.2354 | 0.1483 | 0.3088 | 0.1773 | 0.8673 |
| late fusion | 0.9093 | 0.9219 | 0.1597 | 0.0521 | 0.2547 | 0.2763 | 0.8288 |
| learned encoder | 0.8931 | 0.9056 | 0.1424 | 0.0931 | 0.1847 | 0.2234 | 0.8466 |
| Step31c probe | 0.9039 | 0.9165 | 0.1486 | 0.0521 | 0.2321 | 0.2557 | 0.8310 |
| Step31d probe | 0.8922 | 0.9033 | 0.2152 | 0.0221 | 0.3891 | 0.2969 | 0.8060 |

### RFT1 Calibrated P2 Noisy

| Input | Full | Context | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step30 rev6 | 0.7882 | 0.7962 | 0.3156 | 0.1861 | 0.4307 | 0.5851 | 0.5578 |
| late fusion | 0.8740 | 0.8844 | 0.2584 | 0.1002 | 0.3985 | 0.6820 | 0.5378 |
| learned encoder | 0.8572 | 0.8670 | 0.2716 | 0.1349 | 0.3949 | 0.7036 | 0.5128 |
| Step31c probe | 0.8661 | 0.8760 | 0.2712 | 0.0923 | 0.4343 | 0.6840 | 0.5328 |
| Step31d probe | 0.8580 | 0.8669 | 0.3317 | 0.0521 | 0.5825 | 0.6075 | 0.6056 |

Step31d does not preserve the learned encoder's proposal-sensitive advantage.
On the noisy default backend it loses proposal recall and increases OOS miss
relative to late fusion, Step31c, and the original learned encoder.

## RFT1 Event-Family Summary

Noisy split.

| Family | Input | Full | Changed | Add | Delete | PropR | OOS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| edge_add | late fusion | 0.8629 | 0.1246 | 0.1002 | 0.3654 | 0.3730 | 0.7837 |
| edge_add | learned encoder | 0.8494 | 0.1554 | 0.1349 | 0.3750 | 0.4072 | 0.7506 |
| edge_add | Step31c probe | 0.8560 | 0.1183 | 0.0923 | 0.4135 | 0.3730 | 0.7837 |
| edge_add | Step31d probe | 0.8415 | 0.0899 | 0.0521 | 0.5096 | 0.3213 | 0.8346 |
| edge_delete | late fusion | 0.8749 | 0.3759 | 0.0673 | 0.3985 | 0.7274 | 0.3131 |
| edge_delete | learned encoder | 0.8558 | 0.3759 | 0.1250 | 0.3949 | 0.7517 | 0.2852 |
| edge_delete | Step31c probe | 0.8670 | 0.4102 | 0.0962 | 0.4343 | 0.7439 | 0.2925 |
| edge_delete | Step31d probe | 0.8647 | 0.5474 | 0.0481 | 0.5825 | 0.6490 | 0.3835 |
| motif_type_flip | late fusion | 0.8798 | 0.2880 | 0.1719 | 0.4098 | 0.7935 | 0.5440 |
| motif_type_flip | learned encoder | 0.8634 | 0.3040 | 0.2188 | 0.3934 | 0.8180 | 0.4880 |
| motif_type_flip | Step31c probe | 0.8707 | 0.2160 | 0.0625 | 0.3770 | 0.7979 | 0.5120 |
| motif_type_flip | Step31d probe | 0.8641 | 0.3280 | 0.0781 | 0.5902 | 0.7184 | 0.6160 |
| node_state_update | late fusion | 0.8794 | 0.3626 | 0.1429 | 0.5149 | 0.7539 | 0.5380 |
| node_state_update | learned encoder | 0.8615 | 0.3392 | 0.1000 | 0.5050 | 0.7690 | 0.5380 |
| node_state_update | Step31c probe | 0.8694 | 0.3626 | 0.1143 | 0.5347 | 0.7516 | 0.5497 |
| node_state_update | Step31d probe | 0.8620 | 0.4094 | 0.0429 | 0.6634 | 0.6744 | 0.6433 |

## Diagnosis

Step31d answers the core question negatively for this specific objective. The
late-fusion-informed training objective does teach stability, but it does not
preserve enough of the learned encoder's proposal-sensitive strengths.

Compared with Step31c:

- better precision
- worse recall
- worse noisy F1
- worse proposal recall and OOS miss downstream

Compared with simple late fusion:

- higher noisy recovery precision
- lower noisy recovery F1
- weaker full/context transfer
- worse proposal-sensitive transfer under `rft1_calibrated_p2`

The retained Step31 reference remains `step31_simple_late_fusion`. Step31c
remains the better diagnostic probe for narrowing the learned-vs-late gap.
Step31d is useful evidence that teacher stability must be balanced against
proposal-sensitive recall, but it is not a retained operating point.
