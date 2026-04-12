# Step31c Learned-vs-Late-Fusion Gap Probe

Step31c analyzes why simple late fusion is the retained Step31 backend-transfer
reference, then tests one narrow non-joint-training probe to stabilize learned
fusion.

No backend weights, adapters, interface layers, or new observation families were
added.

Artifacts:

- Script: `train/eval_step31c_fusion_gap_probe.py`
- Summary: `artifacts/step31c_fusion_gap_probe/summary.json`
- Recovery: `artifacts/step31c_fusion_gap_probe/recovery_summary.csv`
- Agreement buckets: `artifacts/step31c_fusion_gap_probe/agreement_bucket_summary.csv`
- Backend transfer: `artifacts/step31c_fusion_gap_probe/backend_summary.csv`

## Probe

The probe is `step31c_agreement_damped_encoder`.

It keeps the learned multi-view node path unchanged and modifies only edge
logits:

- compute cross-view pair disagreement from relation/support hint standard
  deviation
- if views disagree and learned fusion is more positive than simple late fusion,
  damp that learned-only positive excess toward the late-fusion logit
- leave agreement-region learned logits unchanged

This tests whether the learned encoder loses stability because it overreacts to
disagreement-rich pairs while late fusion acts as an implicit stabilizer.

## Recovery Comparison

| Row | Overall P | Overall R | Overall F1 | Clean F1 | Noisy P | Noisy R | Noisy F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| single_view_baseline | 0.7395 | 0.8211 | 0.7782 | 0.8648 | 0.6432 | 0.7616 | 0.6974 |
| simple_late_fusion_baseline | 0.9320 | 0.9078 | 0.9198 | 0.9732 | 0.8821 | 0.8501 | 0.8658 |
| step31_multi_view_encoder | 0.8608 | 0.9504 | 0.9034 | 0.9707 | 0.7777 | 0.9141 | 0.8404 |
| step31c_agreement_damped_encoder | 0.9105 | 0.9250 | 0.9177 | 0.9767 | 0.8449 | 0.8755 | 0.8599 |

Step31c narrows the learned-vs-late recovery gap:

- learned noisy F1: `0.8404`
- Step31c noisy F1: `0.8599`
- late-fusion noisy F1: `0.8658`

The remaining gap is small on recovery, but late fusion still retains the best
noisy F1.

## Agreement Bucket Diagnosis

Noisy split.

| Row | Bucket | GT+ rate | Pred+ rate | Precision | Recall | F1 | FP score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| late fusion | agreement | 0.2254 | 0.2198 | 0.9284 | 0.9051 | 0.9166 | 0.6234 |
| learned encoder | agreement | 0.2254 | 0.2375 | 0.8819 | 0.9289 | 0.9048 | 0.7391 |
| Step31c probe | agreement | 0.2254 | 0.2375 | 0.8819 | 0.9289 | 0.9048 | 0.7391 |
| late fusion | mid | 0.2487 | 0.2395 | 0.8874 | 0.8548 | 0.8708 | 0.6253 |
| learned encoder | mid | 0.2487 | 0.2866 | 0.7940 | 0.9150 | 0.8502 | 0.7485 |
| Step31c probe | mid | 0.2487 | 0.2668 | 0.8347 | 0.8956 | 0.8641 | 0.6921 |
| late fusion | disagreement | 0.2759 | 0.2635 | 0.8305 | 0.7933 | 0.8115 | 0.6309 |
| learned encoder | disagreement | 0.2759 | 0.3666 | 0.6771 | 0.8997 | 0.7727 | 0.7763 |
| Step31c probe | disagreement | 0.2759 | 0.2597 | 0.8354 | 0.7863 | 0.8101 | 0.6327 |

Simple late fusion wins because it is much more conservative in mid/disagreement
pairs. The learned encoder over-admits unstable edges, especially where view
signals disagree. Step31c directly fixes that axis: disagreement precision
improves from `0.6771` to `0.8354`, and disagreement FP rate drops from `0.1634`
to `0.0590`.

## Backend Transfer

### W012 Noisy

| Input | Full | Context | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step30_rev6_reference | 0.8099 | 0.8197 | 0.2354 | 0.1483 | 0.3088 | 0.1773 | 0.8673 |
| step31_simple_late_fusion | 0.9093 | 0.9219 | 0.1597 | 0.0521 | 0.2547 | 0.2763 | 0.8288 |
| step31_multi_view_encoder | 0.8931 | 0.9056 | 0.1424 | 0.0931 | 0.1847 | 0.2234 | 0.8466 |
| step31c_agreement_damped_encoder | 0.9039 | 0.9165 | 0.1486 | 0.0521 | 0.2321 | 0.2557 | 0.8310 |

### RFT1 Calibrated P2 Noisy

| Input | Full | Context | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| step30_rev6_reference | 0.7882 | 0.7962 | 0.3156 | 0.1861 | 0.4307 | 0.5851 | 0.5578 |
| step31_simple_late_fusion | 0.8740 | 0.8844 | 0.2584 | 0.1002 | 0.3985 | 0.6820 | 0.5378 |
| step31_multi_view_encoder | 0.8572 | 0.8670 | 0.2716 | 0.1349 | 0.3949 | 0.7036 | 0.5128 |
| step31c_agreement_damped_encoder | 0.8661 | 0.8760 | 0.2712 | 0.0923 | 0.4343 | 0.6840 | 0.5328 |

Step31c narrows the full/context gap versus late fusion while preserving some
of the learned encoder's proposal-sensitive behavior, but it does not dominate:

- `rft1_calibrated_p2` full/context improve over learned encoder but remain
  below late fusion.
- proposal recall and OOS miss remain better than late fusion only weakly or not
  at all depending on backend.
- delete quality regresses on noisy `rft1_calibrated_p2`, which prevents Step31c
  from becoming the retained backend-transfer point.

## RFT1 Event-Family Summary

Noisy split.

| Family | Input | Full | Changed | Add | Delete | PropR | OOS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| edge_add | rev6 | 0.7760 | 0.2058 | 0.1861 | 0.4038 | 0.3906 | 0.7265 |
| edge_add | late fusion | 0.8629 | 0.1246 | 0.1002 | 0.3654 | 0.3730 | 0.7837 |
| edge_add | learned encoder | 0.8494 | 0.1554 | 0.1349 | 0.3750 | 0.4072 | 0.7506 |
| edge_add | Step31c probe | 0.8560 | 0.1183 | 0.0923 | 0.4135 | 0.3730 | 0.7837 |
| edge_delete | rev6 | 0.7910 | 0.4124 | 0.1635 | 0.4307 | 0.6141 | 0.3956 |
| edge_delete | late fusion | 0.8749 | 0.3759 | 0.0673 | 0.3985 | 0.7274 | 0.3131 |
| edge_delete | learned encoder | 0.8558 | 0.3759 | 0.1250 | 0.3949 | 0.7517 | 0.2852 |
| edge_delete | Step31c probe | 0.8670 | 0.4102 | 0.0962 | 0.4343 | 0.7439 | 0.2925 |
| motif_type_flip | rev6 | 0.7909 | 0.3360 | 0.2656 | 0.4098 | 0.6664 | 0.5200 |
| motif_type_flip | late fusion | 0.8798 | 0.2880 | 0.1719 | 0.4098 | 0.7935 | 0.5440 |
| motif_type_flip | learned encoder | 0.8634 | 0.3040 | 0.2188 | 0.3934 | 0.8180 | 0.4880 |
| motif_type_flip | Step31c probe | 0.8707 | 0.2160 | 0.0625 | 0.3770 | 0.7979 | 0.5120 |
| node_state_update | rev6 | 0.7931 | 0.3509 | 0.1857 | 0.4653 | 0.6297 | 0.5263 |
| node_state_update | late fusion | 0.8794 | 0.3626 | 0.1429 | 0.5149 | 0.7539 | 0.5380 |
| node_state_update | learned encoder | 0.8615 | 0.3392 | 0.1000 | 0.5050 | 0.7690 | 0.5380 |
| node_state_update | Step31c probe | 0.8694 | 0.3626 | 0.1143 | 0.5347 | 0.7516 | 0.5497 |

## Diagnosis

The learned encoder's main weakness is not lack of multi-view signal. It is
admission instability in mid/disagreement buckets: it keeps too much learned-only
positive edge evidence when views conflict. Simple late fusion behaves like an
implicit stabilizer by averaging independently trained single-view logits.

The Step31c probe validates that diagnosis. A conservative disagreement-aware
damping rule substantially improves precision and narrows the recovery gap:

- noisy precision: `0.7777 -> 0.8449`
- noisy F1: `0.8404 -> 0.8599`
- disagreement precision: `0.6771 -> 0.8354`

But it does not beat simple late fusion on the retained backend-transfer point.
Simple late fusion remains the current Step31 reference. Learned fusion is closer
and more interpretable after Step31c, but adapter/interface work is still
premature.
