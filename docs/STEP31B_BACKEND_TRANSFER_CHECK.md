# Step31b Backend Transfer Check

Step31b checks whether the Step31 synthetic multi-view observation bridge transfers
to the frozen proposal/rewrite backend. It does not train backend parameters, add
adapters, or introduce a new observation family.

Artifacts:

- Script: `train/eval_step31_backend_transfer.py`
- Results: `artifacts/step31_backend_transfer_check/summary.json`
- Results CSV: `artifacts/step31_backend_transfer_check/summary.csv`

## Setup

Backends:

- `w012`
- `rft1_calibrated_p2`

Compared inputs:

- `gt_structured`
- `step30_rev6_reference`
- `step31_single_view_baseline`
- `step31_trivial_multi_view`
- `step31_simple_late_fusion`
- `step31_multi_view_encoder`

The Step31 recovery reference going into this check was:

| Row | Noisy edge F1 |
| --- | ---: |
| Step30 rev6 reference | 0.6657 |
| Step31 single-view baseline | 0.6974 |
| Step31 learned multi-view encoder | 0.8404 |
| Step31 simple late fusion | 0.8658 |

Note: `gt_structured` is collected from both Step30 and Step31 loaders, so its
overall count is doubled relative to non-GT rows. Both loaders use the same
target distribution, and deltas are computed against the corresponding backend
GT metric.

## W012 Noisy Overall

Each cell is `value (delta vs GT)`.

| Input | Full | Context | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gt_structured | 0.9652 | 0.9799 | 0.0737 | 0.0237 | 0.1212 | 0.3170 | 0.7989 |
| step30_rev6_reference | 0.8099 (-0.1553) | 0.8197 (-0.1603) | 0.2354 (+0.1617) | 0.1483 (+0.1246) | 0.3088 (+0.1876) | 0.1773 (-0.1397) | 0.8673 (+0.0685) |
| step31_single_view_baseline | 0.8220 (-0.1432) | 0.8315 (-0.1484) | 0.2469 (+0.1733) | 0.1893 (+0.1656) | 0.3022 (+0.1810) | 0.1666 (-0.1503) | 0.8802 (+0.0813) |
| step31_trivial_multi_view | 0.8574 (-0.1078) | 0.8665 (-0.1134) | 0.3025 (+0.2288) | 0.0442 (+0.0205) | 0.5394 (+0.4182) | 0.2948 (-0.0221) | 0.8003 (+0.0014) |
| step31_simple_late_fusion | 0.9093 (-0.0558) | 0.9219 (-0.0581) | 0.1597 (+0.0860) | 0.0521 (+0.0284) | 0.2547 (+0.1336) | 0.2763 (-0.0407) | 0.8288 (+0.0300) |
| step31_multi_view_encoder | 0.8931 (-0.0721) | 0.9056 (-0.0743) | 0.1424 (+0.0687) | 0.0931 (+0.0694) | 0.1847 (+0.0635) | 0.2234 (-0.0936) | 0.8466 (+0.0478) |

## RFT1 Calibrated P2 Noisy Overall

Each cell is `value (delta vs GT)`.

| Input | Full | Context | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gt_structured | 0.9140 | 0.9253 | 0.2300 | 0.0591 | 0.3942 | 0.7868 | 0.4772 |
| step30_rev6_reference | 0.7882 (-0.1258) | 0.7962 (-0.1291) | 0.3156 (+0.0856) | 0.1861 (+0.1270) | 0.4307 (+0.0365) | 0.5851 (-0.2017) | 0.5578 (+0.0806) |
| step31_single_view_baseline | 0.7995 (-0.1145) | 0.8071 (-0.1182) | 0.3370 (+0.1070) | 0.2232 (+0.1640) | 0.4460 (+0.0518) | 0.6164 (-0.1704) | 0.5364 (+0.0592) |
| step31_trivial_multi_view | 0.8359 (-0.0781) | 0.8440 (-0.0813) | 0.3564 (+0.1263) | 0.0591 (+0.0000) | 0.6241 (+0.2299) | 0.4772 (-0.3096) | 0.6812 (+0.2040) |
| step31_simple_late_fusion | 0.8740 (-0.0400) | 0.8844 (-0.0409) | 0.2584 (+0.0284) | 0.1002 (+0.0410) | 0.3985 (+0.0044) | 0.6820 (-0.1048) | 0.5378 (+0.0606) |
| step31_multi_view_encoder | 0.8572 (-0.0567) | 0.8670 (-0.0583) | 0.2716 (+0.0416) | 0.1349 (+0.0757) | 0.3949 (+0.0007) | 0.7036 (-0.0832) | 0.5128 (+0.0357) |

## Learned Encoder vs Simple Late Fusion

Simple late fusion is the strongest overall backend-transfer point on both
backends by full/context metrics. It also has the best recovery reference.

The learned multi-view encoder transfers differently:

- On `rft1_calibrated_p2`, it beats simple late fusion on proposal edge recall
  (`0.7036` vs `0.6820`) and out-of-scope miss (`0.5128` vs `0.5378`).
- It also has higher add and changed-edge values on `rft1_calibrated_p2`.
- Simple late fusion remains stronger on full/context and is the cleaner
  retained backend-transfer reference for this check.
- On `w012`, simple late fusion is stronger on full/context, proposal recall,
  out-of-scope miss, changed, and delete. The learned encoder only has higher
  add.

## RFT1 Event-Family Summary

### Edge Add

| Input | Full | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gt_structured | 0.9012 | 0.0970 | 0.0591 | 0.5000 | 0.4054 | 0.7926 |
| step30_rev6_reference | 0.7760 | 0.2058 | 0.1861 | 0.4038 | 0.3906 | 0.7265 |
| step31_single_view_baseline | 0.7903 | 0.2397 | 0.2232 | 0.4615 | 0.3980 | 0.7023 |
| step31_trivial_multi_view | 0.8185 | 0.1033 | 0.0591 | 0.5769 | 0.2567 | 0.8664 |
| step31_simple_late_fusion | 0.8629 | 0.1246 | 0.1002 | 0.3654 | 0.3730 | 0.7837 |
| step31_multi_view_encoder | 0.8494 | 0.1554 | 0.1349 | 0.3750 | 0.4072 | 0.7506 |

### Edge Delete

| Input | Full | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gt_structured | 0.9142 | 0.3591 | 0.0385 | 0.3942 | 0.8519 | 0.1772 |
| step30_rev6_reference | 0.7910 | 0.4124 | 0.1635 | 0.4307 | 0.6141 | 0.3956 |
| step31_single_view_baseline | 0.7987 | 0.4307 | 0.2596 | 0.4460 | 0.6507 | 0.3811 |
| step31_trivial_multi_view | 0.8442 | 0.5832 | 0.0385 | 0.6241 | 0.5192 | 0.5073 |
| step31_simple_late_fusion | 0.8749 | 0.3759 | 0.0673 | 0.3985 | 0.7274 | 0.3131 |
| step31_multi_view_encoder | 0.8558 | 0.3759 | 0.1250 | 0.3949 | 0.7517 | 0.2852 |

### Motif Type Flip

| Input | Full | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gt_structured | 0.9228 | 0.2320 | 0.0938 | 0.3770 | 0.9198 | 0.4880 |
| step30_rev6_reference | 0.7909 | 0.3360 | 0.2656 | 0.4098 | 0.6664 | 0.5200 |
| step31_single_view_baseline | 0.8049 | 0.3760 | 0.2969 | 0.4590 | 0.7117 | 0.5040 |
| step31_trivial_multi_view | 0.8400 | 0.3840 | 0.1250 | 0.6557 | 0.5676 | 0.6560 |
| step31_simple_late_fusion | 0.8798 | 0.2880 | 0.1719 | 0.4098 | 0.7935 | 0.5440 |
| step31_multi_view_encoder | 0.8634 | 0.3040 | 0.2188 | 0.3934 | 0.8180 | 0.4880 |

### Node State Update

| Input | Full | Changed | Add | Delete | PropR | OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gt_structured | 0.9174 | 0.2456 | 0.0571 | 0.3762 | 0.8723 | 0.4503 |
| step30_rev6_reference | 0.7931 | 0.3509 | 0.1857 | 0.4653 | 0.6297 | 0.5263 |
| step31_single_view_baseline | 0.8043 | 0.3860 | 0.2571 | 0.4752 | 0.6564 | 0.5439 |
| step31_trivial_multi_view | 0.8415 | 0.4503 | 0.0571 | 0.7228 | 0.5252 | 0.7193 |
| step31_simple_late_fusion | 0.8794 | 0.3626 | 0.1429 | 0.5149 | 0.7539 | 0.5380 |
| step31_multi_view_encoder | 0.8615 | 0.3392 | 0.1000 | 0.5050 | 0.7690 | 0.5380 |

## Diagnosis

Step31 multi-view recovery gains transfer downstream. The transfer is not just
trivial multi-view leakage: trivial multi-view is competitive on some structural
metrics but is unstable on proposal recall and out-of-scope miss, especially
under `rft1_calibrated_p2`.

The retained Step31b backend-transfer reference should be
`step31_simple_late_fusion` because it gives the best full/context behavior and
the cleanest overall transfer profile. The learned multi-view encoder remains an
important diagnostic because it transfers better on proposal-sensitive axes in
the noisy backend.

This justifies continuing the synthetic multi-view phase. It does not yet justify
adapter/interface work or backend joint training: the next work should first
understand why learned fusion improves proposal-sensitive behavior while simple
late fusion remains stronger overall.
