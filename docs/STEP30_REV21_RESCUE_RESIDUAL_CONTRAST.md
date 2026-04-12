# Step30 Rev21 Rescue Residual Contrast / Suppression

Rev21 is a tiny rescue-scope training probe. It does not add a weak cue, run backend integration, modify the backend, or reopen the parked Step22-29 line.

## Question

Can one small residual-shaping objective make the rev19 rescue-scoped bundle branch safer by reducing positive residual boosts on unsafe low-hint pair-support false admissions while preserving enough missed-edge rescue?

## Setup

- Data:
  - train: `data/graph_event_step30_weak_obs_rev17_train.pkl`
  - val: `data/graph_event_step30_weak_obs_rev17_val.pkl`
  - test: `data/graph_event_step30_weak_obs_rev17_test.pkl`
- Initialization: `checkpoints/step30_encoder_recovery_rev19/best.pt`
- Frozen parameters: all non-rescue parameters frozen
- Trainable parameters:
  - `pair_evidence_rescue_head.0.weight`
  - `pair_evidence_rescue_head.0.bias`
  - `pair_evidence_rescue_head.2.weight`
  - `pair_evidence_rescue_head.2.bias`
- Rescue scope: `relation_hint < 0.50` and `pair_support_hint >= 0.55`
- Decode thresholds: `clean=0.50`, `noisy=0.55`

Checkpoint: `checkpoints/step30_encoder_recovery_rev21/best.pt`

Artifacts:

- `artifacts/step30_encoder_recovery_rev21/`
- `artifacts/step30_edge_diagnostics_rev21/`
- `artifacts/step30_rescue_scope_diagnostics_rev21/`

## Objective

Rev21 adds one small objective on `pair_evidence_rescue_residual` only:

- penalize positive residual on unsafe rescue negatives;
- keep a modest positive residual floor for safe missed positives;
- add a tiny batch-level margin between safe and unsafe residual means.

The loss is intentionally local to the rescue branch. The ordinary/global path remains frozen.

Training used:

- `rescue_residual_contrast_loss_weight = 0.75`
- `rescue_residual_margin = 0.25`
- `rescue_residual_relation_max = 0.50`
- `rescue_residual_support_min = 0.55`

Best epoch: 1.

## Recovery Comparison

All rows use the same rev17 test split and retained `clean=0.50`, `noisy=0.55` decode thresholds.

| model | overall edge F1 | clean edge F1 | noisy precision | noisy recall | noisy edge F1 |
|---|---:|---:|---:|---:|---:|
| rev6 | 0.7383 | 0.8215 | 0.6300 | 0.7072 | 0.6550 |
| rev19 | 0.7328 | 0.8181 | 0.6006 | 0.7297 | 0.6475 |
| rev21 | 0.7435 | 0.8300 | 0.6306 | 0.7115 | 0.6571 |

Rev21 technically clears the rev6 noisy edge F1 reference by about +0.0021, but the margin is small.

Event-family edge F1:

| model | edge_add | edge_delete | motif_type_flip | node_state_update |
|---|---:|---:|---:|---:|
| rev6 | 0.7398 | 0.7397 | 0.7387 | 0.7341 |
| rev19 | 0.7348 | 0.7333 | 0.7329 | 0.7285 |
| rev21 | 0.7452 | 0.7436 | 0.7457 | 0.7401 |

## Rescue-Scope Diagnostics

Noisy rescue-eligible scope has 14,874 candidate pairs, including 3,070 GT-positive missed true edges.

| model | precision | recall | F1 | admitted | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| rev6 | 0.3299 | 0.5062 | 0.3995 | 4,710 | 1,554 | 3,156 | 1,516 |
| rev19 | 0.3040 | 0.6436 | 0.4130 | 6,500 | 1,976 | 4,524 | 1,094 |
| rev21 | 0.3414 | 0.5407 | 0.4186 | 4,862 | 1,660 | 3,202 | 1,410 |

Rev21 improves rescue-scope precision over both rev6 and rev19, and improves rescue-scope F1 over rev19. It also sharply reduces false admissions relative to rev19. The cost is that much of rev19's rescue recall is lost.

Subtype behavior for rev21:

| subtype | count | admitted | admission rate | avg score | avg base score | avg residual |
|---|---:|---:|---:|---:|---:|---:|
| safe_missed_true_edge | 3,070 | 1,660 | 0.5407 | 0.5643 | 0.5421 | 0.2114 |
| low_hint_pair_support_false_admission | 2,479 | 1,307 | 0.5272 | 0.5567 | 0.5352 | 0.1958 |
| ambiguous_rescue_candidate | 9,325 | 1,895 | 0.2032 | 0.4373 | 0.4442 | -0.0566 |

Residual behavior:

| group | residual mean | positive residual frac | key interpretation |
|---|---:|---:|---|
| rev19 safe missed true | 0.5592 | 0.8762 | strong rescue, but broad boosting |
| rev19 unsafe false candidates | 0.3644 | 0.7558 | unsafe boosts too frequent |
| rev21 safe missed true | 0.2114 | 0.6590 | rescue retained, but much weaker |
| rev21 unsafe false candidates | -0.0036 | 0.4970 | unsafe boosts reduced substantially |
| rev21 admitted unsafe false | 0.2872 | 0.7380 | admitted false edges still often get positive residual |
| rev21 rejected safe true | 0.0372 | 0.5170 | many missed true edges now get too little residual |

## Targeted Diagnostics

| model | hint-missed recall | hint-missed avg score | hint-supported FP error | hint-missed vs hard-neg win |
|---|---:|---:|---:|---:|
| rev6 | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev19 | 0.3730 | 0.4761 | 0.4688 | 0.4143 |
| rev21 | 0.3003 | 0.4389 | 0.4688 | 0.3568 |

Rev21 is safer than rev19, but it gives back most of rev19's targeted missed-edge rescue improvement. It remains only slightly above rev6 on the named bottleneck.

## Threshold Sanity

Inside rescue scope for rev21:

| threshold | precision | recall | F1 | accepted |
|---:|---:|---:|---:|---:|
| 0.50 | 0.3061 | 0.6603 | 0.4182 | 6,623 |
| 0.55 | 0.3414 | 0.5407 | 0.4186 | 4,862 |
| 0.60 | 0.3775 | 0.4156 | 0.3957 | 3,380 |
| 0.65 | 0.4140 | 0.2997 | 0.3477 | 2,222 |
| 0.70 | 0.4584 | 0.1993 | 0.2779 | 1,335 |

The objective improves the operating point relative to rev19, but the rescue ranking is still shallow: stricter thresholds mostly trade recall away rather than revealing a strong high-precision rescue region.

## Diagnosis

Rev21 did reduce positive residual on unsafe rescue candidates:

- unsafe positive residual fraction: 0.7558 -> 0.4970
- rescue-scope FP count: 4,524 -> 3,202
- rescue-scope precision: 0.3040 -> 0.3414

But it did not preserve enough safe rescue recall:

- safe missed true admission: 0.6436 -> 0.5407
- hint-missed recall: 0.3730 -> 0.3003
- hint-missed vs hard-negative win rate: 0.4143 -> 0.3568

The result is cleaner than rev19 and narrowly above rev6 in noisy edge F1, but not by a decisive margin. It does not create a clearly stronger Step30 operating point.

## Backend Gate

No Step30c backend rerun was run.

Reason: rev21 technically clears rev6 noisy edge F1 by a very small margin, but the gain is not decisive and the targeted missed-edge rescue signal mostly regresses toward rev6. This is not strong enough to justify backend/interface work yet.

## Conclusion

Rev21 makes rescue-scoped bundle consumption cleaner, but not decisively better. The residual suppression objective fixes part of rev19's unsafe-boost problem, at the cost of suppressing too much safe rescue.

The next smallest justified move is not backend integration. If continuing this line, the next probe should keep the unsafe suppression but add a better safe-positive preservation term, selected on rescue-scope precision/recall rather than global edge F1 alone.
