# Step30 Rev22 Safe-Positive Preservation

Rev22 is a narrow rescue-scope follow-up to rev21. It does not add a weak observation cue, run backend integration, update proposal/rewrite backends, or reopen the parked Step22-29 line.

## Question

Can a small safe-positive preservation term keep rev21's unsafe suppression while restoring more safe missed-edge rescue signal?

## Setup

- Data:
  - train: `data/graph_event_step30_weak_obs_rev17_train.pkl`
  - val: `data/graph_event_step30_weak_obs_rev17_val.pkl`
  - test: `data/graph_event_step30_weak_obs_rev17_test.pkl`
- Initialization: `checkpoints/step30_encoder_recovery_rev21/best.pt`
- Frozen parameters: all non-rescue parameters frozen
- Trainable parameters:
  - `pair_evidence_rescue_head.0.weight`
  - `pair_evidence_rescue_head.0.bias`
  - `pair_evidence_rescue_head.2.weight`
  - `pair_evidence_rescue_head.2.bias`
- Rescue scope: `relation_hint < 0.50` and `pair_support_hint >= 0.55`
- Decode thresholds: `clean=0.50`, `noisy=0.55`

Checkpoint: `checkpoints/step30_encoder_recovery_rev22/best.pt`

Artifacts:

- `artifacts/step30_encoder_recovery_rev22/`
- `artifacts/step30_edge_diagnostics_rev22/`
- `artifacts/step30_rescue_scope_diagnostics_rev22/`

## Objective Change

Rev22 keeps the rev21 rescue residual contrast/suppression objective and adds one explicit safe-positive preservation term on `pair_evidence_rescue_residual`.

The added term applies only inside rescue scope and only to GT-positive safe missed true edges. It penalizes residual below a small floor:

- `safe_rescue_residual_preservation_loss_weight = 0.50`
- `safe_rescue_residual_floor = 0.45`
- `safe_rescue_residual_relation_max = 0.50`
- `safe_rescue_residual_support_min = 0.55`

The intent is to keep rev21's unsafe suppression while preventing the rescue residual from collapsing on safe positives.

Best epoch: 1.

## Recovery Comparison

All rows use the same rev17 test split and retained `clean=0.50`, `noisy=0.55` decode thresholds.

| model | overall edge F1 | clean edge F1 | noisy precision | noisy recall | noisy edge F1 |
|---|---:|---:|---:|---:|---:|
| rev6 | 0.7383 | 0.8215 | 0.6300 | 0.7072 | 0.6550 |
| rev19 | 0.7328 | 0.8181 | 0.6006 | 0.7297 | 0.6475 |
| rev21 | 0.7435 | 0.8300 | 0.6306 | 0.7115 | 0.6571 |
| rev22 | 0.7401 | 0.8266 | 0.6176 | 0.7205 | 0.6537 |

Rev22 restores some recall relative to rev21, but gives up precision and falls below rev6 on noisy edge F1.

Event-family edge F1:

| model | edge_add | edge_delete | motif_type_flip | node_state_update |
|---|---:|---:|---:|---:|
| rev6 | 0.7398 | 0.7397 | 0.7387 | 0.7341 |
| rev19 | 0.7348 | 0.7333 | 0.7329 | 0.7285 |
| rev21 | 0.7452 | 0.7436 | 0.7457 | 0.7401 |
| rev22 | 0.7414 | 0.7407 | 0.7408 | 0.7365 |

## Rescue-Scope Diagnostics

Noisy rescue-eligible scope has 14,874 candidate pairs, including 3,070 GT-positive missed true edges.

| model | precision | recall | F1 | admitted | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| rev6 | 0.3299 | 0.5062 | 0.3995 | 4,710 | 1,554 | 3,156 | 1,516 |
| rev19 | 0.3040 | 0.6436 | 0.4130 | 6,500 | 1,976 | 4,524 | 1,094 |
| rev21 | 0.3414 | 0.5407 | 0.4186 | 4,862 | 1,660 | 3,202 | 1,410 |
| rev22 | 0.3247 | 0.5971 | 0.4207 | 5,645 | 1,833 | 3,812 | 1,237 |

Rev22 preserves more safe rescue than rev21 and slightly improves rescue-scope F1, but it gives back too much unsafe suppression.

Subtype behavior for rev22:

| subtype | count | admitted | admission rate | avg score | avg base score | avg residual |
|---|---:|---:|---:|---:|---:|---:|
| safe_missed_true_edge | 3,070 | 1,833 | 0.5971 | 0.5826 | 0.5421 | 0.3729 |
| low_hint_pair_support_false_admission | 2,479 | 1,464 | 0.5906 | 0.5752 | 0.5352 | 0.3551 |
| ambiguous_rescue_candidate | 9,325 | 2,348 | 0.2518 | 0.4575 | 0.4442 | 0.1205 |

The core failure remains visible: safe positives and low-hint pair-support false admissions still receive very similar admission rates.

## Residual Safety

| group | rev21 residual mean | rev22 residual mean | rev21 positive frac | rev22 positive frac |
|---|---:|---:|---:|---:|
| safe missed true | 0.2114 | 0.3729 | 0.6590 | 0.7876 |
| unsafe false candidates | -0.0036 | 0.1697 | 0.4970 | 0.6311 |
| admitted unsafe false | 0.2872 | 0.3988 | 0.7380 | 0.8221 |
| rejected safe true | 0.0372 | 0.2148 | 0.5170 | 0.6564 |

The preservation term works on safe positives, but it also raises unsafe residuals. The unsafe-positive residual fraction worsens from 0.4970 to 0.6311.

## Targeted Diagnostics

| model | hint-missed recall | hint-missed avg score | hint-supported FP error | hint-missed vs hard-neg win |
|---|---:|---:|---:|---:|
| rev6 | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev19 | 0.3730 | 0.4761 | 0.4688 | 0.4143 |
| rev21 | 0.3003 | 0.4389 | 0.4688 | 0.3568 |
| rev22 | 0.3361 | 0.4561 | 0.4688 | 0.3831 |

Rev22 recovers part of the rev19 targeted rescue signal, but not enough to create a clean operating point.

## Threshold Sanity

Inside rescue scope for rev22:

| threshold | precision | recall | F1 | accepted |
|---:|---:|---:|---:|---:|
| 0.50 | 0.2942 | 0.7121 | 0.4163 | 7,431 |
| 0.55 | 0.3247 | 0.5971 | 0.4207 | 5,645 |
| 0.60 | 0.3643 | 0.4704 | 0.4106 | 3,964 |
| 0.65 | 0.4010 | 0.3430 | 0.3697 | 2,626 |
| 0.70 | 0.4406 | 0.2296 | 0.3019 | 1,600 |

This confirms the issue is not a simple threshold-placement problem. The best visible rescue-scope F1 remains shallow, and stricter thresholds trade away too much rescue recall.

## Backend Gate

No Step30c backend rerun was run.

Reason: rev22 does not beat rev6 on noisy edge F1 and does not beat rev21 as a global recovery point. It improves safe rescue preservation, but it does so by weakening unsafe suppression.

## Conclusion

Rev22 answers the core question: safe-positive preservation can restore some safe rescue signal, but with the current residual head and signals it also reopens unsafe false admissions. It does not produce a genuinely cleaner rescue-scope operating point.

The exact remaining tradeoff:

- rev21 is safer but too blunt.
- rev22 preserves more safe positives but gives back too many unsafe positives.
- Neither is strong enough to justify backend integration or adapter/interface work.

Recommended next move: stop this residual-loss micro-line for now and consolidate rev19-rev22. If Step30 continues, the next mechanism should not be another scalar residual loss; it should revisit the rescue representation or candidate labeling so safe positives and low-hint support-backed false admissions are separable before applying residual shaping.
