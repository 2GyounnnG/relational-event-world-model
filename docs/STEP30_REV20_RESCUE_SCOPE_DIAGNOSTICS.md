# Step30 Rev20 Rescue-Scope Diagnostics

Rev20 is a diagnostic-only pass on the Step30 `pair_evidence_bundle` line.

It does not train a model, add a cue, run backend integration, or reopen the parked Step22-29 noisy interaction line.

## Question

Inside the low-relation rescue scope, what separates safe missed true edges from low-hint pair-support false admissions, and why does the rev19 rescue-scoped residual still admit too many false positives?

## Setup

- Data: `data/graph_event_step30_weak_obs_rev17_test.pkl`
- Split analyzed: noisy weak observation only
- Rescue scope: `relation_hint < 0.50` and `pair_support_hint >= 0.55`
- Models compared:
  - rev6 reference: `checkpoints/step30_encoder_recovery_rev6/best.pt`
  - rev17 global bundle: `checkpoints/step30_encoder_recovery_rev17/best.pt`
  - rev19 rescue-scoped bundle: `checkpoints/step30_encoder_recovery_rev19/best.pt`
- Decode thresholds: `clean=0.50`, `noisy=0.55`
- Artifacts: `artifacts/step30_rescue_scope_diagnostics_rev20/`

The noisy rescue scope contains 14,874 candidate pairs, with 3,070 GT-positive safe missed true edges.

## Rescue-Scope Summary

| model | precision | recall | F1 | admitted | TP | FP | FN | avg score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rev6 | 0.3299 | 0.5062 | 0.3995 | 4,710 | 1,554 | 3,156 | 1,516 | 0.4796 |
| rev17 | 0.3316 | 0.7052 | 0.4511 | 6,529 | 2,165 | 4,364 | 905 | 0.5136 |
| rev19 | 0.3040 | 0.6436 | 0.4130 | 6,500 | 1,976 | 4,524 | 1,094 | 0.5250 |

Rev19 reduces broad non-rescue contamination relative to rev17, but inside the rescue scope it is worse than rev17: lower precision, lower recall, and lower F1.

## Subtype Decomposition

For rev19:

| rescue subtype | count | admitted | admission rate | avg final score | avg base score | avg residual |
|---|---:|---:|---:|---:|---:|---:|
| safe_missed_true_edge | 3,070 | 1,976 | 0.6436 | 0.6033 | 0.5421 | 0.5592 |
| low_hint_pair_support_false_admission | 2,479 | 1,614 | 0.6511 | 0.5959 | 0.5352 | 0.5360 |
| ambiguous_rescue_candidate | 9,325 | 2,910 | 0.3121 | 0.4804 | 0.4442 | 0.3188 |

The most damaging signal is direct: the rev19 branch admits low-hint pair-support false admissions at almost the same rate as safe missed true edges.

## Residual Behavior

| group | count | residual mean | q25 | q50 | q75 | positive residual frac |
|---|---:|---:|---:|---:|---:|---:|
| safe_missed_true_edges | 3,070 | 0.5592 | 0.3608 | 0.7105 | 0.8876 | 0.8762 |
| unsafe_false_admissions_all | 11,804 | 0.3644 | 0.0107 | 0.4949 | 0.7922 | 0.7558 |
| rev19_admitted_safe_edges | 1,976 | 0.6471 | 0.4932 | 0.7680 | 0.9033 | 0.9347 |
| rev19_admitted_unsafe_false_edges | 4,524 | 0.5632 | 0.3686 | 0.6871 | 0.8557 | 0.9043 |
| rev19_rejected_safe_edges | 1,094 | 0.4004 | 0.0476 | 0.5545 | 0.8372 | 0.7706 |

The residual is not simply too weak. It is under-separating and often mis-safe: 75.6% of all unsafe rescue candidates receive positive residual, and 90.4% of admitted unsafe candidates receive positive residual. Meanwhile, 13.7% of safe missed true edges have residual <= 0.05, rising to 25.1% among safe edges rev19 rejects.

## Feature Separation

| group | count | relation | support | bundle positive | bundle warning | bundle margin | base score | final score | residual |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| safe_missed_true_edges | 3,070 | 0.3949 | 0.6588 | 0.5499 | 0.4580 | 0.0919 | 0.5421 | 0.6033 | 0.5592 |
| unsafe_false_candidates | 11,804 | 0.3603 | 0.6287 | 0.4688 | 0.5446 | -0.0758 | 0.4633 | 0.5046 | 0.3644 |
| rev19_admitted_safe_edges | 1,976 | 0.4265 | 0.6857 | 0.5828 | 0.4224 | 0.1604 | 0.6210 | 0.6903 | 0.6471 |
| rev19_admitted_unsafe_false_edges | 4,524 | 0.4069 | 0.6708 | 0.5370 | 0.4668 | 0.0702 | 0.5880 | 0.6521 | 0.5632 |
| rev19_rejected_safe_edges | 1,094 | 0.3380 | 0.6103 | 0.4905 | 0.5223 | -0.0318 | 0.3996 | 0.4461 | 0.4004 |

There is some separable structure: safe candidates have higher bundle positive support, lower warning, higher positive-minus-warning margin, higher support, and higher base score. But the admitted unsafe group is close to the admitted safe group on every one of these signals. The current residual is therefore not extracting a clean enough safety boundary.

Endpoint-local summaries were not decisive: hint/common-neighbor means are nearly identical between safe and unsafe groups.

## Sensitivity

Score-threshold sensitivity inside rescue scope:

| threshold | precision | recall | F1 | accepted | TP | FP |
|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.2788 | 0.7596 | 0.4079 | 8,365 | 2,332 | 6,033 |
| 0.55 | 0.3040 | 0.6436 | 0.4130 | 6,500 | 1,976 | 4,524 |
| 0.60 | 0.3441 | 0.5293 | 0.4171 | 4,722 | 1,625 | 3,097 |
| 0.65 | 0.3889 | 0.4049 | 0.3967 | 3,196 | 1,243 | 1,953 |
| 0.70 | 0.4165 | 0.2691 | 0.3269 | 1,983 | 826 | 1,157 |

Raising the threshold modestly improves rescue-scope F1, but only from 0.4130 to 0.4171 and at a large recall cost. This is not enough to overturn the rev19 failure.

Top-k by final score shows the same issue: the very top rescue candidates are cleaner, but precision decays quickly. At the current accepted count of 6,500, precision is 0.3040. Ranking by residual alone or bundle margin is worse than ranking by final score.

## Event-Family Cross-Check

For rev19, rescue-scope behavior is similar across families:

| event family | precision | recall | F1 | admitted | TP | FP | avg residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge_add | 0.3009 | 0.6223 | 0.4056 | 2,114 | 636 | 1,478 | 0.4070 |
| edge_delete | 0.3025 | 0.6537 | 0.4136 | 2,215 | 670 | 1,545 | 0.4038 |
| motif_type_flip | 0.3034 | 0.6386 | 0.4113 | 1,747 | 530 | 1,217 | 0.4043 |
| node_state_update | 0.3006 | 0.6396 | 0.4090 | 1,913 | 575 | 1,338 | 0.3964 |

The remaining failure is not concentrated in a single event family. It is a shared rescue-scope precision/ranking issue.

## Diagnosis

Rev20 answers the core question:

1. Safe missed true edges and unsafe rescue false candidates are weakly distinguishable, mainly by bundle positive support, warning, positive-minus-warning margin, support, and base score.
2. Current rev19 residual does not separate them sharply enough.
3. The residual often increases unsafe false-admission candidates, so the failure is not only calibration or budget placement.
4. Threshold/budget sensitivity shows ranking quality is the main bottleneck: stricter admission improves precision but sacrifices too much recall and only slightly improves rescue-scope F1.
5. The problem is shared across event families, so a family-specific fix is not the clean next move.

## Recommendation

Run exactly one tiny rescue-scope training probe: keep the rev19 ordinary path frozen and add a direct rescue-residual contrast/suppression objective inside the rescue scope, contrasting safe missed true edges against low-hint pair-support false admissions and explicitly penalizing positive residual on unsafe rescue candidates.

Do not run backend integration yet.
