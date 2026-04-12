# Step30 rev14 Signed Pair Witness Consumption

## Question

Can a small refinement in how the encoder consumes `signed_pair_witness` preserve rev13's targeted rescue gains while recovering enough noisy precision and global edge F1 to beat the rev6 recovery reference?

This pack keeps the rev13 benchmark fixed. It does not add another weak observation cue, does not continue the rev7-rev12 rescue-decode micro-line, and does not run backend adapter/interface work.

## Mechanism

rev13 fed `signed_pair_witness` directly into the pairwise edge head. rev14 instead tests a dedicated witness correction path:

- The main edge head keeps the rev6 feature shape.
- The model initializes from `checkpoints/step30_encoder_recovery_rev6/best.pt`.
- `signed_pair_witness` is not concatenated into the main edge head.
- A tiny correction head consumes:
  - signed witness
  - positive witness magnitude
  - negative witness magnitude
  - absolute witness magnitude
  - relation hint
  - pair-support hint
  - pair-support minus relation hint
- The correction is bounded with `tanh` and added to the edge logit.
- The correction head is zero-initialized, so the starting point is the rev6 scorer.

Two narrow correction scales were checked:

- `rev14_s050`: correction scale `0.50`
- `rev14_s025`: correction scale `0.25`

This is not a broad sweep; it is a single consumption refinement with one smaller safety check.

## Training

Both variants use the rev6-style recovery recipe:

- relation-logit residual
- pair-support hints
- missed-edge recovery loss
- no selective-rescue decode
- no backend training

Retained rev14 variant:

`rev14_s050`

Reason:

- It was the better rev14 test point by noisy edge F1.
- It improved over rev13 but did not clearly beat rev6.

## Recovery Results

Hard decode:

- clean threshold: `0.50`
- noisy threshold: `0.55`
- selective rescue: off

| Row | Overall Edge F1 | Clean Edge F1 | Noisy Precision | Noisy Recall | Noisy Edge F1 |
|---|---:|---:|---:|---:|---:|
| rev6 reference | 0.7383 | 0.8215 | 0.6300 | 0.7072 | 0.6550 |
| rev13 encoder | 0.7349 | 0.8218 | 0.6026 | 0.7257 | 0.6481 |
| rev14_s050 | 0.7364 | 0.8201 | 0.5959 | 0.7474 | 0.6528 |
| rev14_s025 | 0.7351 | 0.8182 | 0.5919 | 0.7517 | 0.6520 |
| rev13 trivial | 0.5496 | 0.6166 | 0.4923 | 0.4970 | 0.4826 |

Event-family edge F1 for retained rev14_s050:

| Event family | Edge F1 |
|---|---:|
| edge_add | 0.7372 |
| edge_delete | 0.7381 |
| motif_type_flip | 0.7367 |
| node_state_update | 0.7328 |

## Targeted Rescue Diagnostics

Noisy split:

| Row | Hint-Missed True Recall | Hint-Missed Avg Score | Hint-Supported FP Error | HM vs Hard-Neg Win Rate |
|---|---:|---:|---:|---:|
| rev6 reference | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev13 encoder | 0.3925 | 0.4790 | 0.4529 | 0.4535 |
| rev14_s050 | 0.3644 | 0.4710 | 0.5018 | 0.3941 |
| rev13 trivial | 0.5662 | 0.5178 | 0.5021 | 0.5466 |

## Interpretation

rev14 improves over rev13 on global noisy edge F1:

- rev13 noisy edge F1: `0.6481`
- rev14_s050 noisy edge F1: `0.6528`

But it does not beat rev6:

- rev6 noisy edge F1: `0.6550`

The correction path preserves some targeted rescue gain relative to rev6:

- hint-missed recall improves from `0.2817` to `0.3644`
- hint-missed average score improves from `0.4366` to `0.4710`
- hint-missed vs hard-negative win rate improves from `0.3476` to `0.3941`

However, it does not preserve the cleaner rev13 targeted signal:

- hint-missed recall drops from `0.3925` to `0.3644`
- hint-supported false-positive error worsens from `0.4529` to `0.5018`

The retained rev14 point is therefore a partial recovery of rev13's global F1 problem, but not a successful conversion of the new cue into a better operating point than rev6.

## Sanity Judgment

The cue remains non-leaky:

- trivial noisy edge F1 remains low at `0.4826`
- trivial overall edge F1 remains low at `0.5496`
- encoder recovery remains far above trivial

Step30 is not ready for backend rerun:

- rev14 does not clearly beat rev6 noisy edge F1.
- rev14 does not produce a cleaner precision/recall tradeoff.
- rev14 improves recall but still gives back too much precision.

## Backend Rerun Decision

Focused Step30c backend integration was not run.

Reason:

`retained rev14 improves over rev13 but does not clearly beat rev6 noisy edge F1.`

## Conclusion

rev14 shows that a separate bounded witness correction path is a better consumption pattern than the direct rev13 edge-head fusion, but it still falls short of the rev6 reference. The new cue is real, and the encoder can use it, but current consumption remains too recall-heavy and not precise enough.

Recommended next action:

- Do not move to adapter/interface work.
- Do not add another new weak cue immediately.
- If continuing Step30, the next smallest move should regularize or supervise the witness correction path so positive witness rescue does not also raise unsafe false admissions.
