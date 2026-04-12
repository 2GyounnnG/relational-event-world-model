# Step30 rev15 Witness Correction Supervision

## Question

Can a small signed supervision term make the rev14 `signed_pair_witness` correction path safer, preserving rescue gains while reducing false-admission damage enough to beat the rev6 noisy edge F1 reference?

This pack keeps the rev13 benchmark fixed. It does not add another weak observation cue, does not run backend training, and does not move to adapter/interface work.

## Mechanism

rev15 keeps the rev14 bounded additive witness correction path and adds one small auxiliary supervision term:

`signed_witness_correction_loss`

The loss is attached directly to `signed_pair_witness_correction_logits`.

Target design:

- low-relation true edges with active witness: positive correction
- low-relation, pair-support-backed false admissions with active witness: negative correction
- ambiguous low-witness cases: small correction, with reduced weight

Training settings:

- initialization: `checkpoints/step30_encoder_recovery_rev6/best.pt`
- correction scale: `0.50`
- signed correction loss weight: `0.10`
- witness active threshold: `0.25`
- ambiguous weight: `0.25`
- decode: hard threshold only, no selective rescue
- thresholds: clean `0.50`, noisy `0.55`

## Recovery Results

| Row | Overall Edge F1 | Clean Edge F1 | Noisy Precision | Noisy Recall | Noisy Edge F1 |
|---|---:|---:|---:|---:|---:|
| rev6 reference | 0.7383 | 0.8215 | 0.6300 | 0.7072 | 0.6550 |
| rev14 | 0.7364 | 0.8201 | 0.5959 | 0.7474 | 0.6528 |
| rev15 | 0.7377 | 0.8207 | 0.5972 | 0.7499 | 0.6546 |
| rev13 trivial | 0.5496 | 0.6166 | 0.4923 | 0.4970 | 0.4826 |

Event-family edge F1 for rev15:

| Event family | Edge F1 |
|---|---:|
| edge_add | 0.7388 |
| edge_delete | 0.7392 |
| motif_type_flip | 0.7376 |
| node_state_update | 0.7339 |

## Targeted Rescue Diagnostics

Noisy split:

| Row | Hint-Missed True Recall | Hint-Missed Avg Score | Hint-Supported FP Error | HM vs Hard-Neg Win Rate |
|---|---:|---:|---:|---:|
| rev6 reference | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev14 | 0.3644 | 0.4710 | 0.5018 | 0.3941 |
| rev15 | 0.3553 | 0.4657 | 0.5122 | 0.3806 |
| rev13 trivial | 0.5662 | 0.5178 | 0.5021 | 0.5466 |

## Interpretation

rev15 improves global recovery over rev14:

- rev14 noisy edge F1: `0.6528`
- rev15 noisy edge F1: `0.6546`

It nearly ties rev6:

- rev6 noisy edge F1: `0.6550`
- rev15 noisy edge F1: `0.6546`

But it does not beat rev6, and it does not solve the intended safety problem:

- noisy precision remains far below rev6: `0.5972` vs `0.6300`
- hint-supported false-positive error worsens: `0.5122` vs rev6 `0.4688`
- hint-missed recall remains above rev6, but lower than rev14

The signed supervision term made the correction path slightly better for global F1, but it did not make the witness correction safer. The operating point is still recall-heavy.

## Sanity Judgment

The witness cue remains non-leaky:

- trivial noisy edge F1 remains `0.4826`
- trivial overall edge F1 remains `0.5496`
- encoder recovery remains far above trivial

Backend rerun was not run.

Reason:

`rev15 nearly ties but does not clearly beat rev6 noisy edge F1.`

## Conclusion

rev15 did not successfully stabilize the witness correction path. It is the closest witness-cue operating point so far, but it still fails the core criterion: beating rev6 on noisy edge F1 while preserving a cleaner false-admission profile.

Recommended next move:

- Do not move to adapter/interface work yet.
- Do not add another weak cue immediately.
- The next smallest justified Step30 move is a diagnostic consolidation of the witness line, or a single sharper correction-safety objective that directly penalizes false-positive score increases, because the current signed target still permits too many unsafe admissions.
