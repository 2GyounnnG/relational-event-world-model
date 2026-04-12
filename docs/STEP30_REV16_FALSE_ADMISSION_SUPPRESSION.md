# Step30 rev16 False-Admission Suppression

## Question

Can a small false-admission suppression objective make the rev15 `signed_pair_witness` correction path safer, preserving enough missed-edge rescue while reducing unsafe false-positive score increases enough to beat the rev6 noisy edge F1 reference?

This pack keeps the rev13 benchmark fixed. It does not add another weak observation cue, does not run backend training, and does not move to adapter/interface work.

## Mechanism

rev16 keeps the rev14/15 bounded additive witness correction path and adds one direct penalty:

`false_admission_correction_loss`

The penalty applies only to rescue-like unsafe candidates:

- relation hint below `0.50`
- pair-support hint at least `0.55`
- GT edge is negative
- valid non-diagonal node pair

For those candidates, rev16 penalizes positive correction magnitude:

`relu(signed_pair_witness_correction_logits)^2`

This is deliberately different from rev15's signed supervision. rev15 asked the correction head to learn a broad positive/negative/zero target. rev16 directly says: do not increase unsafe false-admission candidates.

Training settings:

- initialization: `checkpoints/step30_encoder_recovery_rev6/best.pt`
- correction scale: `0.50`
- false-admission correction loss weight: `0.20`
- false-admission relation max: `0.50`
- false-admission support min: `0.55`
- decode: hard threshold only, no selective rescue
- thresholds: clean `0.50`, noisy `0.55`

## Recovery Results

| Row | Overall Edge F1 | Clean Edge F1 | Noisy Precision | Noisy Recall | Noisy Edge F1 |
|---|---:|---:|---:|---:|---:|
| rev6 reference | 0.7383 | 0.8215 | 0.6300 | 0.7072 | 0.6550 |
| rev15 | 0.7377 | 0.8207 | 0.5972 | 0.7499 | 0.6546 |
| rev16 | 0.7365 | 0.8196 | 0.5970 | 0.7471 | 0.6533 |
| rev13 trivial | 0.5496 | 0.6166 | 0.4923 | 0.4970 | 0.4826 |

Event-family edge F1 for rev16:

| Event family | Edge F1 |
|---|---:|
| edge_add | 0.7370 |
| edge_delete | 0.7383 |
| motif_type_flip | 0.7376 |
| node_state_update | 0.7322 |

## Targeted Rescue Diagnostics

Noisy split:

| Row | Hint-Missed True Recall | Hint-Missed Avg Score | Hint-Supported FP Error | HM vs Hard-Neg Win Rate |
|---|---:|---:|---:|---:|
| rev6 reference | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev15 | 0.3553 | 0.4674 | 0.5122 | 0.3806 |
| rev16 | 0.3602 | 0.4692 | 0.5028 | 0.3912 |
| rev13 trivial | 0.5662 | 0.5178 | 0.5021 | 0.5466 |

## Interpretation

rev16 moves the targeted rescue metrics in the intended direction relative to rev15:

- hint-missed true recall improves from `0.3553` to `0.3602`
- hint-missed average score improves from `0.4674` to `0.4692`
- hint-supported false-positive error improves from `0.5122` to `0.5028`
- hint-missed vs hard-negative win rate improves from `0.3806` to `0.3912`

However, the improvement is not enough to make the global operating point better:

- noisy precision is effectively unchanged: `0.5972` to `0.5970`
- noisy recall drops: `0.7499` to `0.7471`
- noisy edge F1 drops: `0.6546` to `0.6533`
- rev16 remains below rev6 noisy edge F1: `0.6533` vs `0.6550`

The direct false-admission penalty made the targeted axis slightly cleaner than rev15, but it did not convert the witness-correction mechanism into a better global recovery point.

## Sanity Judgment

The witness cue remains non-leaky:

- trivial noisy edge F1 remains low at `0.4826`
- trivial overall edge F1 remains low at `0.5496`
- encoder recovery remains far above trivial

Backend rerun was not run.

Reason:

`rev16 does not beat rev6 noisy edge F1 and does not provide a strong enough recovery-side justification for Step30c.`

## Conclusion

rev16 did not successfully make witness correction safe enough. It is slightly cleaner than rev15 on the targeted false-admission diagnostic, but it loses global noisy edge F1 and remains below the rev6 reference.

Recommended next move:

- Do not move to adapter/interface work yet.
- Do not add another decode micro-tweak.
- If continuing Step30, the next smallest justified move is to consolidate the signed-witness correction line and decide whether to park it, because rev14-rev16 repeatedly keep rescue recall but fail to recover enough precision/global F1.
