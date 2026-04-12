# Step30 rev19 Rescue-Scoped Bundle Consumption

## Question

Can rescue-scoped `pair_evidence_bundle` consumption preserve rev17's targeted rescue gains while avoiding the broader precision damage caused by feeding the bundle into global pair scoring?

This is recovery-first. It does not add a new cue, does not reopen Step22-Step29, does not run backend joint training, and does not move to adapter/interface work.

## Mechanism

rev19 keeps the rev17 benchmark and the same four-channel `pair_evidence_bundle`.

The only mechanism change is where the bundle is consumed:

- ordinary edge path uses safer rev6-style inputs: node pair features, relation hint, and pair-support hint
- `pair_evidence_bundle` is not fed into the ordinary global edge head
- a tiny rescue-scoped residual head consumes the bundle only for low-relation candidates
- residual scope: `relation_hint < 0.50`
- residual is bounded with `tanh`
- residual scale: `0.50`

Training:

- initialization: `checkpoints/step30_encoder_recovery_rev6/best.pt`
- non-rescue parameters frozen
- trainable parameters: only `pair_evidence_rescue_head`
- best epoch: `5`
- best validation selection score: `1.345537`
- decode thresholds: clean `0.50`, noisy `0.55`

This makes the ordinary path remain the rev6-style default and isolates bundle learning to the low-relation rescue scope.

## Recovery Results

| Row | Overall Edge F1 | Clean Edge F1 | Noisy Precision | Noisy Recall | Noisy Edge F1 |
|---|---:|---:|---:|---:|---:|
| rev6 reference | 0.7383 | 0.8215 | 0.6300 | 0.7072 | 0.6550 |
| rev17 global bundle | 0.7337 | 0.8193 | 0.5893 | 0.7453 | 0.6482 |
| rev19 rescue-scoped bundle | 0.7328 | 0.8181 | 0.6006 | 0.7297 | 0.6475 |
| rev17 trivial | 0.5011 | 0.5682 | 0.4542 | 0.4397 | 0.4340 |

Event-family edge F1 for rev19:

| Event family | Edge F1 |
|---|---:|
| edge_add | 0.7348 |
| edge_delete | 0.7333 |
| motif_type_flip | 0.7329 |
| node_state_update | 0.7285 |

## Targeted Rescue Diagnostics

Noisy split:

| Row | Hint-Missed True Recall | Hint-Missed Avg Score | Hint-Supported FP Error | HM vs Hard-Neg Win Rate |
|---|---:|---:|---:|---:|
| rev6 reference | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev17 global bundle | 0.4580 | 0.5109 | 0.4611 | 0.4932 |
| rev19 rescue-scoped bundle | 0.3730 | 0.4761 | 0.4688 | 0.4143 |
| rev17 trivial | 0.6128 | 0.5253 | 0.4751 | 0.5972 |

rev19 preserves some rescue signal relative to rev6, but gives back much of rev17's targeted gain:

- hint-missed recall stays above rev6: `0.3730` vs `0.2817`
- hint-missed recall drops below rev17: `0.3730` vs `0.4580`
- hint-supported FP error returns to rev6-like safety: `0.4688`

## Precision-Damage Diagnostics

### False-Positive Sources

Noisy split. Fractions are within each model's false positives.

| Row | Classic Hint FP | Ambiguous Mid-Hint FP | Low-Hint High-Bundle FP | Low-Hint Pair-Support Rescue FP | Other |
|---|---:|---:|---:|---:|---:|
| rev17 global bundle | 3085 / 26.1% | 4788 / 40.6% | 3366 / 28.5% | 541 / 4.6% | 27 / 0.2% |
| rev19 rescue-scoped bundle | 3430 / 30.8% | 4326 / 38.8% | 2648 / 23.8% | 719 / 6.5% | 14 / 0.1% |

rev19 reduces the broad contamination diagnosed in rev18:

- ambiguous mid-hint FP: `4788 -> 4326`
- low-hint high-bundle FP: `3366 -> 2648`

But it does not remove the precision problem enough to improve global F1, and rescue-eligible false positives rise in the low-hint pair-support subtype:

- low-hint pair-support rescue FP: `541 -> 719`

### Rescue vs Non-Rescue Admissions

Noisy split.

| Row | Bucket | Precision | Recall Contribution | FP Count | FP Fraction |
|---|---|---:|---:|---:|---:|
| rev17 | ordinary non-rescue | 0.6670 | 0.6505 | 7443 | 63.0% |
| rev17 | rescue eligible | 0.3316 | 0.0945 | 4364 | 37.0% |
| rev19 | ordinary non-rescue | 0.6896 | 0.6413 | 6613 | 59.4% |
| rev19 | rescue eligible | 0.3040 | 0.0862 | 4524 | 40.6% |

rev19 does what it was supposed to do for ordinary scoring:

- ordinary precision improves: `0.6670 -> 0.6896`
- ordinary FP drops: `7443 -> 6613`

But rescue-scope quality worsens:

- rescue precision drops: `0.3316 -> 0.3040`
- rescue FP rises: `4364 -> 4524`
- rescue recall contribution drops: `0.0945 -> 0.0862`

## Gate Check

| Gate | Result | Pass? |
|---|---|---|
| Noisy edge F1 beats rev6 `0.6550` | rev19 `0.6475` | no |
| Hint-missed recall stays above rev6 | rev19 `0.3730` vs rev6 `0.2817` | yes |
| Global false-positive damage reduced vs rev17 | ordinary FP and low-hint high-bundle FP reduced | partial |
| Strong recovery-side justification for backend rerun | noisy F1 remains below rev6 and rev17 | no |

Backend rerun was not run.

## Interpretation

Rescue-scoped bundle consumption partially solves the rev17 failure mode:

- it reduces ordinary non-rescue contamination
- it reduces ambiguous and low-hint high-bundle false positives
- it restores high-relation hint-supported FP safety to the rev6 level

But it does not solve the overall recovery problem:

- noisy precision improves over rev17 but remains below rev6
- noisy recall drops too much
- noisy F1 is below both rev6 and rev17
- the rescue-scoped branch itself is not precise enough

Chinese summary: rev19 证明“不要全局喂 bundle”这个诊断是对的，普通路径污染确实下降了；但低 relation rescue residual 本身还不够干净，救回的 recall 不足，误收也没压住，所以 recovery gate 仍然失败。

## Conclusion

rev19 did not become a retained Step30 recovery operating point.

It is useful diagnostically because it confirms that scoped consumption reduces broad contamination. However, it under-delivers on the rescue branch and does not beat rev6 on noisy edge F1.

Recommended next move:

- Do not run backend integration.
- Do not move to adapter/interface work.
- The next smallest justified move is a diagnostic comparison of rescue residual targets/scores inside the low-relation scope to determine why the scoped branch admits too many low-hint pair-support false positives while losing much of rev17's hint-missed recall.
