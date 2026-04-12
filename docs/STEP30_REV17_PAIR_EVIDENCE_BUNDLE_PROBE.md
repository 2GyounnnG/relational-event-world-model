# Step30 rev17 Pair Evidence Bundle Probe

## Question

Can a small multi-source weak pair evidence packet provide more rescue-safety information than a single scalar `signed_pair_witness`, while staying synthetic, structured, noisy, weak, overlapping, and non-leaky?

This is recovery-first only. It does not reopen Step22-Step29, does not continue rev13-rev16 signed-witness correction/supervision tweaks, does not add backend joint training, and does not move to adapter/interface work.

## Mechanism

rev17 introduces one new weak observation field:

`pair_evidence_bundle`

It is a pair-level packet with four channels:

1. `positive_support`: weak positive evidence for connectivity, with a small boost for low-relation true edges.
2. `false_admission_warning`: weak warning evidence for low-relation, pair-support-backed GT-negative candidates.
3. `corroboration`: weak evidence that the low-relation rescue pattern is internally safer or less safe.
4. `endpoint_compatibility`: noisy endpoint type/state compatibility evidence.

All four channels are corrupted with dropout, flips, jitter, and quantization. Values are intentionally overlapping and low-amplitude. No single channel is intended to expose clean adjacency.

Model-side change:

- The encoder directly consumes `pair_evidence_bundle` in the pairwise edge path.
- `relation_hint`, `pair_support_hint`, and `signed_pair_witness` remain available.
- No bounded correction path is used.
- No signed supervision or false-admission suppression loss is used.
- Training uses the rev13-style recovery recipe.

## Data

Generated files:

- `data/graph_event_step30_weak_obs_rev17_train.pkl`
- `data/graph_event_step30_weak_obs_rev17_val.pkl`
- `data/graph_event_step30_weak_obs_rev17_test.pkl`

Split composition:

| Split | Samples | Clean | Noisy | Bundle Dim |
|---|---:|---:|---:|---:|
| train | 20000 | 10000 | 10000 | 4 |
| val | 4000 | 2000 | 2000 | 4 |
| test | 4000 | 2000 | 2000 | 4 |

Test event counts:

| Event family | Count |
|---|---:|
| edge_add | 1374 |
| edge_delete | 1448 |
| motif_type_flip | 1110 |
| node_state_update | 1284 |

## Training

Single rev17 probe:

- checkpoint dir: `checkpoints/step30_encoder_recovery_rev17`
- epochs: `20`
- best epoch: `10`
- best validation selection score: `1.350615`
- decode thresholds: clean `0.50`, noisy `0.55`
- selective rescue: off

## Recovery Results

| Row | Overall Edge F1 | Clean Edge F1 | Noisy Precision | Noisy Recall | Noisy Edge F1 |
|---|---:|---:|---:|---:|---:|
| rev6 reference | 0.7383 | 0.8215 | 0.6300 | 0.7072 | 0.6550 |
| rev13 diagnostic reference | 0.7349 | 0.8218 | 0.6026 | 0.7257 | 0.6481 |
| rev17 encoder | 0.7337 | 0.8193 | 0.5893 | 0.7453 | 0.6482 |
| rev17 trivial | 0.5011 | 0.5682 | 0.4542 | 0.4397 | 0.4340 |

Event-family edge F1 for rev17 encoder:

| Event family | Edge F1 |
|---|---:|
| edge_add | 0.7371 |
| edge_delete | 0.7318 |
| motif_type_flip | 0.7331 |
| node_state_update | 0.7320 |

## Targeted Rescue Diagnostics

Noisy split:

| Row | Hint-Missed True Recall | Hint-Missed Avg Score | Hint-Supported FP Error | HM vs Hard-Neg Win Rate |
|---|---:|---:|---:|---:|
| rev6 reference | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev13 diagnostic reference | 0.3925 | 0.4790 | 0.4529 | 0.4535 |
| rev17 encoder | 0.4580 | 0.5109 | 0.4611 | 0.4932 |
| rev17 trivial | 0.6128 | 0.5253 | 0.4751 | 0.5972 |

## Gate Check

Required recovery-level gates:

| Gate | Result | Pass? |
|---|---|---|
| Noisy edge F1 beats rev6 `0.6550` | rev17 `0.6482` | no |
| Hint-missed recall stays above rev6 | rev17 `0.4580` vs rev6 `0.2817` | yes |
| Hint-supported FP error does not materially exceed rev6 | rev17 `0.4611` vs rev6 `0.4688` | yes |
| Trivial decode remains clearly below encoder | trivial noisy F1 `0.4340` vs encoder `0.6482` | yes |

Backend rerun was not run because the first gate failed.

## Interpretation

`pair_evidence_bundle` is a real observation/representation-side mechanism signal. It improves the named rescue bottleneck beyond rev13:

- hint-missed true recall improves from `0.3925` to `0.4580`
- hint-missed average score improves from `0.4790` to `0.5109`
- hint-missed vs hard-negative win rate improves from `0.4535` to `0.4932`

The bundle remains non-leaky at the global recovery level:

- rev17 trivial noisy edge F1 is only `0.4340`
- rev17 trivial overall edge F1 is only `0.5011`
- encoder recovery remains far above trivial

However, the first probe does not beat the rev6 recovery reference:

- rev17 noisy edge F1 is `0.6482`
- rev6 noisy edge F1 is `0.6550`

The bundle strengthened the targeted rescue signal, but the global precision/recall tradeoff is still not clean enough. In particular, noisy precision dropped to `0.5893`.

Chinese summary: `pair_evidence_bundle` 比单个 `signed_pair_witness` 更会救 low-hint true edges，但第一版仍然没有把这个 targeted gain 转成更好的全局 edge recovery。它不是 leaky shortcut，但也还不是可进入 backend rerun 的 operating point。

## Conclusion

rev17 validates `pair_evidence_bundle` as a genuinely stronger targeted observation signal, but it fails the recovery gate. It should be retained as a diagnostic probe, not promoted as a Step30 recovery operating point.

Recommended next action:

- Do not run Step30c backend integration from rev17.
- Do not move to adapter/interface work.
- If continuing this family, first diagnose where the extra global false positives are coming from outside the named hint-supported false-positive subtype, especially ambiguous/mid-hint and non-rescue false-positive regimes.
