# Step30 rev18 Pair Evidence Precision Diagnosis

## Question

Where exactly do rev17's extra global false positives come from?

This is a diagnostic-only pack. It does not add another cue, does not train, does not run backend integration, and does not move to adapter/interface work.

## Setup

Data:

- `data/graph_event_step30_weak_obs_rev17_test.pkl`
- noisy split only for the detailed pair diagnostics

Rows compared:

- rev6 reference
- rev13 diagnostic reference
- rev17 encoder
- rev17 trivial decode

Artifacts:

- `artifacts/step30_pair_evidence_diagnostics_rev18/summary.json`
- `artifacts/step30_pair_evidence_diagnostics_rev18/pair_regime_table.csv`
- `artifacts/step30_pair_evidence_diagnostics_rev18/fp_source_table.csv`
- `artifacts/step30_pair_evidence_diagnostics_rev18/rescue_vs_non_rescue_table.csv`
- `artifacts/step30_pair_evidence_diagnostics_rev18/bundle_channel_table.csv`
- `artifacts/step30_pair_evidence_diagnostics_rev18/event_family_fp_table.csv`
- `artifacts/step30_pair_evidence_diagnostics_rev18/mute_ablation_table.csv`

## Main Finding

rev17's precision failure is not primarily the old classic high-relation false-positive problem.

The largest new false-positive burden is:

- low-hint pairs with high bundle support
- ambiguous / mid-hint pairs

The damage appears in both:

- rescue-eligible candidates
- ordinary non-rescue scoring

Chinese summary: rev17 的 bundle 确实更会救 hint-missed true edges，但它也把一批 low-hint/high-bundle-support 和 ambiguous pairs 推过阈值。问题不是单纯 rescue channel 误收，而是 bundle 信号污染了更广的 edge scoring。

## False-Positive Source Decomposition

Noisy split. Fractions are within each model's false positives.

| Model | Classic Hint-Supported FP | Ambiguous Mid-Hint FP | Low-Hint High-Bundle FP | Low-Hint Pair-Support Rescue FP | Other |
|---|---:|---:|---:|---:|---:|
| rev6 | 3430 / 36.3% | 4110 / 43.4% | 1340 / 14.2% | 579 / 6.1% | 2 / 0.0% |
| rev13 | 3106 / 28.6% | 4522 / 41.7% | 2308 / 21.3% | 857 / 7.9% | 60 / 0.6% |
| rev17 | 3085 / 26.1% | 4788 / 40.6% | 3366 / 28.5% | 541 / 4.6% | 27 / 0.2% |
| trivial | 1469 / 12.1% | 3571 / 29.4% | 6846 / 56.4% | 96 / 0.8% | 159 / 1.3% |

Compared with rev6, rev17 changes the false-positive burden like this:

- classic high-relation false positives decrease: `3430 -> 3085`
- low-hint high-bundle false positives increase sharply: `1340 -> 3366`
- ambiguous mid-hint false positives increase: `4110 -> 4788`

So the main precision damage is bundle-associated low-hint admission plus ambiguous/mid-hint over-admission.

## Rescue vs Non-Rescue Admissions

Noisy split. `rescue_eligible` means low relation hint and high pair-support hint.

| Model | Bucket | Precision | Recall Contribution | FP Count | FP Fraction |
|---|---|---:|---:|---:|---:|
| rev6 | ordinary non-rescue | 0.6984 | 0.6371 | 6305 | 66.6% |
| rev6 | rescue eligible | 0.3299 | 0.0678 | 3156 | 33.4% |
| rev13 | ordinary non-rescue | 0.6874 | 0.6380 | 6648 | 61.3% |
| rev13 | rescue eligible | 0.3216 | 0.0870 | 4205 | 38.7% |
| rev17 | ordinary non-rescue | 0.6670 | 0.6505 | 7443 | 63.0% |
| rev17 | rescue eligible | 0.3316 | 0.0945 | 4364 | 37.0% |
| trivial | ordinary non-rescue | 0.4672 | 0.3790 | 9903 | 81.6% |
| trivial | rescue eligible | 0.3799 | 0.0598 | 2238 | 18.4% |

rev17 gains recall contribution in both ordinary and rescue buckets, but false positives rise in both as well:

- ordinary FP: rev6 `6305`, rev17 `7443`
- rescue FP: rev6 `3156`, rev17 `4364`

This means the precision failure is broader than unsafe rescue admissions alone.

## Bundle-Channel Summary

Noisy split, rev17 score/decode.

| Group | Count | Positive Mean | Warning Mean | Corroboration Mean | Endpoint Mean |
|---|---:|---:|---:|---:|---:|
| hint-missed true edges | 5694 | 0.5513 | 0.4579 | 0.5123 | 0.4981 |
| true rescued positives | 2608 | 0.6199 | 0.4044 | 0.5360 | 0.5001 |
| false rescue admissions | 4364 | 0.5742 | 0.4491 | 0.4858 | 0.5012 |
| hard negatives | 13114 | 0.4714 | 0.5032 | 0.4977 | 0.4935 |
| all true-positive admissions | 17071 | 0.5588 | 0.4372 | 0.5133 | 0.4933 |
| all false positives | 11807 | 0.5696 | 0.4348 | 0.5082 | 0.4960 |

The channels do separate true rescued positives from false rescue admissions somewhat:

- true rescue positive support is higher: `0.6199` vs `0.5742`
- true rescue warning is lower: `0.4044` vs `0.4491`
- true rescue corroboration is higher: `0.5360` vs `0.4858`

But the separation is not strong enough. Across all admissions, false positives actually have slightly higher positive-support mean than true positives:

- all false positives positive-support mean: `0.5696`
- all true-positive admissions positive-support mean: `0.5588`

This points to over-triggering of the positive-support/corroboration side of the bundle.

## Top rev17 False-Positive Regimes

Noisy split, largest rev17 false-positive pair regimes.

| Relation | Support | Witness | Bundle Regime | Count | FP | Precision | GT Pos Rate | Pred Pos Rate |
|---|---|---|---|---:|---:|---:|---:|---:|
| low_mid | high | positive | support_only | 705 | 397 | 0.3350 | 0.3149 | 0.8468 |
| low_mid | high | positive | conflict | 734 | 376 | 0.3225 | 0.2766 | 0.7561 |
| ambiguous | mid | positive | support_only | 602 | 287 | 0.4427 | 0.4136 | 0.8555 |
| ambiguous | high | negative | support_only | 750 | 283 | 0.5592 | 0.5120 | 0.8560 |
| low_mid | high | negative | support_only | 851 | 278 | 0.2191 | 0.1645 | 0.4183 |

The worst regimes are not simply "high relation hint" errors. They are low/mid relation, high support, and bundle-support-driven admissions.

## Event-Family Cross-Check

Noisy split.

| Event Family | rev6 Precision | rev6 FP | rev17 Precision | rev17 FP | FP Delta |
|---|---:|---:|---:|---:|---:|
| edge_add | 0.6307 | 3080 | 0.5906 | 3859 | +779 |
| edge_delete | 0.6315 | 3202 | 0.5919 | 3983 | +781 |
| motif_type_flip | 0.6321 | 2554 | 0.5917 | 3193 | +639 |
| node_state_update | 0.6261 | 2826 | 0.5876 | 3516 | +690 |

The false-positive burden is broad across event families. It is not only an edge-add-family issue.

## Tiny Channel-Mute Diagnostic

This is not a new operating point; it is an offline diagnostic on the rev17 checkpoint. Metrics are noisy split, global pair micro metrics, so they should be read directionally rather than compared directly to per-sample macro F1.

| Row | Precision | Recall | F1 | FP |
|---|---:|---:|---:|---:|
| rev17 base micro | 0.5911 | 0.7450 | 0.6592 | 11807 |
| mute positive_support | 0.5942 | 0.7443 | 0.6609 | 11647 |
| mute false_admission_warning | 0.5934 | 0.7432 | 0.6599 | 11667 |

Muting either channel slightly reduces false positives and slightly improves micro F1. This supports the diagnosis that bundle channel consumption is too permissive, but the effect is small, so simple channel deletion is not the answer.

## Diagnosis

`pair_evidence_bundle` is not failing because the cue is leaky or useless. It is failing because the current encoder consumes the bundle globally in the edge head, letting weak bundle support raise too many low/mid-relation pairs.

Main precision failure:

- low-hint high-bundle-support false positives
- ambiguous/mid-hint false positives
- broad contamination across ordinary and rescue-eligible admissions

The bundle is doing what it was designed to do on targeted rescue, but it is not scoped tightly enough to safe rescue contexts.

## Recommended Next Move

Run exactly one small follow-up:

`Step30 rev19 rescue-scoped bundle consumption`

Narrow definition:

- keep the same rev17 `pair_evidence_bundle`
- do not add any new cue
- stop feeding the bundle globally into the edge head
- use the bundle only in a low-relation rescue-scoped path or residual
- leave ordinary edge scoring on the safer rev6-style relation/pair-support path
- evaluate recovery first; no backend rerun unless rev19 beats rev6 noisy edge F1

This is the smallest justified move because rev18 shows the problem is not lack of targeted rescue signal; it is insufficiently scoped consumption of that signal.
