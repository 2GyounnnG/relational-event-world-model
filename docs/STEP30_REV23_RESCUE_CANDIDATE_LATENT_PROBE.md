# Step30 Rev23 Rescue Candidate Latent Probe

Rev23 is a recovery/diagnostic-first rescue-scope representation probe.

It does not add a new weak observation cue, run Step30c backend integration, update proposal/rewrite backends, or continue the rev19-rev22 scalar rescue-residual micro-line.

## Short Chinese Summary

rev23 明显比 scalar residual 线更有信号：一个小的 first-class rescue-candidate latent probe 把 safe missed true edges 和 unsafe rescue false admissions 分得更开。它不是最终 operating point，也不是 backend 结果，但它证明当前 rescue-scope 输入里确实还有可学习的 separability。下一步应该把这个 latent probe 集成成一个小的 recovery-side rescue-aware head，而不是继续调 residual loss。

## Question

Can a small first-class rescue-candidate latent representation separate:

- `safe_missed_true_edge`
- `low_hint_pair_support_false_admission`
- `ambiguous_rescue_candidate`

better than rev19-rev22 scalar residual shaping?

## Setup

Candidate scope:

- noisy weak-observation split only
- `relation_hint < 0.50`
- `pair_support_hint >= 0.55`

Data:

- train: `data/graph_event_step30_weak_obs_rev17_train.pkl`
- val: `data/graph_event_step30_weak_obs_rev17_val.pkl`
- test: `data/graph_event_step30_weak_obs_rev17_test.pkl`

Candidate counts:

| split | candidates | safe_missed_true_edge | low_hint_pair_support_false_admission | ambiguous_rescue_candidate |
|---|---:|---:|---:|---:|
| train | 74,587 | 15,495 | 12,282 | 46,810 |
| val | 14,782 | 3,027 | 2,388 | 9,367 |
| test | 14,874 | 3,070 | 2,479 | 9,325 |

Probe:

- tiny MLP over extracted rescue-candidate features
- latent dim: `32`
- hidden dim: `64`
- target: 3-way candidate type classification
- selection metric: validation safe-vs-unsafe AP
- best epoch: `14`
- best validation safe AP: `0.4176`

Checkpoint:

- `checkpoints/step30_rescue_candidate_latent_probe_rev23/best.pt`

Artifacts:

- `artifacts/step30_rescue_candidate_latent_probe_rev23/`

## Features

No new weak cue is introduced.

The probe uses existing rescue-scope signals only:

- relation hint
- pair support hint
- signed pair witness
- `pair_evidence_bundle` channels
- bundle positive-minus-warning margin
- rev6 score/logit
- rev19 score/residual
- rev21 score/residual
- hint/support endpoint degree summaries
- hint/support common-neighbor summaries
- rev6 endpoint node latents as sum, absolute difference, and product

This is meaningfully different from rev19-rev22 because the latent probe learns an explicit candidate representation and class boundary before any edge-logit correction.

## Rescue-Candidate Ranking

Safe-vs-unsafe test ranking:

| scorer | safe AP | safe AUROC | safe score mean positive | safe score mean negative |
|---|---:|---:|---:|---:|
| rev23_latent_probe | 0.4293 | 0.7214 | 0.3949 | 0.2387 |
| rev6_score | 0.3429 | 0.6578 | 0.5421 | 0.4633 |
| rev19_residual | 0.2931 | 0.6232 | 0.5592 | 0.3644 |
| pair_support_hint | 0.2928 | 0.6188 | 0.6588 | 0.6287 |
| bundle_margin | 0.2914 | 0.6194 | 0.0919 | -0.0758 |
| rev21_residual | 0.2878 | 0.6177 | 0.2114 | -0.0036 |

The latent probe is substantially better than both scalar residual baselines and the strongest raw scalar baseline.

## 3-Way Classification

| class | count | predicted | precision | recall | F1 |
|---|---:|---:|---:|---:|---:|
| safe_missed_true_edge | 3,070 | 2,525 | 0.4550 | 0.3743 | 0.4107 |
| low_hint_pair_support_false_admission | 2,479 | 3,057 | 0.6529 | 0.8052 | 0.7211 |
| ambiguous_rescue_candidate | 9,325 | 9,292 | 0.8432 | 0.8402 | 0.8417 |

Confusion summary:

| true class | predicted safe | predicted low-hint false | predicted ambiguous |
|---|---:|---:|---:|
| safe_missed_true_edge | 1,149 | 632 | 1,289 |
| low_hint_pair_support_false_admission | 315 | 1,996 | 168 |
| ambiguous_rescue_candidate | 1,061 | 429 | 7,835 |

The probe is not perfect. Safe positives remain the hardest class. But unlike the scalar residuals, it learns a useful candidate-level separation.

## Rescue Admission Simulation

This is rescue-scope only, not backend integration.

At the rev21 current rescue budget of 4,862 admitted candidates:

| scorer | precision | recall | F1 | admitted | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| rev23_latent_probe | 0.3686 | 0.5837 | 0.4518 | 4,862 | 1,792 | 3,070 | 1,278 |
| rev6_score | 0.3283 | 0.5199 | 0.4024 | 4,862 | 1,596 | 3,266 | 1,474 |
| rev19_residual | 0.2908 | 0.4606 | 0.3565 | 4,862 | 1,414 | 3,448 | 1,656 |
| rev21_residual | 0.2865 | 0.4537 | 0.3512 | 4,862 | 1,393 | 3,469 | 1,677 |
| rev21 threshold decode | 0.3414 | 0.5407 | 0.4186 | 4,862 | 1,660 | 3,202 | 1,410 |

At the rev19 current rescue budget of 6,500 admitted candidates:

| scorer | precision | recall | F1 | admitted | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|
| rev23_latent_probe | 0.3268 | 0.6919 | 0.4439 | 6,500 | 2,124 | 4,376 | 946 |
| rev6_score | 0.2946 | 0.6238 | 0.4002 | 6,500 | 1,915 | 4,585 | 1,155 |
| rev19_residual | 0.2715 | 0.5749 | 0.3689 | 6,500 | 1,765 | 4,735 | 1,305 |
| rev21_residual | 0.2657 | 0.5625 | 0.3609 | 6,500 | 1,727 | 4,773 | 1,343 |
| rev19 threshold decode | 0.3040 | 0.6436 | 0.4130 | 6,500 | 1,976 | 4,524 | 1,094 |

The latent probe improves rescue-scope precision/recall tradeoff over the scalar residual rankings and the historical threshold operating points.

## Lightweight Recovery-Side Simulation

This is an upper-triangle recovery-side simulation only. It is not Step30c backend integration.

Setup:

- rev6 current decode outside rescue scope
- rescue-scope admissions selected by each scorer
- noisy split only

At the rev21 rescue budget:

| simulation | precision | recall | F1 |
|---|---:|---:|---:|
| rev6_current_decode | 0.6306 | 0.7049 | 0.6657 |
| rev21_current_decode | 0.6310 | 0.7095 | 0.6680 |
| rev23_latent_probe_rev21_budget | 0.6361 | 0.7153 | 0.6734 |
| rev6_score_rev21_budget | 0.6285 | 0.7067 | 0.6653 |
| rev19_residual_rev21_budget | 0.6214 | 0.6988 | 0.6578 |
| rev21_residual_rev21_budget | 0.6206 | 0.6979 | 0.6570 |

This suggests the latent probe is likely useful for recovery-side integration, but it is still not a backend result.

## Trivial Sanity

The latent probe learns more than a trivial ranking from raw rescue features:

- best raw scalar AP among listed baselines is rev6 score at `0.3429`;
- rev23 latent AP is `0.4293`;
- rev19/rev21 residual AP is only `0.2931` / `0.2878`.

The cue family has not collapsed into a trivial decode. The rescue scope remains hard, but there is real separability when represented explicitly.

## Gate Judgment

rev23 does not trigger backend integration in this task.

Reason:

- this is a representation probe, not an integrated recovery model;
- no backend-facing structured graph decode was formally retained;
- the correct next step is to integrate the latent probe into recovery-side decoding/training and evaluate recovery first.

## Diagnosis

The rescue scope now appears meaningfully more separable with a first-class latent probe than with scalar residual shaping.

This supports a later integrated rescue-aware recovery model. It does not support moving directly to adapter/interface work or backend integration.

## Recommendation

Run exactly one small rev24 recovery-side integration probe: add a tiny `rescue_candidate_latent` auxiliary head to the Step30 encoder, keep the ordinary rev6-style edge path frozen, and use the latent safe score for bounded rescue-scope admission selected on validation recovery metrics.
