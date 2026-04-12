# Step30 Rescue-Residual Micro-Line Consolidation

This document consolidates the Step30 rev19-rev22 rescue-residual sub-line and defines the next mechanism direction.

It does not add a new cue, run backend integration, update proposal/rewrite backends, or reopen the parked Step22-29 noisy interaction line.

## Short Chinese Summary

rev19-rev22 已经把 rescue-residual 这条微调线走到了当前阶段的边界：rescue-scoped bundle consumption 是对的，残差 shaping 也确实能在 recall 和 false admission 之间移动 operating point；但 safe missed true edges 和 low-hint pair-support false admissions 在当前表示里仍然分不开。继续加一个小 scalar residual loss 只会在“更安全但少救边”和“多救边但误收多”之间来回摆动。

## Fixed Context

- Stable defaults remain unchanged.
- The broad clean default is `W012`.
- The broad noisy default is `RFT1 + calibrated P2`.
- The interaction-aware alternative is `I1520`.
- The consistency reference is `C005`.
- The rollout-aware reference is `R050`.
- The retained noisy interaction-aware branch candidate remains `Step26 proposal + RFT1`, not the broad noisy default.
- For Step30, `rev6 missed-edge evidence recovery with retained rev6 decode settings` remains the current best recovery reference.
- The rev7-rev12 rescue-decode micro-line is parked.
- The rev13-rev16 signed-pair-witness micro-line is parked.

## What Each Step Established

| step | change | what improved | unresolved problem |
|---|---|---|---|
| rev19 | Restricted `pair_evidence_bundle` to a low-relation rescue-scoped residual path while freezing the ordinary rev6-style path. | Reduced broad contamination from global bundle consumption; ordinary non-rescue false positives dropped. | Rescue-eligible false positives increased; rescue precision remained weak; noisy F1 stayed below rev6. |
| rev20 | Diagnostic-only rescue-scope analysis. | Showed the remaining failure is inside rescue scope, not broad global contamination. | Safe missed true edges and low-hint pair-support false admissions receive very similar residual/score treatment. |
| rev21 | Added direct rescue residual contrast/suppression. | Unsafe positive residuals dropped sharply; rescue-scope precision improved; noisy F1 barely exceeded rev6. | Suppression was too blunt and suppressed too many safe missed true edges; targeted rescue recall regressed toward rev6. |
| rev22 | Added safe-positive preservation on top of rev21. | Safe missed true admission and hint-missed recall recovered partly. | Unsafe false admissions rose again; noisy F1 fell below rev6; the precision/recall tradeoff remained unresolved. |

## Compact Comparison

All rows use the rev17 Step30 test split and retained `clean=0.50`, `noisy=0.55` hard decode thresholds.

| model | overall edge F1 | noisy precision | noisy recall | noisy edge F1 | rescue precision | rescue recall | hint-missed recall | hint-supported FP error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rev6 | 0.7383 | 0.6300 | 0.7072 | 0.6550 | 0.3299 | 0.5062 | 0.2817 | 0.4688 |
| rev19 | 0.7328 | 0.6006 | 0.7297 | 0.6475 | 0.3040 | 0.6436 | 0.3730 | 0.4688 |
| rev21 | 0.7435 | 0.6306 | 0.7115 | 0.6571 | 0.3414 | 0.5407 | 0.3003 | 0.4688 |
| rev22 | 0.7401 | 0.6176 | 0.7205 | 0.6537 | 0.3247 | 0.5971 | 0.3361 | 0.4688 |

Rescue-scope counts:

| model | admitted | TP | FP | FN |
|---|---:|---:|---:|---:|
| rev6 | 4,710 | 1,554 | 3,156 | 1,516 |
| rev19 | 6,500 | 1,976 | 4,524 | 1,094 |
| rev21 | 4,862 | 1,660 | 3,202 | 1,410 |
| rev22 | 5,645 | 1,833 | 3,812 | 1,237 |

Residual behavior:

| model | unsafe positive residual frac | safe too-small residual frac | interpretation |
|---|---:|---:|---|
| rev19 | 0.7558 | 0.1368 | strong rescue, but unsafe false admissions are boosted too often |
| rev21 | 0.4970 | 0.3661 | much safer, but safe rescue is suppressed too much |
| rev22 | 0.6311 | 0.2365 | more safe rescue, but unsafe admissions reopen |

## Park / Retain Decision

Retain `rev21` as the diagnostic reference for the rescue-residual shaping sub-line.

Reason:

- It is the cleanest proof that direct unsafe residual suppression can reduce unsafe boosts.
- It has the best noisy edge F1 among rev19-rev22: `0.6571`.
- It improves rescue-scope precision relative to rev19: `0.3040 -> 0.3414`.

But rev21 is diagnostic only, not a usable retained Step30 operating point.

Reason:

- Its noisy F1 gain over rev6 is tiny: `0.6571` vs `0.6550`.
- It gives back most of rev19's targeted missed-edge rescue signal.
- It does not provide a convincing backend-rerun gate.

The current Step30 recovery reference remains rev6, not rev21.

Step30 is not ready for backend rerun, adapter/interface work, or backend joint training based on rev19-rev22.

The rev19-rev22 rescue-residual micro-line should now be parked unless a genuinely new observation/representation mechanism is introduced.

## Why Scalar Residual Tweaks Did Not Close The Gap

The failure is not that the rescue path has no signal. The failure is that the current representation does not expose enough separability for a scalar residual to act safely.

rev20 showed that safe and unsafe rescue candidates differ weakly on:

- `pair_evidence_bundle` positive support
- false-admission warning
- positive-minus-warning margin
- pair support
- base score

But the admitted unsafe false group remains very close to the admitted safe group. This makes scalar residual shaping underpowered:

- suppress unsafe residuals and safe missed true edges are suppressed too;
- preserve safe residuals and unsafe false admissions reopen;
- change thresholds and the model mostly trades recall away rather than exposing a clean high-precision rescue band.

In short: the current rescue branch is being asked to solve a representation problem with a scalar correction.

## Next-Mechanism Design Memo

### Missing Information

The current rescue-scope representation still fails to expose a stable latent distinction between:

- safe low-hint true-edge rescues;
- unsafe low-hint pair-support false admissions;
- ambiguous low-relation candidates that should remain conservative.

The available scalar signals are too entangled. Positive rescue evidence also fires on unsafe admissions, and warning evidence is not strong enough to define a clean safety boundary.

### Mechanism Family

The next mechanism should be a rescue-candidate representation family, not another residual-loss family.

Proposed family:

`rescue_candidate_latent_probe`

Core idea:

- Learn a first-class rescue-scope pair representation before edge-logit correction.
- Train it to separate candidate types directly, not merely to push one scalar residual up or down.
- Keep ordinary edge scoring frozen or rev6-style.
- Use the existing `pair_evidence_bundle`, relation hint, pair support, endpoint node latents, and base edge score as inputs.
- Do not add a new weak observation cue in the first probe.

Candidate labels:

- `safe_missed_true_edge`
- `low_hint_pair_support_false_admission`
- `ambiguous_rescue_candidate`

Possible objective:

- a tiny candidate-type classifier over a rescue latent;
- optionally a supervised contrastive term that pulls safe rescues away from unsafe false admissions;
- use the latent for diagnostics first, and only then consider a bounded rescue decision.

This is meaningfully different from rev19-rev22 because it does not ask a scalar residual to both encode and act. It first tests whether the representation can expose separability.

### Smallest First Experiment

Run exactly one recovery-first probe:

1. Keep the rev6 ordinary edge path frozen.
2. Use the rev17/rev19 rescue-scope candidate definition: `relation_hint < 0.50`, `pair_support_hint >= 0.55`.
3. Add a tiny `rescue_candidate_latent` MLP over existing rescue-scope features only.
4. Train a 3-way candidate-type auxiliary head on rescue-scope pairs.
5. Evaluate representation quality before any backend rerun:
   - safe-vs-unsafe candidate AP/AUC;
   - safe missed true recall at fixed unsafe admission rate;
   - rescue-scope precision/recall/F1;
   - noisy edge F1 only if the latent is used for a bounded rescue decision.

Success gate:

- safe-vs-unsafe ranking clearly improves over rev21/rev22;
- rescue-scope precision improves without collapsing safe rescue recall;
- noisy edge F1 beats rev6 by more than a razor-thin margin.

No backend integration should run unless this recovery-side gate is clearly met.

## Recommended Next Action

Park the rev19-rev22 rescue-residual micro-line now, and run exactly one small first experiment in the new `rescue_candidate_latent_probe` family.
