# Step32 Source Of Truth Consolidation

## Decision

Step32 remains the **leading next-phase candidate**, but it is **not yet formalized as the official next-phase entry point**.

This is the conservative decision supported by the current docs. README names Step32 as the active next-phase question, and the Step31 handoff points toward a synthetic rendered / image-like bridge probe. However, the Step32 status doc explicitly says Step32 is not formally promoted, remains threshold-sensitive, remains below the retained Step30 rev6 recovery reference, and keeps backend transfer closed.

## 1. What Step32 Is Actually Trying To Test

Step32 tests whether the observation substrate can move from structured weak multi-view signals toward a more rendered / image-like synthetic observation family while staying controlled, synthetic, labeled, recovery-first, and comparable to the existing backend.

Step32 differs from Step30:

- Step30 explored weak-observation recovery and rescue mechanisms inside a structured observation regime.
- Step30 concluded that observation/representation changes were more promising than decode/admission micro-tweaks.
- Step30 retained rev6 as the current recovery reference and parked its rescue/decode/residual/admission micro-lines.
- Step32 is not another Step30 rescue tweak; it asks whether a more perceptual synthetic observation family can produce stronger recovery signal.

Step32 differs from Step31:

- Step31 introduced multiple independent weak structured views of the same graph-event recovery target.
- Step31 validated synthetic multi-view as a stronger bridge family.
- Step31 retained `step31_simple_late_fusion` as the Step31 multi-view bridge / backend-transfer reference.
- Step32 moves beyond structured multi-view hints toward synthetic rendered / image-like observations, while preserving clean graph/event labels as the recovery target.

Step32 differs from Step33:

- Step33 was a physics-like local event rewrite diagnostic line focused on noisy `spring_retension`, source/target boundaries, staged rewrite targets, and oracle support.
- Step33 implementation is now frozen as diagnostic-only.
- Step32 is an observation/recovery bridge question, not a Step33 rewrite continuation.

## 2. Current Step32 Fixed Protocol

The current Step32 candidate line has a fixed comparison protocol.

Default threshold setting:

- clean threshold: `0.50`
- noisy threshold: `0.50`

Validated calibrated threshold setting:

- clean threshold: `0.30`
- noisy threshold: `0.30`

The Step32 status doc states that threshold fiddling is closed for current candidate comparisons. Future comparisons for the current line should use exactly these two settings.

Retained isolated Step32 candidate references:

- best calibrated isolated reference: `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30`
- best observed default-threshold isolated reference: `step32_rendered_bridge_candidate_scale @ clean=0.50, noisy=0.50`

The default-threshold reference carries a variance caveat. Same-scale repeats confirm meaningful default-threshold behavior, but the original candidate_scale default result is above the repeat mean.

Canonical comparisons:

- Step30 rev6 retained recovery reference
- Step31 retained `step31_simple_late_fusion` reference
- trivial rendered baselines
- `step32_rendered_bridge_candidate_next` under the calibrated `0.30 / 0.30` protocol
- `step32_rendered_bridge_candidate_scale` under the default `0.50 / 0.50` protocol

Backend transfer remains closed. Formal Step32 checkpoint promotion is not supported by the current docs.

## 3. What Has Actually Been Established

Step32 positive signal is real.

- Step32 infrastructure worked end to end.
- The early default `0.50` operating point collapsed, but calibrated evaluation exposed nonzero recovery signal.
- Step32 later beat trivial rendered baselines under calibrated evaluation.
- candidate_main became a strong isolated candidate at the validated calibrated threshold.
- candidate_next improved calibrated and default-threshold behavior.
- candidate_scale improved the default-threshold operating point and became the best observed default-threshold isolated reference.

Step32 remains threshold-sensitive.

- candidate_next is the best calibrated isolated reference at `0.30 / 0.30`.
- candidate_scale is the best observed default-threshold reference at `0.50 / 0.50`.
- candidate_scale does not improve the best calibrated reference; under `0.30 / 0.30`, it regresses slightly versus candidate_next.
- the Step32 status doc explicitly preserves separate calibrated and default-threshold references.

Step32 remains variance-sensitive at the default threshold.

- same-scale candidate_scale repeats showed default overall F1 mean `0.5454` with range `0.5185-0.5611`.
- the original candidate_scale default result, `0.5860`, is better than the repeat mean and should be treated as a high-end seed.
- calibrated performance is more stable, with overall F1 mean `0.5834` and range `0.5806-0.5886`.

Step32 is still below Step30 rev6 in the documented recovery comparison.

- Step30 rev6 reference: overall F1 `0.7452`, noisy F1 `0.6657`.
- best Step32 calibrated isolated reference, candidate_next at `0.30 / 0.30`: overall F1 `0.5859`, noisy F1 `0.5268`.
- best Step32 default-threshold isolated reference, candidate_scale at `0.50 / 0.50`: overall F1 `0.5860`, noisy F1 `0.5193`.
- same-scale default-threshold repeat mean: overall F1 `0.5454`, noisy F1 `0.4724`.

Backend transfer remains closed.

- no backend transfer was run for candidate_scale.
- the Step32 status doc records `beats_step30_rev6_overall_f1: false`, `beats_step30_rev6_noisy_f1: false`, and `backend_transfer_rerun: false` for candidate_next and candidate_scale.

## 4. Real Remaining Blocker

The remaining blocker is an interaction of:

- threshold sensitivity
- default-threshold seed variance
- calibrated saturation near candidate_next / candidate_scale
- mechanism gap beyond scale

Threshold sensitivity matters because the best calibrated and best default-threshold references are different rows. The line no longer collapses only because of calibration, but the operating point still materially changes the interpretation.

Same-scale variance matters because default-threshold gains are real but seed-sensitive. The candidate_scale default result is the best observed default-threshold isolated reference, but the repeat mean is lower.

Scale alone is no longer enough. candidate_scale improved default-threshold behavior but did not improve the best calibrated operating point. The Step32 status doc says no more scale-only escalation is justified without a new mechanism.

The blocker is not Step33. Step33 preserved useful diagnostic knowledge but is frozen. The blocker is not Step22-29 local interaction tuning. That line has already retained `Step26 proposal + RFT1` as a branch candidate. The blocker is whether Step32 has a strong enough observation/representation mechanism to exceed the retained Step30 rev6 recovery reference and justify backend transfer later.

## 5. Formal Decision

Chosen decision:

**Step32 remains the leading candidate, but is not yet formalized as the official next-phase entry point.**

The evidence is strong enough to preserve Step32 candidate references and make Step32 the leading direction. It is not strong enough to formalize Step32 as the next official phase entry because the Step32 source docs explicitly say:

- Step32 is not formally promoted.
- Step32 remains threshold-sensitive.
- Step32 remains below Step30 rev6.
- backend transfer remains closed.

## 6. What Must Happen First

Before Step32 can be formalized as the official next-phase entry point, exactly one bounded planning action should happen:

**Write a Step32 mechanism-gap decision memo.**

That memo should decide whether there is a specific, evidence-bounded Step32 mechanism worth specifying next, under the fixed protocol, or whether Step32 should remain a retained isolated candidate line only.

The memo should be planning-only:

- no code
- no training
- no eval
- no backend transfer
- no threshold tuning
- no scale-only escalation
- no new rendered observation generation

It should use the current fixed Step32 protocol and retained references, and it should decide whether a future mechanism is justified by the documented blocker: threshold sensitivity, default-threshold variance, calibrated saturation, and the gap to Step30 rev6.

## 7. What Not To Do Next

Do not formally promote Step32 yet.

Do not reopen backend transfer.

Do not tune thresholds for the current Step32 candidates.

Do not run more scale-only Step32 escalation.

Do not replace the split retained Step32 references with a single promoted checkpoint.

Do not reopen Step30 rescue/decode/residual/admission micro-lines.

Do not reopen the Step31 learned-vs-late-fusion micro-line.

Do not demote `step31_simple_late_fusion` as the retained Step31 reference.

Do not reopen the current Step33 implementation line.

Do not reopen parked Step22-29 noisy multievent interaction local tweaking.

Do not add rendered observation as an immediate implementation step.

Do not broaden to real-world data, raw real-world images, hypergraphs, LLM integration, or backend joint training.

## Final Status

Step32 is the leading next-phase candidate, but not yet the formal next-phase entry point. The next action should be a planning-only Step32 mechanism-gap decision memo, not implementation.

README should remain unchanged now. This memo consolidates Step32 status but does not change repo-wide defaults, backend-transfer status, retained branch status, or public phase entry points.
