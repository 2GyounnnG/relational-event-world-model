# Step32 Mechanism Gap Decision Memo

## Decision

Choose option **(b): do not specify a new Step32 prototype now; keep Step32 as retained isolated candidates only**.

Step32 has real isolated positive signal, but the available docs do not yet justify exactly one evidence-bounded mechanism prototype. The current source of truth supports preserving the fixed protocol and retained isolated references, not reopening implementation.

## 1. Current Step32 Status

Step32 is healthy as a controlled synthetic rendered / image-like bridge probe.

It is trying to test whether the observation substrate can move from structured weak multi-view signals toward a more rendered / image-like synthetic observation family while keeping the task:

- synthetic
- fully controlled
- fully labeled
- recovery-first
- interpretable
- comparable to the existing backend
- isolated from backend joint training

Step32 is retained as a leading candidate direction, but it is not formally promoted.

Current retained Step32 isolated references:

- best calibrated isolated reference: `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30`
- best observed default-threshold isolated reference: `step32_rendered_bridge_candidate_scale @ clean=0.50, noisy=0.50`

The canonical fixed protocol is:

- default threshold:
  - clean `0.50`
  - noisy `0.50`
- validated calibrated threshold:
  - clean `0.30`
  - noisy `0.30`

Threshold fiddling is closed for current candidate comparisons. The retained comparisons are:

- Step30 rev6 retained recovery reference
- Step31 retained `step31_simple_late_fusion` reference
- trivial rendered baselines
- `step32_rendered_bridge_candidate_next` under the calibrated `0.30 / 0.30` protocol
- `step32_rendered_bridge_candidate_scale` under the default `0.50 / 0.50` protocol

Backend transfer remains closed. Formal Step32 checkpoint promotion is not supported by the current docs.

## 2. What Step32 Has Actually Established

Step32 has established real isolated positive signal.

- infrastructure worked end to end
- calibrated evaluation exposed nonzero recovery signal after the early default `0.50` operating point collapsed
- Step32 later beat trivial rendered baselines under calibrated evaluation
- candidate_main became the first strong isolated candidate at the validated calibrated threshold
- candidate_next improved calibrated and default-threshold behavior
- candidate_scale improved default-threshold behavior and became the best observed default-threshold isolated reference

The calibrated/default split is real.

- candidate_next is the best calibrated isolated reference at `0.30 / 0.30`
- candidate_scale is the best observed default-threshold reference at `0.50 / 0.50`
- candidate_scale regresses slightly versus candidate_next under calibrated `0.30 / 0.30`
- therefore the repo should preserve split Step32 references rather than force one promoted checkpoint

The same-scale variance caveat is real.

- candidate_scale default overall F1: `0.5860`
- same-scale default overall F1 mean: `0.5454`
- same-scale default overall F1 range: `0.5185-0.5611`
- calibrated same-scale overall F1 mean: `0.5834`
- calibrated same-scale overall F1 range: `0.5806-0.5886`

Interpretation: default-threshold progress is real but seed-sensitive. Calibrated performance is more stable, but appears saturated around the candidate_next / candidate_scale level under pure scale-up.

Step32 remains below Step30 rev6.

- Step30 rev6 retained recovery reference: overall F1 `0.7452`, noisy F1 `0.6657`
- Step32 candidate_next calibrated: overall F1 `0.5859`, noisy F1 `0.5268`
- Step32 candidate_scale default: overall F1 `0.5860`, noisy F1 `0.5193`
- same-scale default repeat mean: overall F1 `0.5454`, noisy F1 `0.4724`

Backend transfer remains closed.

- candidate_next and candidate_scale do not beat Step30 rev6 overall or noisy F1
- backend transfer was not rerun for candidate_scale
- the Step32 status doc explicitly says not to reopen backend transfer yet

## 3. The Real Remaining Blocker

The remaining blocker is an interaction of:

- threshold sensitivity
- default-threshold seed variance
- calibrated saturation
- missing mechanism beyond scale

It is not just threshold sensitivity. Step32 no longer exists only at calibrated thresholds: candidate_scale has meaningful default-threshold behavior. But the best calibrated and best default-threshold references are different rows, and the operating point still materially changes the conclusion.

It is not just seed variance. Same-scale repeats confirm real default-threshold progress, but the original candidate_scale default result is above the repeat mean. The row should be retained with a variance caveat, not promoted.

It is not just scale. Pure scale-up helped default-threshold behavior, but did not improve the best calibrated operating point. The Step32 docs explicitly say no more scale-only escalation is justified without a new mechanism.

The missing piece is a stronger observation/representation mechanism that can improve under the fixed protocol and move toward or beyond Step30 rev6. The current docs do not specify such a mechanism tightly enough to justify implementation.

## 4. Formal Decision

Chosen decision:

**Do not specify a new prototype; keep Step32 as retained isolated candidates only.**

Reason:

- Step32 positive signal is real.
- Step32 is the leading next-phase candidate.
- Step32 has not reached Step30 rev6.
- backend transfer remains closed.
- current Step32 docs do not define one concrete next mechanism beyond saying scale-only and threshold-tuning are not justified.

The evidence is enough to preserve Step32 candidate references, not enough to start another implementation run.

## 5. What To Preserve

Preserve Step32 docs:

- `docs/STEP32_SYNTHETIC_RENDERED_BRIDGE_PROBE.md`
- `docs/STEP32_RENDERED_BRIDGE_CANDIDATE_STATUS.md`
- `docs/STEP32_SOURCE_OF_TRUTH_CONSOLIDATION.md`
- `docs/STEP32_MECHANISM_GAP_DECISION_MEMO.md`

Preserve direct handoff / comparison docs:

- `docs/STEP30_CONSOLIDATION_TO_NEXT_PHASE.md`
- `docs/STEP31_MULTI_VIEW_OBSERVATION_BRIDGE.md`
- `docs/STEP31_LEARNED_VS_LATE_FUSION_CONSOLIDATION.md`
- `docs/NEXT_PHASE_ENTRY_PLAN_AFTER_STEP33.md`

Preserve retained candidate references:

- `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30`
- `step32_rendered_bridge_candidate_scale @ clean=0.50, noisy=0.50`

Preserve comparison rows:

- Step30 rev6 retained recovery reference
- Step31 `step31_simple_late_fusion`
- trivial rendered baselines
- candidate_next calibrated
- candidate_scale default with variance caveat

Preserve current policy:

- split calibrated/default Step32 references
- fixed threshold protocol
- no formal Step32 promotion
- backend transfer closed

## 6. What Not To Do Next

Do not specify or implement a new Step32 prototype yet.

Do not formally promote Step32.

Do not reopen backend transfer.

Do not tune thresholds for current Step32 candidates.

Do not run more scale-only Step32 escalation.

Do not replace split retained Step32 references with one promoted checkpoint.

Do not resume trivial rendered-baseline chasing as a main line.

Do not reopen Step30 rescue/decode/residual/admission micro-lines.

Do not reopen Step31 learned-vs-late-fusion micro-lines.

Do not demote `step31_simple_late_fusion` as the retained Step31 reference.

Do not reopen the current Step33 implementation line.

Do not reopen parked Step22-29 noisy multievent interaction local tweaking.

Do not broaden to real-world data, raw real-world images, hypergraphs, LLM integration, backend joint training, or Step33-style rewrite diagnostics.

## Final Status

Step32 remains the leading next-phase candidate, but only as retained isolated candidate references for now. The repo should wait for a concrete, evidence-bounded mechanism definition before resuming Step32 implementation.

README should remain unchanged now. This memo does not change repo-wide defaults, retained branch status, backend-transfer status, formal Step32 promotion status, or public phase entry points.
