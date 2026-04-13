# Step33 Event-Edge Source Redesign Memo

## Status

This is a planning memo only.

Do not treat this as a training plan, rendered-observation plan, backend-transfer plan, or a reason to reopen parked Step22-31 micro-lines.

Step33 remains a controlled synthetic physics-like graph-event benchmark. The benchmark substrate is healthy, proposal-side learned signal remains viable, and promoted `structured_propagation_v2` with near-node-velocity weighting remains the current best learned rewrite family.

The current event-edge source-estimation implementation line is paused.

## 1. Current Event-Edge Status

The noisy `spring_retension` rewrite line has now separated three facts:

- `full_v2` is the best learned rewrite family so far, but it is not candidate-ready.
- `spring_neighbor_scope` still wins total changed-region error on noisy `spring_retension`.
- event-edge source quality is a real lever, but the current learned source-estimation path does not recover it.

The strongest event-edge diagnostic row is `denoise_from_clean_current_source`:

- noisy total changed-region error: `0.1232`
- noisy event-edge stiffness MAE: `0.0085`
- oracle event-edge patch on support `full_v2`: `0.1230`

This row nearly matches the oracle event-edge patch. It shows that event-edge source quality matters.

The learned `clean_current_estimator` did not retain that gain:

- noisy total changed-region error: `0.1299`
- support `full_v2`: `0.1297`
- current joint event-denoise: `0.1299`
- hard `stiffness_factor < 1` stiffness MAE: `0.5089`
- support `full_v2` hard `stiffness_factor < 1` stiffness MAE: `0.4787`

The estimator improved noisy clean-current source stiffness from `0.4783` to `0.3765`, but this was not enough to improve event-edge target quality or total changed-region error. Clean sanity also regressed slightly.

## 2. Why The Clean-Current Upper Bound Matters

The clean-current upper bound matters because it isolates the source problem from the rewrite architecture problem.

When clean-current edge parameters are available, applying the retension factor almost solves the event-edge target:

- the event-edge stiffness error collapses from the `0.36-0.38` learned-support band to `0.0085`
- noisy total changed-region error moves from about `0.1297` to `0.1232`
- the hard down-factor bucket is fixed rather than merely shifted

This means event-edge denoising is not a dead end scientifically. The source state is the right object.

But the upper bound depends on oracle `clean_edge_features`. It is not a deployable noisy structured observation solution. It should be retained as a diagnostic ceiling and source-design target, not as a candidate row.

## 3. Why The Current Estimator Failed

The current estimator asks a tiny MLP to infer clean-current event-edge rest/stiffness from noisy event-edge inputs plus local endpoint and structural summaries.

It failed in the way that matters:

- it did not improve noisy total changed-region error over support `full_v2`
- it did not fix the hard `stiffness_factor < 1` bucket
- it retained the same underprediction pattern seen in the previous residual source formulations
- it regressed clean sanity relative to support `full_v2`

The estimator did learn something: noisy clean-current source stiffness improved from `0.4783` to `0.3765`. The problem is that partial denoising is not enough. The event-edge target needs a much stronger source estimate, especially in down-factor buckets where noisy observations are biased low.

This is not a reason to try another tiny source MLP. The failure mode is now well identified: the current local noisy features do not expose enough source information in a form the small estimator can reliably use.

## 4. What Stronger Source/Observation Redesign Should Mean

A stronger source/observation redesign must change what information is available or how it is represented before the event-edge retension target is predicted.

It should not mean:

- another residual around noisy rule params
- another direct clean-target MLP
- another sign-aware residual head
- another small clean-current source MLP over the same features
- another deterministic event-edge rule patch

For Step33, a stronger source/observation redesign could mean one of the following bounded changes:

- explicitly separate observed spring parameters from inferred clean-current spring parameters in the structured observation schema
- add a diagnostic observation channel that represents uncertainty or confidence for rest/stiffness corruption
- predict clean-current source from a short local consistency check using endpoint geometry, current distance, incident spring context, and retension factor, rather than from a flat event-edge feature vector
- define a narrower event-edge target where source denoising and retension-factor application are evaluated as separate stages
- add a controlled oracle/noisy paired source diagnostic that measures which observation channels are actually necessary to reconstruct clean-current event-edge stiffness

The redesign must remain synthetic, structured, and evaluation-first. It should preserve exact labels and oracle support diagnostics.

## 5. Smallest Future Bounded Prototype Worth Trying

The smallest worthwhile future prototype is not another learned rewrite candidate. It is a source/observation diagnostic.

Recommended future prototype:

### Event-edge clean-current observability probe

Scope:

- `spring_retension` only
- noisy structured observation only, with clean evaluation as sanity
- existing Step33 data if possible
- no rendered observation
- no backend transfer
- no broad candidate training
- node rollout fixed/report-only

Question:

Can clean-current event-edge rest/stiffness be reconstructed from controlled subsets of structured observation channels?

Rows to compare:

- noisy observed event-edge rest/stiffness only
- noisy observed event edge plus endpoint node state
- endpoint state plus current distance and near-contact
- endpoint state plus incident spring summary
- all local structured features currently available
- oracle clean-current source

Metrics:

- clean-current rest MAE
- clean-current stiffness MAE
- event-edge target rest/stiffness after applying retension factor
- hard `stiffness_factor < 1` bucket stiffness MAE and bias
- clean sanity error

Decision rule:

- If no feature subset gets close to the clean-current upper bound, event-edge denoising should stay paused.
- If a specific channel subset recovers most of the source error, Step33 can consider a small observation-schema or source-stage redesign around that channel.

This prototype is useful because it answers whether the source is observable under the current structured regime before another learned rewrite path is attempted.

## 6. What Not To Try Again

Do not continue these event-edge paths now:

- residual-around-noisy-rule event-edge denoising
- direct clean-target prediction from noisy event-edge inputs
- sign-aware residual denoising
- tiny clean-current estimator over the same local features
- deterministic retension rule patch
- existing event-edge cleanup checkpoint patch
- event-edge denoising reconnected to near-node rollout before source observability is resolved
- active/contact/event-rule threshold or gate variants
- broad Step33 rewrite candidate training

Also do not treat `denoise_from_clean_current_source` as a candidate. It is an oracle-source upper bound.

## Decision Recommendation

Pause the current event-edge source-estimation implementation line.

Keep Step33 active as a benchmark and proposal-side line. Keep promoted `full_v2` as the best learned rewrite reference so far. Retain `denoise_from_clean_current_source` as the event-edge source upper bound.

The next meaningful event-edge step, if taken, should be an observability/source-design diagnostic. It should answer whether clean-current event-edge stiffness is recoverable from the structured observation at all. Until that is answered, do not run another small source estimator or reconnect event-edge denoising to near-node rollout.
