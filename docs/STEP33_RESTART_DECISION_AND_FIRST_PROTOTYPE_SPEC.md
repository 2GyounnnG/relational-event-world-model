# Step33 Restart Decision And First Prototype Spec

## Decision

Step33 should **stop implementation and preserve the current line as diagnostic-only**.

Do not restart Step33 for another bounded implementation cycle now. The current evidence is strong enough to preserve the benchmark, rows, checkpoints, and memos, but not strong enough to justify another local implementation tweak or a broad candidate phase.

## 1. Current Step33 Status

The Step33 synthetic physics-like benchmark substrate remains healthy. Event scope and changed region diverge in useful ways, `oracle_scope` still leaves clear headroom, and the clean/noisy structured smoke benchmarks are implemented and usable.

Proposal-side learned signal remains viable. The remaining issue is rewrite quality on noisy structured `spring_retension`, not proposal-side feasibility.

The current best references are:

| role | retained row |
| --- | --- |
| best trivial structured baseline | `spring_neighbor_scope` |
| best full learned rewrite family reference | promoted `structured_propagation_v2` with near-node-velocity weighting |
| retained staged target | `source_held_out_full` |
| best source-side diagnostic row | guarded 5-view event-edge source patch |
| retained best diagnostic assembly | `composed_source_edge_force_rollout` |
| stable trained diagnostic row | `combined_source_edge_force` |

The current implementation line is paused. The retained implementation-side status is:

- `source_held_out_full` remains the retained staged diagnostic target.
- guarded 5-view event-edge source patch remains the retained source-side redesign reference.
- `source_patched_rollout_distance` remains a positive reconnect reference for changed-edge `current_distance`.
- `composed_source_edge_force_rollout` remains the retained best diagnostic assembly reference.
- `combined_source_edge_force` remains a stable positive trained diagnostic row.
- neither `composed_source_edge_force_rollout` nor `combined_source_edge_force` is candidate-ready.
- `spring_neighbor_scope` remains ahead on stratified noisy test.

## 2. Why The Current Implementation Line Stopped

`combined_source_edge_force` is real. The three-seed stability check showed that it is stable rather than seed noise, and it consistently beats `source_patched_rollout_distance`.

It is not promotable. The trained combined row does not beat the retained no-training composed diagnostic assembly. On stratified noisy test:

- `composed_source_edge_force_rollout`: `0.1131`
- `combined_source_edge_force` three-seed mean: `0.1134`
- `spring_neighbor_scope`: `0.1101`

The stratified noisy test gap is stable enough that another small local tweak is not justified. The combined row improved over one parent row, but it did not change the plateau. It still depends on oracle support, guarded multiview source diagnostics, and staged benchmark-specific assembly. That makes it useful as a diagnostic row, not as a candidate phase entry.

The line is no longer worth more local variants because the obvious local knobs have already been tested:

- tiny residual and gate heads
- denoising target heads
- propagation-biased edge heads
- event-rule and active/contact patches
- tiny source estimators and kNN-style source probes
- rollout-only refinements and frozen-edge rollout isolation
- changed-edge-only target narrowing
- guarded multiview source patching
- source-patched `current_distance` reconnect
- force-frame rollout representation
- trained combined source+edge+force row

The remaining gap is not behaving like one more residual head, weighting change, source estimator, rollout head, or `current_distance` tweak away.

## 3. What Has Actually Been Established

Proposal-side viability is established. Step33 can support learned proposal-side signal under the synthetic physics-like substrate.

`source_held_out_full` is a real retained staged target. It beats promoted `full_v2` across original noisy and stratified noisy checks, and it also beats the reduced `aux_plus_non_event_params` row. It remains diagnostic rather than candidate-ready because event-edge source recovery and clean sanity remain unresolved in the broader rewrite frame.

Source-separated scoring proved that the learnable rewrite target is already strong and close to `spring_neighbor_scope`. The old full learned family was no longer the main blocker after the staged source-held-out separation. The gap became concentrated in the held-out source channel and then, after source patching, in rollout-centered residuals.

Paired and then guarded 5-view multiview source redesign produced a real source-side signal. One additional independent noisy structured source view improved clean-current event-edge observability, and 5-view aggregation made that source protocol stronger and more stable. Guarded 5-view patching materially improved staged total changed-region error while preserving the learnable target strengths of `source_held_out_full`.

The composed and combined line established that source-patched edge/current-distance assembly and force-frame rollout can add constructively. The no-training `composed_source_edge_force_rollout` row is the retained best diagnostic assembly reference. The trained `combined_source_edge_force` row is stable and positive, but it does not beat the composed row or close the stratified noisy test gap to `spring_neighbor_scope`.

## 4. The Real Remaining Blocker

The unresolved blocker is no longer mainly event-edge source after guarded 5-view patching. Event-edge source recovery was a major blocker earlier, and the clean-current upper bound showed that the lever was real. The paired and 5-view source diagnostics then produced a meaningful source-side protocol, enough that event-edge source is no longer the dominant post-patch gap.

The remaining blocker is mainly an interaction centered on rollout representation, with residual `current_distance` and preservation-geometry effects. The post-patch decomposition showed that endpoint, one-hop, and two-hop changed-node velocity remained the dominant gap area, with smaller `current_distance` residuals. Force-frame rollout improved rollout metrics but did not produce a candidate-ready total-error row. The combined row was stable but still did not beat the composed assembly or `spring_neighbor_scope` on stratified noisy test.

This is not evidence of benchmark failure. The benchmark remains useful: it separated proposal viability, source observability, staged target learning, assembly effects, and rollout weakness. The issue is that the current implementation family has reached a diagnostic plateau under this benchmark.

## 5. Restart Decision

The decision is: **stop Step33 implementation and preserve it as diagnostic-only**.

Do not define a new restart prototype. Do not run another bounded implementation cycle. The next useful Step33 work, if any, must begin from a new redesign decision rather than from another local implementation variant.

## 6. What To Preserve

Preserve the Step33 planning and status record:

- `docs/CODEX_STEP33_LONGRUN_REPORT.md`
- `docs/STEP33_SMOKE_PROTOCOL.md`
- `docs/STEP33_REWRITE_REDESIGN_MEMO.md`
- `docs/STEP33_TARGET_SOURCE_BOUNDARY_REDESIGN_MEMO.md`
- `docs/STEP33_SOURCE_REDESIGN_DECISION_MEMO.md`
- `docs/STEP33_EVENT_EDGE_SOURCE_REDESIGN_PROTOTYPE_SPEC.md`
- `docs/STEP33_NEAR_NODE_ROLLOUT_REDESIGN_MEMO.md`
- `docs/STEP33_COMBINED_LINE_STATUS_MEMO.md`
- `docs/STEP33_IMPLEMENTATION_LINE_FINAL_STATUS.md`
- `docs/STEP33_RESTART_DECISION_AND_FIRST_PROTOTYPE_SPEC.md`

Preserve the key checkpoints:

- promoted full learned rewrite family reference:
  - `checkpoints/step33_structured_propagation_v2_nearvel_seed1/best.pt`
  - `checkpoints/step33_structured_propagation_v2_nearvel_seed2/best.pt`
  - `checkpoints/step33_structured_propagation_v2_nearvel_seed3/best.pt`
- retained staged target:
  - `checkpoints/step33_source_held_out_rewrite_seed1/best.pt`
  - `checkpoints/step33_source_held_out_rewrite_seed2/best.pt`
  - `checkpoints/step33_source_held_out_rewrite_seed3/best.pt`
- source-patched current-distance reconnect reference:
  - `checkpoints/step33_source_patched_rollout_distance/best.pt`
- force-frame rollout diagnostic references:
  - `checkpoints/step33_near_node_force_frame_rollout_seed1/best.pt`
  - `checkpoints/step33_near_node_force_frame_rollout_seed2/best.pt`
  - `checkpoints/step33_near_node_force_frame_rollout_seed3/best.pt`
- stable trained combined diagnostic row:
  - `checkpoints/step33_combined_source_edge_force_seed1/best.pt`
  - `checkpoints/step33_combined_source_edge_force_seed2/best.pt`
  - `checkpoints/step33_combined_source_edge_force_seed3/best.pt`

Preserve the rows to cite as best references:

- `spring_neighbor_scope` as the best trivial structured baseline
- promoted `structured_propagation_v2` with near-node-velocity weighting as the best full learned rewrite family reference
- `source_held_out_full` as the retained staged target
- guarded 5-view event-edge source patch as the best source-side diagnostic row
- `source_patched_rollout_distance` as the positive current-distance reconnect reference
- `composed_source_edge_force_rollout` as the retained best diagnostic assembly
- `combined_source_edge_force` as the stable positive trained diagnostic row
- `oracle_scope` as the upper reference

## 7. What Not To Do Next

Do not resume parked Step22-31 micro-lines.

Do not add rendered observation.

Do not run backend transfer.

Do not resume tiny residual, gate, denoise, or event-rule variants.

Do not resume active/contact-only variants.

Do not resume tiny source estimators or kNN-style source estimators.

Do not reconnect event-edge source recovery through the old local noisy feature bundle.

Do not run rollout-only tweaks under the same representation.

Do not run more local `current_distance`-only tweaks.

Do not run more local variants of `source_patched_rollout_distance`.

Do not run more local variants of `combined_source_edge_force`.

Do not start broad Step33 candidate training.

Do not define a first restart prototype under this decision. The chosen decision is to stop Step33 implementation and preserve the line as diagnostic-only.

README should remain unchanged because this is a Step33-local preservation decision and does not change repository-wide defaults or phase entry points.
