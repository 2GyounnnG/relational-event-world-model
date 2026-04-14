# Step33 Combined Line Status Memo

## Current Best Diagnostic Rows

Step33 benchmark health and proposal-side learned signal remain intact. The current rewrite-side status is narrower:

- promoted `structured_propagation_v2` with near-node-velocity weighting remains the best full learned rewrite family reference
- `source_held_out_full` remains the retained staged diagnostic target
- guarded 5-view event-edge source patch remains the retained source-side redesign reference
- `source_patched_rollout_distance` remains the retained positive reconnect reference for changed-edge `current_distance`
- `composed_source_edge_force_rollout` is the retained best diagnostic assembly reference
- `combined_source_edge_force` is a stable positive trained diagnostic row around that same assembly
- neither row is candidate-ready, and the implementation line is now paused

## Why The Combined Line Matters

The combined line is meaningful because it is not another free-form tiny residual variant. It combines two independently diagnosed, stable Step33 levers:

- source-patched changed-edge `current_distance` assembly
- force-frame near-node rollout representation

The no-training composed row showed that these two levers add constructively. The seed stability check showed that the composed row beats its parent rows across clean sanity, original noisy test, stratified noisy validation, and stratified noisy test.

The bounded trained `combined_source_edge_force` prototype then confirmed that the same structure can be trained end to end without losing the established edge-side strengths. It improves over `source_patched_rollout_distance`.

The exact three-seed stability check confirms that this trained row is real and stable, not seed noise. It also resolves the previous ambiguity: the trained combined row does **not** beat the retained no-training composed row. The composed row remains the best diagnostic assembly reference.

## Why It Is Not Candidate-Ready

The remaining blocker is still visible on the harder split. On stratified noisy test:

- `combined_source_edge_force` three-seed mean: `0.1134`
- retained composed row: `0.1131`
- `source_patched_rollout_distance`: `0.1138`
- `spring_neighbor_scope`: `0.1101`

The combined prototype improves over its source-patched parent, but endpoint and one-hop rollout are not solved and the stratified-test gap to `spring_neighbor_scope` remains stable. The row is stronger than the older learned variants, but it still depends on oracle support, guarded multiview source diagnostics, and a staged benchmark-specific assembly. It should remain a diagnostic row, not a candidate phase entry.

## Why More Small Variants Are Low Value

This session has already exhausted the obvious small implementation knobs:

- tiny residual and gate heads
- denoise-target edge heads
- propagation-biased edge heads
- hop0/1 loss weighting
- active/contact/event-rule patches
- single-view event-edge source estimators
- rollout-only weighting and frozen-edge rollout isolation
- changed-edge-only target narrowing

The combined prototype was the one bounded implementation justified by the stable no-training composition. Its gain over `source_patched_rollout_distance` is real but small, and it does not beat the composed row. Another small residual, loss weight, event-rule, source-estimator, rollout-head, or `current_distance` tweak would not address the persistent stratified-test rollout gap.

## Final Seed-Stability Check

The only small implementation run that remained defensible was a seed-stability check of the exact `combined_source_edge_force` row. That check is complete.

Results:

- clean sanity: combined `0.0281` mean / `0.0005` range, composed `0.0277`
- original noisy test: combined `0.1105` mean / `0.0002` range, composed `0.1102`
- stratified noisy validation: combined `0.1137` mean / `0.0002` range, composed `0.1135`
- stratified noisy test: combined `0.1134` mean / `0.0004` range, composed `0.1131`, `spring_neighbor_scope` `0.1101`

The result is stable, but it does not promote the trained combined row over the composed diagnostic assembly. It also does not close the harder stratified-test gap.

## Default Recommendation

Pause Step33 rewrite implementation at the current combined-line result.

Preserve these rows as references:

- promoted `structured_propagation_v2` near-node-velocity weighted full-family reference
- retained `source_held_out_full` staged target
- `source_patched_rollout_distance`
- `composed_source_edge_force_rollout`
- `combined_source_edge_force`
- `spring_neighbor_scope`
- `oracle_scope`

Do not escalate to broad candidate training. Do not open more tiny variants. Future Step33 rewrite work should resume only from a new redesign decision, not from another local residual, loss-weight, source, rollout, or `current_distance` variant on this line.
