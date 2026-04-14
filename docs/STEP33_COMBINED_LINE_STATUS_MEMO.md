# Step33 Combined Line Status Memo

## Current Best Diagnostic Rows

Step33 benchmark health and proposal-side learned signal remain intact. The current rewrite-side status is narrower:

- promoted `structured_propagation_v2` with near-node-velocity weighting remains the best full learned rewrite family reference
- `source_held_out_full` remains the retained staged diagnostic target
- guarded 5-view event-edge source patch remains the retained source-side redesign reference
- `source_patched_rollout_distance` remains the retained positive reconnect reference for changed-edge `current_distance`
- `composed_source_edge_force_rollout` is the best no-training diagnostic assembly
- `combined_source_edge_force` is the latest bounded trained prototype around that same assembly

## Why The Combined Line Matters

The combined line is meaningful because it is not another free-form tiny residual variant. It combines two independently diagnosed, stable Step33 levers:

- source-patched changed-edge `current_distance` assembly
- force-frame near-node rollout representation

The no-training composed row showed that these two levers add constructively. The seed stability check showed that the composed row beats its parent rows across clean sanity, original noisy test, stratified noisy validation, and stratified noisy test.

The bounded trained `combined_source_edge_force` prototype then confirmed that the same structure can be trained end to end without losing the established edge-side strengths. It improves over `source_patched_rollout_distance` and roughly ties or slightly improves the retained composed row on noisy totals.

## Why It Is Not Candidate-Ready

The remaining blocker is still visible on the harder split. On stratified noisy test:

- `combined_source_edge_force`: `0.1131`
- retained composed row: `0.1131`
- `source_patched_rollout_distance`: `0.1138`
- `spring_neighbor_scope`: `0.1101`

The combined prototype improves `current_distance` and some two-hop velocity behavior, but endpoint and one-hop rollout are not solved. The row is stronger than the older learned variants, but it still depends on oracle support, guarded multiview source diagnostics, and a staged benchmark-specific assembly. It should remain a diagnostic row, not a candidate phase entry.

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

The combined prototype was the one bounded implementation justified by the stable no-training composition. Its gain is real but small. Another small residual, loss weight, event-rule, or source-estimator variant would not address the persistent stratified-test rollout gap.

## Optional Remaining Small Run

The only small implementation run that remains defensible is a seed-stability check of the exact `combined_source_edge_force` row.

That check would answer whether the tiny trained gain over the composed row is stable. It should not alter the architecture, source protocol, event family, benchmark, support assumptions, or loss surface beyond seed repetition.

If that stability check is not explicitly requested, the implementation line should pause here.

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

Do not escalate to broad candidate training. Do not open more tiny variants. Future Step33 rewrite work should resume only with a clearer redesign of near-node rollout or a deliberate seed-stability check of the exact combined row.
