# Step33 Physics-Like Smoke Protocol

## Scope

This protocol records the first Step33 data/eval smoke.

It is not a learned candidate.
It does not reopen parked Step22-31 micro-lines.
It does not use CLEVRER directly, real-world data, raw videos, raw images, hypergraphs, or LLM integration.
It does not run backend transfer.

The goal is to validate the smallest synthetic physics-like benchmark substrate before any learned proposal/rewrite model is introduced.

---

## Smoke World

The first Step33 smoke uses a tiny structured 2D particle-spring-contact world:

- 6-8 particles
- 2D bounded box
- sparse spring graph
- simple proximity/contact hints
- deterministic short rollout after one local event
- clean structured observation first
- noisy structured observation as the first controlled extension

Node features:

- `type_id`
- `x`, `y`
- `vx`, `vy`
- `radius`
- `mass`
- `pinned`

Edge features:

- `spring_active`
- `rest_length`
- `stiffness`
- `current_distance`
- `near_contact`

---

## Event Family

The first smoke event family is intentionally small:

1. `node_impulse`
2. `spring_break`
3. `spring_retension`

Each transition contains exactly one event.

---

## Labels

Each sample includes:

- event type
- event scope node mask
- event scope edge mask
- changed node mask
- changed edge mask
- next node state
- next edge state
- simple physical diagnostics

The smoke should verify that event scope and changed region are often different, because Step33 is meant to stress local physical propagation rather than one-step symbolic edits only.

---

## Baselines

The smoke evaluator reports five non-learned baselines:

- `no_change`
- `copy_state`
- `one_hop_scope`
- `spring_neighbor_scope`
- `oracle_scope`

The local scope baselines use oracle next-state values inside their proposed scope and copy current state outside. This isolates the proposal/rewrite-spine question: does the proposed local region cover the labeled changed region?

---

## Metrics

Scope-side metrics:

- event-scope precision / recall / F1
- changed-node precision / recall / F1
- changed-edge precision / recall / F1

Rewrite-side metrics:

- position MAE
- velocity MAE
- changed-region error
- unchanged-region preservation error

Physical diagnostics:

- spring residual
- contact/proximity violation count

---

## Current Rewrite-Line Status

The Step33 benchmark substrate is healthy enough for controlled diagnostics:

- event scope and changed region consistently diverge in useful ways
- oracle scope clearly leaves headroom above trivial local baselines
- proposal-side learned smoke runs showed useful signal
- noisy structured observation preserves the core benchmark pressure

The first noisy `spring_retension` learned rewrite line is now paused.

The investigated tiny learned edge-head family included:

- clean/noisy-gated residual edge update
- oracle-clean spring-parameter denoising target
- rewrite-residual spring-parameter denoising target
- propagation-biased non-event changed-edge update

The stratified noisy `spring_retension` diagnostic rerun corrected the model-selection split artifact:

- original noisy val/test unreachable non-event changed-edge mismatch: `0.00%` vs `4.45%`
- stratified diagnostic val/test mismatch: `2.47%` vs `2.02%`
- diagnostic val best learned edge variant by total changed-region error: `denoise oracle_clean`
- diagnostic test best learned edge variant by total changed-region error: `edge_gate`
- `propagation_edge` still ranked last among the learned edge variants on both diagnostic val and diagnostic test
- all learned edge variants remained behind `spring_neighbor_scope`

The staged-target noisy `spring_retension` diagnostic then quantified the target leverage:

- `noisy_copy` total changed-region error: `0.1791`
- `event_edge_only`: `0.1633`
- `changed_nodes_only`: `0.1612`
- `event_edge_plus_changed_nodes`: `0.1453`
- `all_changed_edges_only`: `0.0180`
- `event_edge_plus_changed_edges`: `0.0180`
- `full_staged_oracle`: `0.0000`
- reference rows: `spring_neighbor_scope` `0.1125`, `edge_gate_learned` `0.1550`

Follow-up learned rewrite probes tested the tighter changed-edge target and a stronger structured propagation prototype:

- `changed_edge_param_learned` noisy total changed-region error: `0.1570`
- `split_target_edge_learned`: `0.1561`
- `propagation_target_learned`: `0.1591`
- stronger `structured_propagation_learned`: `0.1721`
- reference rows remain `spring_neighbor_scope` `0.1125` and `edge_gate_learned` `0.1550`

Interpretation:

- the earlier propagation validation gain was mostly a split artifact
- the ranking differences among the tiny learned edge heads are smaller than the larger failure to beat the structured trivial baseline
- the dominant leverage is the full changed-edge spring-parameter target, not event-edge direct edit alone and not changed-node rollout alone
- the current learned edge-gated row captures only a small fraction of that staged-target leverage
- the tighter changed-edge target helped only marginally and did not close the gap to `spring_neighbor_scope`
- the first stronger structured propagation prototype is a negative result; it also failed to improve changed-node velocity meaningfully
- the current learned Step33 rewrite family is formally paused
- future Step33 rewrite work needs a more substantial redesign or a narrower target definition before implementation
- small local smoke variants in the current residual/gate/denoise/propagation family should not continue

Reference artifacts:

- `artifacts/step33_spring_retension_stratified_split/summary.json`
- `artifacts/step33_spring_retension_stratified_split/distribution_summary.csv`
- `artifacts/step33_spring_retension_stratified_rerun/summary.json`
- `artifacts/step33_spring_retension_stratified_rerun/rerun_comparison.csv`
- `artifacts/step33_spring_retension_staged_target_diagnostic/summary.json`
- `artifacts/step33_spring_retension_staged_target_diagnostic/staged_target_summary.csv`
- `artifacts/step33_spring_retension_changed_edge_param_smoke_noisy/summary.json`
- `artifacts/step33_spring_retension_split_target_edge_smoke_noisy/summary.json`
- `artifacts/step33_spring_retension_propagation_target_smoke_noisy/summary.json`
- `artifacts/step33_spring_retension_structured_propagation_smoke_noisy/summary.json`

### Current Pause Update

The later stronger-propagation line found a better learned rewrite family, but it is still not candidate-ready:

- promoted `structured_propagation_v2` with near-node-velocity weighting is the best learned Step33 rewrite family so far
- promoted `full_v2` noisy `spring_retension` mean total changed-region error is about `0.1312`
- `spring_neighbor_scope` still wins total changed-region error on noisy `spring_retension` (`0.1125` on the original noisy test reference)
- auxiliary active/contact assembly improved the original noisy test nearly to `spring_neighbor_scope`, but did not survive the stratified noisy diagnostic test as a retained reference
- deterministic event-edge retension rule patches and the existing event-edge cleanup head both regressed the stratified diagnostic result
- oracle event-edge parameters remain a real lever, but current rule-shaped and tiny learned cleanup paths do not capture it
- current small active/contact/event-rule variants are paused
- future rewrite work should be preceded by a redesign around event-edge denoising and near-node rollout together, not another tiny local smoke variant

Reference artifacts for this pause decision:

- `artifacts/step33_spring_retension_contact_margin_sweep/summary.json`
- `artifacts/step33_spring_retension_contact_margin_sweep_stratified_sanity/summary.json`
- `artifacts/step33_contact_gated_stratified_gap_decomposition/summary.json`
- `artifacts/step33_contact_gated_event_rule_patch_stratified/summary.json`
- `artifacts/step33_contact_gated_existing_event_cleanup_patch_stratified/summary.json`
- `docs/CODEX_STEP33_LONGRUN_REPORT.md`

### Event-Edge Source Update

The event-edge source-estimation line is now also paused under the current local noisy feature set:

- `denoise_from_clean_current_source` remains a real diagnostic upper bound:
  - noisy total changed-region error: `0.1232`
  - event-edge stiffness MAE: `0.0085`
  - oracle event-edge patch on support `full_v2`: `0.1230`
- the learned `clean_current_estimator` did not recover that lever:
  - noisy total changed-region error: `0.1299`
  - support `full_v2`: `0.1297`
  - hard `stiffness_factor < 1` bucket remained poor: `0.5089` stiffness MAE versus support `full_v2` at `0.4787`
- the learned estimator improved noisy clean-current source stiffness from `0.4783` to `0.3765`, but this was not enough to improve event-edge target quality or total changed-region error
- clean sanity regressed slightly: clean total `0.0285` versus support `full_v2` at `0.0280`
- `denoise_from_clean_current_source` should be treated as an oracle-source upper bound, not a deployable noisy-observation solution
- future event-edge work requires a stronger source/observation redesign, not another tiny source MLP

The follow-up no-training observability/source-design diagnostic confirmed that pause:

- the clean-current upper bound remains real, but current noisy structured feature bundles do not robustly recover that source
- on noisy `spring_retension` test, the raw observed-current baseline remained stronger than the tested kNN feature-bundle probes:
  - observed-current event-target stiffness MAE: `0.4730`
  - kNN noisy-event-edge-only event-target stiffness MAE: `0.8199`
  - richer kNN local feature bundles: `1.5669` to `1.6809`
- the hard `stiffness_factor < 1` bucket remains the blocker:
  - observed-current event-target stiffness MAE: `0.7168`
  - kNN noisy-event-edge-only event-target stiffness MAE: `0.7957`
  - richer kNN local feature bundles: `1.5400` to `1.8072`
- noisy val showed a partial kNN noisy-event-edge-only improvement, but that signal did not transfer robustly to noisy test
- endpoint state, local structure, changed-support context, and local physics summaries did not provide a stable recoverable clean-current source signal
- event-edge denoising should not be reconnected to near-node rollout under the current representation

Current retained interpretation:

- Step33 benchmark substrate remains healthy
- proposal-side learned signal remains viable
- promoted `structured_propagation_v2` with near-node-velocity weighting remains the current best learned rewrite family
- the current event-edge source-estimation implementation line is paused
- further Step33 rewrite work should not continue via small event-edge residual/source-estimator/kNN-style variants
- future event-edge work requires either a source/observation redesign or a narrower rewrite target

### Retained Staged Rewrite Diagnostic Target

The source-held-out staged rewrite line is now the retained diagnostic target for the current Step33 rewrite work:

- `source_held_out_full` keeps event-edge rest/stiffness recovery held out and report-only
- it trains/evaluates the staged combination of changed-edge auxiliary assembly, non-event changed-edge parameter correction, and node rollout
- seed stability showed that `source_held_out_full` beats promoted `full_v2` mean total changed-region error on original noisy test, stratified noisy validation, and stratified noisy test
- it also beats the reduced `aux_plus_non_event_params` row; dropping rollout is too costly
- promoted `structured_propagation_v2` with near-node-velocity weighting remains the best full learned rewrite family reference
- `source_held_out_full` is not candidate-ready: clean sanity regresses, event-edge source recovery remains unresolved, and `spring_neighbor_scope` still wins noisy total changed-region error
- future Step33 rewrite refinement should start from this source-held-out staged target, not from older tiny edge-head, active/contact, event-rule, or event-edge source-estimator variants

Reference artifacts:

- `artifacts/step33_source_held_out_full_seed_stability/summary.json`
- `artifacts/step33_source_held_out_full_seed_stability/per_seed_summary.csv`
- `artifacts/step33_source_held_out_full_seed_stability/mean_range_summary.csv`
- `checkpoints/step33_source_held_out_rewrite_seed1/best.pt`
- `checkpoints/step33_source_held_out_rewrite_seed2/best.pt`
- `checkpoints/step33_source_held_out_rewrite_seed3/best.pt`

### Current Rewrite Pause

The frozen-edge rollout isolation diagnostic is now complete and pauses the current Step33 implementation line:

- retained `source_held_out_full` edge predictions were frozen and only changed-node rollout was allowed to vary
- rollout-only refinement did not materially improve total changed-region error on original noisy or stratified noisy `spring_retension`
- endpoint velocity often worsened, and earlier apparent rollout gains were mostly coupled to edge-side drift
- promoted `structured_propagation_v2` with near-node-velocity weighting remains the best full learned rewrite family reference
- `source_held_out_full` remains the retained staged rewrite diagnostic target, but it is still not candidate-ready
- event-edge source recovery remains unresolved and should not be reconnected under the current representation
- do not continue rollout weighting, rollout-only isolation, tiny edge/source estimators, active/contact/event-rule variants, or candidate-phase training from this line
- the next Step33 rewrite move should be a source/target redesign planning step, not another small smoke variant

### Source-Patched Rollout/Distance Consolidation

The guarded 5-view source patch was reconnected to a bounded near-node rollout plus `current_distance` refiner. This produced a real but small positive diagnostic row:

- `source_patched_rollout_distance` improves over the matched guarded 5-view patched source-held-out row on original noisy test, stratified noisy validation, stratified noisy test, and clean sanity
- matched-source attribution shows the gain is stable across source draws and is not explained by event-source variance
- the gain is mostly changed-edge `current_distance` assembly, with only minor endpoint and one-hop velocity improvement
- on original noisy test, the row narrowly beats `spring_neighbor_scope`, but mainly because edge-side strengths offset weak rollout
- on stratified noisy test, it still trails `spring_neighbor_scope`; the remaining gap is dominated by endpoint, one-hop, and two-hop+ velocity plus residual `current_distance`
- `source_patched_rollout_distance` is retained as a positive diagnostic reference only, not a candidate-ready family
- do not continue generic rollout weighting or tiny rollout-head tweaks under the same representation
- any further implementation requires a near-node rollout representation/target redesign, not another small loss-weight or residual-head variant

Reference artifacts:

- `checkpoints/step33_source_patched_rollout_distance/best.pt`
- `artifacts/step33_source_patched_rollout_distance_noisy/summary.json`
- `artifacts/step33_source_patched_rollout_distance_decomposition/summary.json`
- `artifacts/step33_source_patched_rollout_distance_decomposition/decomposition_summary.csv`

## Current Combined-Line Pause Status

The bounded source+edge+force rollout line is now complete enough to pause implementation.

- `composed_source_edge_force_rollout` is the best no-training diagnostic assembly: guarded 5-view source, source-patched edge/current_distance assembly, and force-frame rollout
- the composed row is stable across force-frame seeds and beats its parent rows on clean sanity, original noisy test, stratified noisy validation, and stratified noisy test
- `combined_source_edge_force` is a bounded positive trained prototype around the same structure and improves over `source_patched_rollout_distance`
- the trained combined prototype roughly ties or only very slightly improves the retained composed row on noisy totals
- neither row is candidate-ready: stratified noisy test still trails `spring_neighbor_scope` (`0.1131` vs `0.1101`)
- do not open more tiny residual, loss-weight, event-rule, active/contact, or source-estimator variants from this line
- do not escalate Step33 rewrite to broad candidate training from this result
- default next step is implementation pause plus documentation/status consolidation; the only remaining small run would be an explicit seed-stability check of the exact combined row

Reference artifacts:

- `checkpoints/step33_combined_source_edge_force/best.pt`
- `artifacts/step33_combined_source_edge_force/summary.json`
- `artifacts/step33_combined_source_edge_force/combined_source_edge_force_summary.csv`
- `artifacts/step33_source_edge_force_rollout_stability/summary.json`
- `artifacts/step33_source_edge_force_rollout_stability/mean_range_summary.csv`

---

## Commands

Generate the small default smoke splits:

```bash
python data/generate_step33_physics_like_data.py \
  --generate_default_splits \
  --summary_json artifacts/step33_physics_like_smoke_data/summary.json
```

Evaluate the smoke baselines on the test split:

```bash
python train/eval_step33_physics_like_smoke.py \
  --data_path data/graph_event_step33_physics_like_smoke_test.pkl \
  --output_dir artifacts/step33_physics_like_smoke_eval \
  --batch_size 64 \
  --device cpu \
  --num_workers 0
```

Generate the noisy structured smoke splits:

```bash
python data/generate_step33_physics_like_data.py \
  --generate_default_splits \
  --output_prefix graph_event_step33_physics_like_noisy_smoke \
  --observation_variant noisy_structured \
  --summary_json artifacts/step33_physics_like_noisy_smoke_data/summary.json
```

Evaluate noisy aggregate and event-family diagnostics:

```bash
python train/eval_step33_physics_like_smoke.py \
  --data_path data/graph_event_step33_physics_like_noisy_smoke_test.pkl \
  --output_dir artifacts/step33_physics_like_noisy_smoke_eval \
  --batch_size 64 \
  --device cpu \
  --num_workers 0

python train/eval_step33_event_family_diagnostics.py \
  --data_path data/graph_event_step33_physics_like_noisy_smoke_test.pkl \
  --output_dir artifacts/step33_physics_like_noisy_event_family_diagnostics \
  --device cpu \
  --num_workers 0
```

---

## Pass Condition

The first smoke is useful if:

- generated labels are nonempty and inspectable
- event scope differs from changed region on a substantial fraction of samples
- trivial local baselines have signal but leave headroom
- oracle scope clearly improves rewrite metrics
- no learned model is needed to diagnose the benchmark substrate

If these conditions fail, fix the simulator or labels before adding learned candidates.
