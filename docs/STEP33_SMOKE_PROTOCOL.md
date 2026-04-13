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

Current retained interpretation:

- Step33 benchmark substrate remains healthy
- proposal-side learned signal remains viable
- promoted `structured_propagation_v2` with near-node-velocity weighting remains the current best learned rewrite family
- the current event-edge source-estimation implementation line is paused
- further Step33 rewrite work should not continue via small event-edge residual/source-estimator variants

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
