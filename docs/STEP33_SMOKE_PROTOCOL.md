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

Interpretation:

- the earlier propagation validation gain was mostly a split artifact
- the ranking differences among the tiny learned edge heads are smaller than the larger failure to beat the structured trivial baseline
- another small residual edge-head variant is not currently justified
- future Step33 rewrite work should require either a tighter rewrite target or a genuinely structured propagation model

Reference artifacts:

- `artifacts/step33_spring_retension_stratified_split/summary.json`
- `artifacts/step33_spring_retension_stratified_split/distribution_summary.csv`
- `artifacts/step33_spring_retension_stratified_rerun/summary.json`
- `artifacts/step33_spring_retension_stratified_rerun/rerun_comparison.csv`

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
