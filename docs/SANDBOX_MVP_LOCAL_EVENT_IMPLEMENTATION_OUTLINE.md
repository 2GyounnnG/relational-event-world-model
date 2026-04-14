# Sandbox MVP Local Event Implementation Outline

## 1. MVP Objective

The MVP tests one narrow version of the repo's core hypothesis: when world changes are sparse, local, and event-driven, a model that applies a local event-conditioned rewrite inside an event mask should learn one-step dynamics and event composition more cleanly than a monolithic next-state predictor over the whole graph. This sandbox is trying to validate the mechanics of local proposal/rewrite under oracle masks, local changed-region labels, and independent-event composition in a tiny structured graph world. It is explicitly not trying to validate proposal discovery, noisy observation robustness, rendered perception, backend transfer, broad candidate readiness, or any official phase transition.

## 2. Sandbox World Choice

Use one tiny Step33-style 2D particle-spring world.

World definition:

- `3-5` particles in a bounded 2D box.
- Sparse undirected spring graph.
- Structured graph state only.
- Short deterministic physics step after exactly one event.
- Optional small process noise may be held out until after the clean MVP works.
- No rendered observation.
- No raw image input.
- No backend transfer.

Node state:

- 2D position
- 2D velocity
- mass
- radius
- pinned flag

Edge state:

- spring active flag
- rest length
- stiffness
- current distance

This is the minimum useful world because it has both node-local and edge-local event types, visible local propagation, explicit graph structure, and natural event masks. It is small enough that oracle support, changed masks, commutativity labels, and sanity visualizations can be inspected directly.

Pure collision-only simulation is not preferred as the first MVP substrate. Collision-only worlds make local event scope and changed-region attribution more ambiguous, especially for independent-event-pair construction. A particle-spring graph gives explicit edges, explicit edge-local events, and a cleaner proposal/rewrite contract.

## 3. Event Family

Use exactly two event types.

### `node_impulse`

Definition:

- Select one unpinned node.
- Apply a small impulse vector to that node's velocity at the current state.
- Roll the system forward for one short physics step.

Local state change:

- Direct event edit: selected node velocity changes immediately.
- Propagated change: adjacent spring-connected nodes may change position/velocity after rollout.
- Edge features may change only through derived `current_distance`.

Event scope:

- Event node.
- Incident edges.
- One-hop neighbor nodes.

Changed region:

- Nodes whose next position/velocity differs from copied current state above tolerance.
- Edges whose next edge features differ from copied current edge state above tolerance, especially `current_distance`.

### `spring_break`

Definition:

- Select one active spring edge.
- Set its `spring_active` flag to `0`.
- Roll the system forward for one short physics step.

Local state change:

- Direct event edit: selected edge active flag changes.
- Propagated change: endpoint node velocities/positions and nearby connected nodes may change after rollout.
- Other edge `current_distance` values may change as nodes move.

Event scope:

- Event edge.
- Endpoint nodes.
- Edges incident to endpoint nodes.
- One-hop neighbor nodes adjacent through those incident edges.

Changed region:

- Event edge if active flag or derived features change.
- Nodes whose next position/velocity differs from copied current state above tolerance.
- Edges whose next edge features differ from copied current edge state above tolerance.

## 4. Data Contract

One sample must be a structured graph/event transition, not a generic tensor sequence.

Required fields:

- `node_features_t`: `[N, F_node]`
  - `x`
  - `y`
  - `vx`
  - `vy`
  - `mass`
  - `radius`
  - `pinned`
- `edge_index`: `[2, E]`
  - directed or paired-directed convention must be fixed globally
- `edge_features_t`: `[E, F_edge]`
  - `spring_active`
  - `rest_length`
  - `stiffness`
  - `current_distance`
- `event_type`
  - enum: `node_impulse` or `spring_break`
- `event_node_mask`: `[N]`
- `event_edge_mask`: `[E]`
- `event_params`
  - for `node_impulse`: impulse vector
  - for `spring_break`: selected edge id or equivalent event-edge mask
- `event_scope_node_mask`: `[N]`
- `event_scope_edge_mask`: `[E]`
- `changed_node_mask`: `[N]`
- `changed_edge_mask`: `[E]`
- `node_features_next`: `[N, F_node]`
- `edge_features_next`: `[E, F_edge]`
- `copy_node_features_next`
  - copied current node state baseline for explicit delta checks
- `copy_edge_features_next`
  - copied current edge state baseline for explicit delta checks

Paired-event metadata for commutativity samples:

- `event_a`
- `event_b`
- `a_scope_node_mask`
- `a_scope_edge_mask`
- `b_scope_node_mask`
- `b_scope_edge_mask`
- `a_changed_node_mask`
- `a_changed_edge_mask`
- `b_changed_node_mask`
- `b_changed_edge_mask`
- `independent_pair_flag`
- `ab_node_features_next`
- `ab_edge_features_next`
- `ba_node_features_next`
- `ba_edge_features_next`

Explicit rejection:

- Do not design the dataset as only `[B, T, N, F]`.
- A plain sequence tensor erases event type, event scope, changed region, and independent-pair structure, which are the actual MVP target.

## 5. Oracle Masking Policy

MVP v1 uses oracle event masks and oracle event scope masks.

Proposal discovery is deferred.

Reason:

- The first MVP should test whether local rewrite is mechanically useful when the local event region is known.
- Proposal discovery would mix two questions: finding the event region and rewriting the local region.
- The repo's core hypothesis has two stages, but the sandbox v1 should isolate rewrite and composition before adding proposal prediction.

The model input may receive:

- current node/edge graph state
- event type
- event parameters
- oracle event node/edge masks
- oracle event scope node/edge masks

It should not receive:

- changed masks as inputs
- next-state features as inputs
- paired AB/BA outputs as inputs

Changed masks are labels and diagnostics only.

## 6. Independent Event Pair Generation

Generate event pair `(a, b)` from the same initial graph state.

Operational independence:

- event scopes do not overlap in nodes
- event scopes do not overlap in edges
- event changed regions under single-event rollout do not overlap in nodes
- event changed regions under single-event rollout do not overlap in edges
- neither event invalidates the other's direct target

Valid pair examples:

- two `node_impulse` events on nodes separated by at least two graph hops, with disjoint one-hop scopes and disjoint changed masks
- `node_impulse` and `spring_break` where the impulse node is not an endpoint or one-hop neighbor of the spring-break edge, and single-event changed masks are disjoint
- two `spring_break` events on well-separated active edges with disjoint endpoint neighborhoods and disjoint changed masks

Excluded pairs:

- events sharing any event node or event edge
- events with overlapping event scope masks
- events with overlapping changed masks
- two `spring_break` events targeting the same edge
- any event that makes the other event invalid when applied second
- pairs where AB and BA physical rollout differs above a tolerance because of hidden interaction

For commutativity labels:

- apply `a` then `b` to produce AB target
- apply `b` then `a` to produce BA target
- keep only pairs where AB and BA oracle rollouts match within tolerance
- store the tolerance and the measured AB/BA discrepancy

## 7. Model Plan

Use one minimal local masked event operator.

Input:

- `node_features_t`
- `edge_index`
- `edge_features_t`
- `event_type`
- `event_params`
- `event_scope_node_mask`
- `event_scope_edge_mask`

Output:

- predicted node deltas for nodes inside `event_scope_node_mask`
- predicted edge deltas for edges inside `event_scope_edge_mask`
- copied current state outside oracle scope by construction

Operator:

- shared node encoder MLP
- shared edge encoder MLP
- small event embedding
- one local message-passing/update step restricted to event scope
- node delta head
- edge delta head
- masked assembly that copies outside scope

This should be a simple masked local graph update operator, not architecture shopping. Avoid GNN stack escalation until the first data contract, loss curve, and composition test are working.

Monolithic comparison baseline:

- same current graph state and event metadata
- no oracle local copy assembly
- predicts next node and edge state for the whole graph
- comparable parameter count where practical
- no event-scope masking advantage

The baseline is included to test whether locality helps, not to find the best possible monolithic architecture.

## 8. Loss Plan

### Loss A: Next-State Prediction

Use supervised one-step next-state loss.

Components:

- node position MAE or MSE
- node velocity MAE or MSE
- edge active binary loss for `spring_active`
- edge regression loss for `rest_length`, `stiffness`, and `current_distance`

Masking:

- primary loss inside changed masks
- secondary preservation loss outside changed masks
- optional scope loss inside event scope but outside changed region to prevent drift

For the local operator:

- outside event scope is copied by construction
- preservation loss still reported for sanity

For the monolithic baseline:

- preservation loss is a real metric and should be included in selection/reporting

### Loss B: Commutativity Consistency

Loss B is allowed only for valid independent event pairs.

Allowed when:

- `independent_pair_flag = true`
- event scopes are disjoint
- changed masks are disjoint
- both orders are valid
- oracle AB and BA targets match within tolerance

Not allowed when:

- event scopes overlap
- changed masks overlap
- one event invalidates the other
- oracle AB and BA differ above tolerance

Comparison:

- apply learned operator for A then B to produce predicted AB
- apply learned operator for B then A to produce predicted BA
- compare predicted AB and predicted BA on final node and edge states
- optionally compare both against the shared oracle AB/BA target

Loss terms:

- `L_comm = ||pred_AB - pred_BA||` on node and edge states
- `L_ab_target = ||pred_AB - target_AB||`
- `L_ba_target = ||pred_BA - target_BA||`

Keep `L_comm` small and diagnostic at first. Do not let commutativity loss hide poor one-step prediction.

## 9. Evaluation Plan

One-step prediction metrics:

- total node position MAE
- total node velocity MAE
- changed-node position MAE
- changed-node velocity MAE
- unchanged-node preservation error
- edge active accuracy / F1
- changed-edge regression MAE
- unchanged-edge preservation error
- total changed-region error

Local-vs-monolithic comparison:

- compare local masked operator against monolithic baseline on the same train/val/test splits
- report changed-region accuracy and unchanged-region preservation separately
- require local operator to improve preservation without losing changed-region quality

Composition / OOD pair test:

- train primarily on single events
- evaluate valid independent event pairs
- compare predicted AB vs predicted BA
- compare predicted AB/BA against oracle final state
- include held-out graph layouts and held-out event-pair distances if the initial small generator supports it

Visualizations:

- graph before event
- event scope
- changed region
- predicted next state
- target next state
- AB vs BA overlay for independent pairs

Unit tests:

- generated graph has valid node/edge shapes
- event masks are nonempty and type-correct
- changed masks match next-state differences above tolerance
- outside copied region is unchanged for oracle copy baseline
- independent pair generator excludes overlapping scope pairs
- AB/BA oracle equality tolerance is enforced
- local operator assembly copies outside event scope

## 10. File-By-File Implementation Plan

### `data/generate_sandbox_local_event_mvp.py`

Purpose:

- Generate tiny 2D particle-spring graph/event transition data.

Inputs:

- number of samples
- train/val/test split sizes
- graph size range `3-5`
- event type mix
- random seed
- output paths

Outputs:

- `data/sandbox_local_event_mvp_train.pkl`
- `data/sandbox_local_event_mvp_val.pkl`
- `data/sandbox_local_event_mvp_test.pkl`
- optional `artifacts/sandbox_local_event_mvp_data/summary.json`

Dependencies:

- existing Python / PyTorch / pickle conventions if available
- no Step33 implementation reopening

Why needed:

- The MVP needs a clean, tiny, inspectable graph-event substrate with labels and masks.

### `data/sandbox_local_event_dataset.py`

Purpose:

- Load MVP samples and batch graph/event fields consistently.

Inputs:

- generated pickle path

Outputs:

- batch dictionaries with node features, edge features, masks, metadata, and targets

Dependencies:

- minimal torch dataset utilities

Why needed:

- Keeps train/eval code from hardcoding pickle structure.

### `models/sandbox_local_event_operator.py`

Purpose:

- Define the minimal masked local event operator.

Inputs:

- current graph state
- event metadata
- oracle event scope masks

Outputs:

- assembled next graph state
- optional local deltas for debugging

Dependencies:

- PyTorch

Why needed:

- This is the core model for the local proposal/rewrite hypothesis under oracle masks.

### `models/sandbox_monolithic_baseline.py`

Purpose:

- Define the whole-graph monolithic comparison baseline.

Inputs:

- current graph state
- event metadata

Outputs:

- whole-graph next state prediction

Dependencies:

- PyTorch

Why needed:

- The MVP needs a direct comparison showing whether masked local rewrite helps.

### `train/train_sandbox_local_event_mvp.py`

Purpose:

- Train local operator and monolithic baseline under the same data regime.

Inputs:

- train/val paths
- model type: `local_operator` or `monolithic`
- loss weights
- output checkpoint path

Outputs:

- checkpoint
- training history
- validation summary

Dependencies:

- data loader
- model files

Why needed:

- Provides the first loss curves and one-step prediction comparison.

### `train/eval_sandbox_local_event_mvp.py`

Purpose:

- Evaluate one-step prediction and preservation metrics.

Inputs:

- test path
- checkpoint path
- model type

Outputs:

- metrics JSON
- compact CSV

Dependencies:

- data loader
- model files

Why needed:

- Separates evaluation from training and keeps the comparison reproducible.

### `train/eval_sandbox_commutativity.py`

Purpose:

- Evaluate AB vs BA consistency on valid independent event pairs.

Inputs:

- paired-event test path
- checkpoint path
- model type

Outputs:

- AB target error
- BA target error
- AB-vs-BA prediction discrepancy
- invalid-pair rejection summary

Dependencies:

- paired-event data fields
- local operator assembly

Why needed:

- Commutativity is the minimum composition test for local event operators.

### `notebooks/sandbox_local_event_visualization.ipynb`

Purpose:

- Visualize generated graphs, event masks, changed masks, predictions, and AB/BA overlays.

Inputs:

- generated sample files
- optional evaluation outputs

Outputs:

- inspection plots

Dependencies:

- matplotlib

Why needed:

- The MVP is small enough that visual inspection should catch mask, pair, and rollout errors early.

## 11. Stage Gates

### Gate 1: Outline Approved

Evidence required:

- one chosen substrate only
- event family fixed to `node_impulse` and `spring_break`
- oracle mask policy accepted
- no proposal discovery in v1
- no rendered observation, backend transfer, Step33 reopening, or Step22-29 reopening

### Gate 2: Data Generator Approved

Evidence required:

- generated samples have valid graph shapes
- node/edge state fields match the contract
- event masks and scope masks are correct
- changed masks match next-state diffs
- independent-pair generator rejects overlapping pairs
- AB/BA oracle equality check is enforced
- small visual sample confirms masks and changed regions are sensible

### Gate 3: Local Operator Approved

Evidence required:

- forward pass runs on one batch
- outside-scope copy assembly is exact
- outputs match node/edge target shapes
- local deltas are only applied inside event scope
- unit tests cover `node_impulse` and `spring_break`

### Gate 4: Train Loop Approved

Evidence required:

- local operator loss decreases on train and validation
- monolithic baseline trains under the same split
- no NaNs or shape-dependent failures
- local operator does not trivially copy everything
- unchanged-region preservation is reported separately from changed-region error

### Gate 5: Commutativity Test / Visualization Approved

Evidence required:

- valid independent pairs exist in the test split
- invalid pairs are excluded with clear counts
- AB and BA oracle targets match within tolerance for included pairs
- local operator AB-vs-BA discrepancy is lower than monolithic baseline or meaningfully small in absolute terms
- visualization shows at least one `node_impulse` pair, one `spring_break` pair, and one mixed pair

## 12. Risks And Failure Modes

Ordinary sequence-modeling drift:

- The model may learn generic next-state smoothing rather than local event rewrite.
- Mitigation: require changed-region and unchanged-region metrics separately.

Event masks too weak:

- If event scopes exclude needed one-hop propagation, local operator may fail unfairly.
- Mitigation: define event scopes explicitly and validate against changed masks.

Invalid independent-event pairs:

- Pair generation may include hidden interacting events.
- Mitigation: require disjoint scopes, disjoint changed masks, and AB/BA oracle tolerance.

Loss B applied to non-commuting events:

- This would teach false consistency.
- Mitigation: apply Loss B only when `independent_pair_flag` is true and oracle AB/BA agrees.

Over-engineering before first loss curve:

- Architecture shopping could obscure the MVP.
- Mitigation: keep the first operator small and masked.

Mismatch with repo's real local-event question:

- A generic trajectory predictor would not test proposal/rewrite.
- Mitigation: preserve explicit event metadata, masks, local assembly, changed-region labels, and monolithic comparison.

## 13. Execution Order

Recommended order:

1. Approve this outline.
2. Implement the data generator for clean single-event samples.
3. Add mask/changing-region unit tests.
4. Add independent-pair generation and AB/BA oracle validation.
5. Implement the dataset loader.
6. Implement the masked local event operator.
7. Implement the monolithic baseline.
8. Train both models on one-step prediction only.
9. Evaluate one-step prediction and preservation metrics.
10. Add commutativity evaluation for valid independent pairs.
11. Add visualization notebook.
12. Decide whether the sandbox MVP supports local event-centric rewrite better than monolithic prediction.

## 14. What Not To Do

Do not use pixels.

Do not add rendered observations.

Do not run backend transfer.

Do not add automatic proposal discovery in v1.

Do not use real data.

Do not use raw images.

Do not use hypergraphs.

Do not use LLM integration.

Do not reopen Step33 implementation.

Do not reopen Step22-29 local noisy interaction tweaking.

Do not start with generic abstraction-first engineering.

Do not replace the repo's local-event question with generic trajectory prediction.

Do not design the dataset as only `[B, T, N, F]`.

Do not train on commutativity pairs before single-event one-step prediction works.

Do not broaden beyond the two-event family in MVP v1.
