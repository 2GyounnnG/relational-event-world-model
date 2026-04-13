# Step33 Stronger Propagation Family Memo

## Status

This is a planning memo only.

Do not treat this as a broad Step33 candidate plan, rendered-observation plan, backend-transfer plan, or a reason to reopen parked Step22-31 micro-lines.

Step33 remains a controlled synthetic physics-like graph-event benchmark. The benchmark substrate is healthy, clean/noisy structured smoke data exists, and proposal-side learned smoke has shown useful signal. The current blocker is narrower: noisy `spring_retension` rewrite.

The current learned rewrite family is paused. The project has already tested broad oracle rewrite, preservation-focused rewrite, event-specific heads, edge cleanup, local node and edge features, edge gating, denoising targets, propagation-biased edge updates, stratified noisy reruns, staged-target diagnostics, tighter changed-edge targets, and a first structured propagation prototype. These runs were informative, but they did not produce a viable rewrite family.

## 1. Why A Stronger Propagation Family Is Now Required

Target tightening was the right diagnostic step. It showed that the main oracle leverage in noisy `spring_retension` comes from full changed-edge spring-parameter correction, not from event-edge direct edit alone and not from changed-node rollout alone.

But target tightening was insufficient as a learned rewrite solution. The tighter changed-edge target and split-target refinement only marginally improved the direct edge-head line. The first structured propagation prototype also failed to improve the held-out noisy result. Its noisy total changed-region error was `0.1721`, worse than the existing tiny-family band:

- `edge_gate_learned`: `0.1550`
- `changed_edge_param_learned`: `0.1570`
- `split_target_edge_learned`: `0.1561`
- `propagation_target_learned`: `0.1591`

All of these remain behind `spring_neighbor_scope` at `0.1125`, and far behind the staged changed-edge oracle target. Node velocity also did not improve meaningfully.

The conclusion is not that Step33 is unhealthy. The conclusion is that the current rewrite family is too weak for the physical propagation structure Step33 is exposing.

## 2. What "Genuinely Stronger Propagation Family" Means

A genuinely stronger propagation family must model spring-retension consequences as a staged local physical propagation problem, not as independent edge cleanup.

For Step33, this means:

- event-edge information is represented as a first-class latent cause
- changed nodes and changed edges are both active state carriers
- event-edge effects are propagated through the changed subgraph
- node rollout and edge-parameter correction are coupled through explicit message flow
- outside-support preservation is handled by construction

A few more scalar features, another residual MLP, another gate, another denoising target, or another static local edge heuristic does not count. Those have already been tested enough.

The family is strong enough to be meaningfully different only if it has an explicit recurrent or staged computation over the local changed support. The model should not ask each edge to infer propagation only from fixed one-shot features.

## 3. Minimum Structural Ingredients

A stronger Step33 propagation family needs these pieces:

- Event-edge injection: encode the event edge and its rest/stiffness change as the causal source.
- Oracle support handling for the first prototype: use oracle changed-edge and changed-node masks to remove proposal uncertainty.
- Edge to node propagation: pass event-edge and changed-edge information into changed-node latents.
- Explicit node update: predict changed-node position/velocity deltas, with velocity treated as a first-class target.
- Node to edge propagation: feed changed-node latents back into non-event changed-edge latents.
- Iterated local propagation: use a bounded 1-2 step propagation depth, not a full graph-wide GNN.
- Preservation assembly: copy nodes and edges outside oracle support by construction.
- Stage-level metrics: report event-edge parameter error, non-event changed-edge parameter error, changed-node position/velocity error, total changed-region error, and unchanged preservation.

## 4. What Should Remain Fixed

The next prototype should keep the experimental frame narrow:

- synthetic Step33 benchmark only
- `spring_retension` only
- noisy structured observation first
- clean evaluation as a sanity check, not the optimization target
- oracle changed-edge and changed-node support
- bounded local propagation depth
- no rendered observation
- no backend transfer
- no broad candidate phase
- no Step22-31 micro-line reopening
- no data-scale escalation

The point is to test rewrite structure, not proposal quality, perception, scale, or transfer.

## 5. What Should Change Relative To The Failed Tiny Variants

The failed variants mostly used static local features or shallow edge-specific heads. Even the first structured prototype did not create a strong enough staged physical computation; it behaved like a slightly larger local edge/node updater and did not improve node velocity.

The next family should change the computation itself:

- event-edge latent should be injected into the changed subgraph at every propagation step
- changed-node latents should accumulate incident changed-edge messages
- node delta predictions should be made before final non-event edge correction
- non-event edge correction should condition on both endpoint node latents and event-edge latent
- event-edge direct edit should be supervised separately from non-event changed-edge correction
- losses should be staged, so event-edge, node rollout, and non-event edge correction cannot wash each other out

The model should use propagation as the main inductive bias, not as a decorative layer around the same edge MLP.

## 6. The First Bounded Prototype

Proceed with one bounded prototype: `spring_retension_structured_propagation_v2`.

High-level architecture:

1. Encode the event edge.
   - Inputs: observed event-edge rest/stiffness, event rest/stiffness factors, endpoint node state, noisy flag.
   - Output: event latent and direct event-edge parameter prediction.

2. Initialize changed-node latents.
   - Inputs: node state, event endpoint flag, hop distance, incident changed-edge count, noisy flag.
   - Event latent is injected into event endpoints and one-hop changed nodes.

3. Run edge to node propagation.
   - Changed edges send messages to changed endpoint nodes.
   - Messages include observed spring parameters, event-edge latent, and relative endpoint state.
   - Limit to one or two propagation rounds.

4. Predict changed-node rollout.
   - Output changed-node position/velocity deltas.
   - Copy nodes outside changed-node support.

5. Run node to edge propagation.
   - Non-event changed edges receive endpoint node latents and predicted node-delta context.
   - Predict rest/stiffness correction only on changed edges.
   - Event-edge path remains separately supervised.

6. Assemble final state.
   - Changed nodes receive predicted rollout deltas.
   - Changed edges receive predicted spring parameters.
   - Everything outside oracle support is copied from observation.

Outputs:

- event-edge rest/stiffness prediction
- non-event changed-edge rest/stiffness prediction
- changed-node position/velocity delta
- assembled next node state
- assembled next edge state

This is still a diagnostic prototype. It should stay small, but it must be structurally different from the paused family.

## 7. Success Criteria

A meaningful success on noisy `spring_retension` should meet all of these:

- total changed-region error clearly below the paused tiny-family band, ideally below `0.145`
- non-event changed-edge stiffness MAE clearly below `edge_gate_learned` and `split_target_edge_learned`
- changed-node velocity MAE materially below copied-node behavior, not merely unchanged at about `0.070`
- no severe clean-regression relative to the previous learned rows
- unchanged-region preservation remains copy-stable

A strong success would approach `spring_neighbor_scope` total changed-region error (`0.1125`) while improving at least one mechanistic component that `spring_neighbor_scope` does not solve by learning.

Failure means:

- noisy total changed-region error remains in the `0.155-0.172` band
- node velocity remains near copied-node error
- non-event stiffness does not improve materially
- gains come only from one channel while total rewrite quality remains unchanged

## 8. Risks

The main risk is that the benchmark's noisy structured observation does not expose enough signal to infer the changed-node rollout and non-event changed-edge targets from local state alone.

Other risks:

- oracle changed support may hide proposal/rewrite coupling issues
- staged losses may still compete if not reported separately
- the data may be too small for a structured model even at smoke scale
- a stronger propagation family may become too close to a hand-built simulator
- learned node rollout may remain weak unless the target is narrowed further

If the next prototype still fails to beat the paused family and still leaves node velocity unchanged, Step33 rewrite experiments should pause entirely until the target definition is narrowed or the benchmark is reframed.

## 9. What Not To Do Next

Do not continue with:

- another local edge MLP
- another residual edge head
- another clean/noisy gate
- another denoising-target variant
- another event-edge-only cleanup
- another all-changed-edge static cleanup
- another feature-only propagation edge head
- another small tweak to the current first structured prototype
- broad Step33 candidate training
- rendered observation
- backend transfer
- real data, CLEVRER adoption, raw video, raw images, hypergraphs, or LLM integration

These would mostly repeat known failure modes.

## 10. Decision Recommendation

Proceed to one stronger propagation prototype.

Do not stop Step33 rewrite entirely yet, because the benchmark remains healthy, proposal-side viability exists, and the staged diagnostics show a real oracle rewrite target. But the next implementation must be structurally different from the paused family.

The recommended next implementation is a single bounded `spring_retension_structured_propagation_v2` diagnostic with oracle changed-node and changed-edge support, explicit event-edge injection, staged edge to node and node to edge propagation, supervised node rollout, supervised changed-edge spring-parameter correction, and copy-by-construction outside support.

If that prototype also fails to move noisy total changed-region error below the paused-family band and fails to improve changed-node velocity, pause Step33 rewrite experiments entirely and return to target/benchmark redesign rather than continuing model variants.
