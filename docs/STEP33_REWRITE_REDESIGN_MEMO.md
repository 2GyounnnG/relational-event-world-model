# Step33 Rewrite Redesign Memo

## Status

This is a planning memo only.

Do not treat this as a training plan, backend-transfer plan, rendered-observation plan, or a reason to reopen parked Step22-31 micro-lines.

Step33 remains a controlled synthetic physics-like graph-event benchmark with the same proposal/rewrite spine:

- propose local event scope or changed support
- rewrite local next state
- compare event scope against changed region
- evaluate with exact labels and mechanistic diagnostics

The Step33 benchmark substrate is healthy. The current bottleneck is not benchmark validity. The bottleneck is noisy `spring_retension` rewrite design.

---

## 1. Current Rewrite-Line Status

The Step33 physics-like smoke benchmark is implemented for clean and noisy structured observations. The event family is intentionally small:

- `node_impulse`
- `spring_break`
- `spring_retension`

The benchmark has already shown the properties Step33 was meant to test:

- event scope differs meaningfully from changed region
- oracle scope clearly beats trivial local baselines
- noisy structured observation preserves the benchmark pressure
- proposal-side learned smoke has useful signal

The rewrite line is less healthy.

The noisy `spring_retension` rewrite path has now tested a long enough sequence of small variants to draw a firm conclusion:

- broad oracle rewrite
- preservation-focused rewrite
- event-specific `spring_break` direct edit
- event-specific `spring_retension` direct parameter edit
- event-edge spring-parameter cleanup
- all-changed-edge spring-parameter cleanup
- richer local node-update features
- richer local edge-update features
- clean/noisy edge gating
- denoising-target supervision
- propagation-biased non-event edge update
- stratified noisy val/test rerun

The current tiny learned edge-head family is paused.

Reason: the family does not robustly beat `spring_neighbor_scope` on noisy `spring_retension`, even after correcting the validation/test split artifact. It can solve some direct edits, but it does not provide a reliable rewrite path for propagated noisy non-event edge changes or node rollout.

---

## 2. What Failed And What Succeeded

### Proposal-Side Success

The first learned Step33 smoke showed proposal-side viability. Learned proposal and changed-region classification can beat trivial structured baselines on important scope metrics.

This matters because the Step33 benchmark is not blocked at the proposal/rewrite interface as a whole. The benchmark can support learned models. The current issue is specifically rewrite discipline.

### Broad Rewrite Weakness

The broad oracle rewrite smoke removed proposal uncertainty but still failed to beat `spring_neighbor_scope` on rewrite metrics.

That result isolated the main bottleneck:

- not proposal viability
- not dataset plumbing
- not missing oracle support
- but rewrite quality itself

The broad rewrite head was too loose: it failed changed-region rewrite accuracy and preservation discipline.

### Event-Specific Direct-Edit Successes

The narrow event-specific tests were useful.

`spring_break` showed that a tiny event-specific oracle rewrite path can solve the direct discrete edit: turning the event spring off.

`spring_retension` showed that clean direct parameter updates are largely solvable when the task is narrowed to the event spring's rest length and stiffness.

These successes are important. They show the rewrite family is not completely broken. It can handle direct event-local edits when the target is narrow and well-posed.

### Noisy Non-Event Changed-Edge Bottleneck

The remaining hard case is noisy `spring_retension`, especially non-event changed-edge stiffness.

The issue is not only the oracle event edge. The changed region contains many non-event changed edges, and noisy spring parameters on those edges dominate a large share of changed-region error.

Several cleanup attempts failed to recover enough leverage:

- event-edge-only cleanup was too narrow
- all-changed-edge cleanup helped only slightly
- stronger denoising targets did not materially improve total changed-region error
- propagation-biased non-event edge features did not transfer robustly

### Node Rollout Partial Learnability

Changed-node rollout is not entirely diffuse. Diagnostics showed that velocity change is strongly local:

- event endpoints carry much of the velocity energy
- one-hop nodes capture much of the rest
- two-hop / changed-edge incidence nearly saturates target coverage

Richer local node features materially helped relative to the weakest node updater, but node velocity remained far behind `spring_neighbor_scope`.

So node rollout is partially learnable, but the current local MLP-style updater is not enough to close the rewrite gap.

### Split-Artifact Findings

The propagation-edge smoke originally looked mildly promising on validation but failed on held-out noisy test.

A distribution diagnostic found a val/test mismatch in noisy `spring_retension` propagation structure:

- original validation had `0.00%` unreachable non-event changed edges
- original test had `4.45%`

The stratified diagnostic split corrected this:

- stratified diagnostic validation: `2.47%`
- stratified diagnostic test: `2.02%`

After rerunning the existing edge variants on the stratified split:

- `propagation_edge` still ranked last among learned edge variants on both diagnostic val and diagnostic test
- `denoise oracle_clean` ranked best on diagnostic val
- `edge_gate` ranked best on diagnostic test
- all learned edge variants remained behind `spring_neighbor_scope`

Conclusion: the earlier propagation validation gain was mostly a split artifact.

---

## 3. Why More Tiny Edge Variants Are Low Value

The recent sequence of smoke runs has covered the obvious small local edge-head moves:

- residual cleanup
- clean/noisy gating
- event-edge cleanup
- all-changed-edge cleanup
- stronger denoising target
- local edge context
- propagation-inspired local features
- stratified rerun to rule out a validation artifact

The result is stable enough to stop.

Another tiny residual edge-head variant is unlikely to answer a new question. It would mostly re-test the same failure mode:

- noisy non-event stiffness remains hard
- node velocity remains weak
- total changed-region error does not approach `spring_neighbor_scope`
- validation ranking is not reliable enough to justify model selection within this family

The current issue is not that the last tiny head lacked one more scalar feature. The issue is that the rewrite target and propagation structure are under-specified for the model family being used.

The next Step33 rewrite move should change the problem formulation, not add another residual head.

---

## 4. Option A: Tighter Rewrite Target

A tighter rewrite target means narrowing what the learned rewrite is asked to predict so that each subproblem has a clear physical interpretation and a direct success metric.

In this project, the current broad target asks a small model to do too many things at once:

- denoise observed spring parameters
- apply direct event-edge parameter changes
- infer non-event changed-edge parameter targets
- propagate consequences to node state
- preserve unchanged state
- handle noisy observation corruption

A tighter rewrite target would decompose this into staged targets.

Possible forms:

### Event-Edge-Only Edits

The model predicts only the direct event edit:

- `spring_break`: active flag goes to zero
- `spring_retension`: event spring rest length / stiffness update
- `node_impulse`: direct velocity impulse on the event node

This is the narrowest target and has already shown positive signal.

It is useful as a direct-edit anchor, but it does not solve propagated changed-region recovery by itself.

### Changed-Edge-Only Parameter Edits

The model predicts spring parameter changes only on oracle changed edges, with node state copied.

This isolates edge-side rewrite from node rollout.

For noisy `spring_retension`, this target must distinguish:

- event edge parameter change
- non-event changed-edge observed corruption
- true downstream spring parameter target, if any
- unchanged spring parameter preservation

This could be staged into event-edge and non-event changed-edge subtargets.

### Separate Node Rollout Target

The model predicts changed-node position/velocity deltas separately from spring parameter updates.

This is justified because diagnostics show changed-node velocity is local but not solved by the current tiny updater.

A separate node rollout target should be evaluated with:

- changed-node position MAE
- changed-node velocity MAE
- endpoint vs one-hop vs two-hop breakdown
- preservation outside changed nodes

### Staged Target Decomposition

A staged rewrite target would make the rewrite path explicit:

1. direct event edit
2. spring-parameter cleanup or update on changed edges
3. node rollout update
4. final preservation assembly

Each stage should have its own oracle-support evaluation before any broad learned candidate is attempted.

This may help because current total changed-region error conflates several failure modes. A staged target can reveal whether the blocker is:

- direct edit
- noisy edge cleanup
- non-event edge propagation
- node rollout
- preservation discipline
- target definition itself

Option A is the cheapest way to make the next result interpretable.

---

## 5. Option B: More Structured Propagation Model

A structured propagation model means the rewrite path would include an explicit inductive bias for physical propagation across the local spring graph.

This is not another local edge MLP with more features.

A model counts as structured propagation here only if it explicitly performs limited propagation over the event-local graph. Examples:

- fixed-depth message passing from event edge endpoints
- spring-force-inspired update over changed nodes and changed edges
- event-edge delta injected into neighboring springs/nodes through typed local messages
- separate edge-to-node and node-to-edge propagation phases
- bounded hop count, likely 1-2 hops
- explicit copy/preserve assembly outside oracle support

The smallest plausible version should still be tiny:

- `spring_retension` only
- noisy structured observation only
- oracle changed-node and changed-edge support
- event-edge direct edit frozen or rule-shaped
- one or two propagation rounds over the induced changed subgraph
- separate heads for node velocity delta and non-event spring stiffness/rest correction
- no rendered observation
- no broad Step33 candidate training

This would be meaningfully different from the failed tiny edge heads because it would not ask each edge to infer propagation from static local features alone. It would explicitly move event information through the local spring graph.

However, Option B is more expensive and less diagnostic than Option A. If it fails, it may be unclear whether the failure comes from target ambiguity, propagation design, message-passing capacity, or noisy observation corruption.

---

## 6. Comparison Of A Vs B

### Cost

Option A is cheaper.

It can be implemented as one or two diagnostic eval/train scripts around existing oracle supports and existing Step33 data. It does not require a new model family.

Option B is more expensive because it introduces a new structured propagation bias and therefore a new family of implementation choices.

### Diagnostic Value

Option A is more diagnostic.

It can answer whether the current rewrite target is too broad or ambiguous before adding new propagation machinery.

Option B is scientifically interesting, but it risks mixing target redesign and architecture redesign in one step.

### Likelihood Of Resolving The Current Bottleneck

Option B is more likely to eventually solve the physical propagation problem.

But Option A is more likely to identify the next correct target boundary. The project should not build a structured propagation model until the rewrite target is clearly staged.

### Which Should Happen First

Option A should happen first.

The current evidence says the tiny edge-head family is exhausted, but it does not yet prove that a structured propagation model is the next immediate implementation. The cleaner next step is to tighten the rewrite target and measure which stage still fails.

---

## 7. Recommended Next Move

Pursue tighter rewrite target first.

Do not continue the tiny learned edge-head family.

Do not jump directly to a general structured propagation model yet.

The next phase should define a staged noisy `spring_retension` rewrite target and run one bounded diagnostic probe that separates:

- direct event-edge parameter edit
- non-event changed-edge parameter correction
- changed-node rollout
- unchanged-region preservation

This is the most project-specific next move because Step33's key advantage is exact labels and mechanistic diagnostics. Use that advantage before adding a larger propagation architecture.

---

## 8. Smallest Next Implementation After The Memo

Status update: this staged-target diagnostic has now been run. Its result is recorded in the post-memo update below, and it confirms that the next learned target should be staged changed-edge spring-parameter correction rather than another tiny edge residual variant.

Implement exactly one bounded diagnostic:

### Step33 staged-target rewrite diagnostic for noisy `spring_retension`

Scope:

- noisy `spring_retension` only
- existing Step33 data only
- no rendered observation
- no backend transfer
- no broad candidate training
- oracle event edge, oracle changed edges, and oracle changed nodes available as diagnostic supports

The diagnostic should evaluate staged oracle or rule-shaped assemblies:

1. `event_edge_only`
   - apply only the direct event-edge rest/stiffness update
   - copy everything else

2. `event_edge_plus_changed_edges`
   - event-edge direct edit
   - copy oracle next-state spring parameters on all changed edges
   - copy node state

3. `event_edge_plus_changed_nodes`
   - event-edge direct edit
   - copy oracle next-state node states on changed nodes
   - copy other spring parameters

4. `full_staged_oracle`
   - copy oracle next-state values on event edge, changed edges, and changed nodes
   - copy outside support by construction

5. `current_best_learned_edge_gate`
   - include for comparison, using existing checkpoint only

Metrics:

- event-edge rest/stiffness MAE
- non-event changed-edge rest/stiffness MAE
- changed-node position/velocity MAE
- total changed-region error
- unchanged-region preservation error

Purpose:

- quantify the maximum leverage of each staged target
- decide whether the next learned target should be edge-only, node-only, or combined
- decide whether a structured propagation model is justified after target tightening

This is not a candidate. It is a target-definition diagnostic.

---

## 9. What Not To Do Next

Do not run another tiny residual edge-head variant.

Avoid:

- another local edge MLP with a few more scalar features
- another denoising target variant without target decomposition
- another gating/shrinkage tweak
- another event-edge-only cleanup pass
- another all-changed-edge cleanup pass in the same family
- another propagation-feature-only edge head
- broader Step33 candidate training
- rendered observation
- backend transfer
- Step22-31 micro-line reopening
- raw videos, CLEVRER adoption, real images, hypergraphs, or LLM integration

Also avoid using the original noisy validation split for model selection on `spring_retension` rewrite. Use the stratified diagnostic split or a similarly balanced split for future noisy `spring_retension` diagnostics.

---

## Final Recommendation

Step33 rewrite should be redesigned around a tighter staged target first.

The tiny learned edge-head family is paused. It has produced useful diagnostics but not a viable rewrite direction.

A structured propagation model may still be the right eventual architecture, but it should come after the target boundary is clarified. The next implementation should be a staged-target noisy `spring_retension` diagnostic, not another learned edge residual variant.

## Post-Memo Staged-Target Result

The staged-target noisy `spring_retension` diagnostic is now complete.

Total changed-region error on the noisy test subset:

- `noisy_copy`: `0.1791`
- `event_edge_only`: `0.1633`
- `changed_nodes_only`: `0.1612`
- `event_edge_plus_changed_nodes`: `0.1453`
- `all_changed_edges_only`: `0.0180`
- `event_edge_plus_changed_edges`: `0.0180`
- `full_staged_oracle`: `0.0000`
- `spring_neighbor_scope`: `0.1125`
- `edge_gate_learned`: `0.1550`

This confirms the memo's direction. The dominant leverage is the full changed-edge spring-parameter target, not event-edge direct edit alone and not changed-node rollout alone. The best current learned edge-gated row captures only a small fraction of that leverage and remains behind `spring_neighbor_scope`.

The tiny learned noisy `spring_retension` edge-head family is therefore formally paused. Future rewrite work should target staged changed-edge parameter correction first, with a structured propagation model considered only after that tighter target boundary is tested. Do not escalate `propagation_edge` or another residual/gate/denoise edge-head variant.
