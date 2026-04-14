# Sandbox MVP First Result Analysis

## Evidence Base

This memo analyzes the first completed clean single-event sandbox run:

- generated structured graph/event data with oracle event scope and changed-region labels
- trained `local_operator` and `monolithic_baseline` for 20 epochs
- evaluated both checkpoints on the 200-sample test split

The default eval artifact paths requested for this memo were not present in the workspace at memo time. The metric values below use the completed first eval outputs produced from the same checkpoints and test split during the prior eval validation run.

## Did The Plumbing Succeed?

Yes, the sandbox MVP plumbing succeeded.

The data contract is usable. Samples carry structured graph state, sparse `edge_index`, event metadata, event masks, event scope masks, changed masks, next-state targets, and copy baselines. The batching layer supports variable graph sizes and offsets edge indices correctly.

The mask contract also held. Generation reported zero scope-violation rejections across train, val, and test. The inspector showed sampled changed regions fully inside event scope. The local operator copies outside event scope by construction.

Training converged for both models. The local operator best validation loss was `0.0399068`; the monolithic baseline best validation loss was `0.0394593`. Both loss curves decreased from the initial epochs into the same narrow band.

The comparison is fair enough for this first MVP question. Both models use the same structured graph state, event type, event params, data splits, loss weights, optimizer settings, and hidden size. The intended difference is present: the local operator uses oracle event scope and masked copy assembly; the monolithic baseline predicts whole-graph updates without scope-copy construction.

## First-Result Conclusion

The local operator is not better on the first result. It is essentially tied but slightly worse overall.

Overall test metrics:

| model | total_loss | changed_node_error | unchanged_node_preservation_error | changed_edge_error | unchanged_edge_preservation_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_operator | 0.031436 | 0.012842 | 0.005863 | 0.079198 | 0.062302 |
| monolithic_baseline | 0.031368 | 0.011986 | 0.005912 | 0.076359 | 0.059506 |

The gap is small, but the conservative read is clear: the monolithic baseline is slightly better on total loss, changed-node error, changed-edge error, and unchanged-edge preservation. The local operator only wins, narrowly, on unchanged-node preservation.

This does not validate the local-event hypothesis yet. It validates that the sandbox can measure it.

## Event-Type Breakdown

### `node_impulse`

| model | total_loss | changed_node_error | unchanged_node_preservation_error | changed_edge_error | unchanged_edge_preservation_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_operator | 0.000819 | 0.017156 | 0.006490 | 0.004461 | 0.003924 |
| monolithic_baseline | 0.000773 | 0.016859 | 0.009303 | 0.002240 | 0.001462 |

`node_impulse` is easy for both models. The local operator is slightly worse on total loss, changed-node error, changed-edge error, and unchanged-edge preservation. It is better only on unchanged-node preservation.

### `spring_break`

| model | total_loss | changed_node_error | unchanged_node_preservation_error | changed_edge_error | unchanged_edge_preservation_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_operator | 0.073719 | 0.005999 | 0.005530 | 0.202340 | 0.118287 |
| monolithic_baseline | 0.073618 | 0.004256 | 0.004113 | 0.198483 | 0.115171 |

`spring_break` dominates the absolute loss scale and is the main hard case. The local operator is worse on every reported `spring_break` metric. The main local underperformance is therefore not broad failure on node impulses; it is concentrated in spring-break behavior, especially edge-side prediction and preservation.

## Mechanism Gap

The current weakness is mostly edge-side.

Changed-edge error is worse for local overall: `0.079198` vs `0.076359`. Unchanged-edge preservation is also worse: `0.062302` vs `0.059506`. The spring-break edge gap is much larger in absolute terms than the node-side gap: changed-edge error is about `0.202340` local vs `0.198483` monolithic, and unchanged-edge preservation is `0.118287` local vs `0.115171` monolithic.

Node-side behavior is closer. The local operator is worse on changed-node error overall, but only by about `0.000856`. It is slightly better on overall unchanged-node preservation, `0.005863` vs `0.005912`, which is the one visible benefit of the local copy bias.

Preservation is therefore only a narrow relative strength. It helps on unchanged nodes, but not on unchanged edges. Because event scopes are conservative and expanded, many unchanged-but-in-scope edges can still be modified by the local operator. Exact outside-scope copy is not enough to protect all unchanged regions.

The local inductive bias is visible, but weak. The only clear signal is unchanged-node preservation. It is not yet converting oracle locality into better total one-step prediction.

## Next Action

Choose **(b) run exactly one bounded local-operator revision**.

The one revision: add direct event-target mask conditioning to the local operator.

Rationale: the current local operator receives event type, event params, and event scope, but not the direct event target masks. For `spring_break`, the scope contains multiple edges, while only one edge is directly broken. The event params encode endpoint ids, but those ids are weak scalar metadata in a collated graph and do not directly mark the target edge. The observed failure is exactly where this matters: spring-break edge prediction.

What stays fixed:

- same generated data
- same train/val/test split
- same monolithic baseline
- same loss weights
- same hidden size and one-message-pass budget
- same training schedule
- same evaluation metrics
- no proposal module
- no pair-event or commutativity training

What changes:

- local operator input gets `event_node_mask` as one node scalar channel
- local operator input gets `event_edge_mask` as one edge scalar channel
- scope masks remain the only masks used for copy assembly
- no extra message-passing depth, no extra heads, no new event families

Success criteria:

- local operator beats monolithic baseline on overall test `total_loss`
- local operator beats monolithic baseline on `spring_break` `changed_edge_error`
- local operator does not lose its overall unchanged-node preservation advantage

Failure criteria:

- local remains tied or worse on overall `total_loss`
- or local remains worse on `spring_break` `changed_edge_error`
- or the revision improves spring-break edges only by sacrificing preservation enough to erase the local-bias rationale

## What Not To Do Next

Keep the following closed:

- no pair-event generation
- no commutativity evaluation
- no proposal discovery module
- no rendered observations or pixels
- no backend transfer
- no real data
- no hypergraphs
- no LLM integration
- no Step33 reopening
- no Step22-29 local tweaking
- no broad architecture search
- no loss-function sweep
- no official phase/status update

## Decision

This first result does not prove the local-event hypothesis. The monolithic baseline is slightly better overall, and the local operator's weakness is mainly spring-break edge behavior. The sandbox line is still worth exactly one more bounded revision because the failure matches a specific missing input: direct event-target identity inside an expanded event scope. If that revision does not beat the monolithic baseline on overall loss and spring-break changed-edge error, this clean single-event sandbox should not be expanded into pair/commutativity work yet.
