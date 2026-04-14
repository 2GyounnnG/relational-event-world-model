# Sandbox MVP Consolidated Status

## Purpose

The sandbox MVP tested one narrow hypothesis: given structured graph state, explicit event metadata, and oracle locality masks, a local event-conditioned rewrite operator should be able to learn sparse one-step changes and compose independent events more cleanly than a monolithic whole-graph predictor.

This was never intended to validate proposal discovery, noisy observations, rendered inputs, backend transfer, real data, broad Step33 readiness, or any official phase transition.

## What Was Implemented

The sandbox line now contains:

- a clean single-event particle-spring generator
- a dataset/collate layer for variable-size sparse graphs
- a masked `SandboxLocalEventOperator`
- a `SandboxMonolithicBaseline`
- single-event training and evaluation
- independent pair generation for composition-only checks
- AB/BA commutativity evaluation
- result and robustness memos

The local operator uses oracle event scope for copy-by-construction outside scope. The final revised local operator also receives direct event-target masks, `event_node_mask` and `event_edge_mask`, as conditioning channels only.

## First Local-vs-Monolithic Failure

The first local operator did not beat the monolithic baseline.

Initial single-event test result:

| model | total_loss | changed_node_error | unchanged_node_preservation_error | changed_edge_error | unchanged_edge_preservation_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_operator | 0.031436 | 0.012842 | 0.005863 | 0.079198 | 0.062302 |
| monolithic_baseline | 0.031368 | 0.011986 | 0.005912 | 0.076359 | 0.059506 |

The conservative conclusion was that the first local model was essentially tied but slightly worse overall. The visible gap was mainly spring-break edge behavior: the local model had event scope but not direct target identity, so it had to infer which in-scope edge was the actual broken spring.

## Bounded Revision

Exactly one bounded revision was made: add direct event-target mask conditioning to the local operator.

What changed:

- append `event_node_mask` as one node-side scalar input channel
- append `event_edge_mask` as one edge-side scalar input channel

What stayed fixed:

- same generated single-event data
- same loss weights
- same one-message-pass budget
- same hidden size default
- same event embedding
- same output contract
- same copy-by-construction semantics using only event scope masks
- no proposal module
- no pair-event training
- no commutativity loss

This revision changed the single-event result qualitatively.

Revised local single-event test result:

| model | total_loss | changed_node_error | unchanged_node_preservation_error | changed_edge_error | unchanged_edge_preservation_error |
| --- | ---: | ---: | ---: | ---: | ---: |
| local_operator_eventmask | 0.000013 | 0.001795 | 0.001596 | 0.001971 | 0.001204 |
| monolithic_baseline | 0.031368 | 0.011986 | 0.005912 | 0.076359 | 0.059506 |

Event-type breakdown for the revised local model:

| event_type | samples | total_loss | changed_node_error | changed_edge_error |
| --- | ---: | ---: | ---: | ---: |
| node_impulse | 117 | 0.000014 | 0.001863 | 0.002014 |
| spring_break | 83 | 0.000013 | 0.001686 | 0.001900 |

## Final Single-Event Conclusion

The final single-event result is positive for the revised local operator.

The important caveat is that the success depends on direct event-target mask conditioning. Oracle event scope alone was not enough. The local architecture became effective once it knew both the scope to rewrite and the direct event target inside that scope.

This establishes a useful feasibility result for local event-conditioned rewriting under oracle event target and scope metadata. It does not establish proposal discovery.

## Composition-Only Result

Independent event-pair evaluation was added as a separate composition-only subtask. It used the revised local checkpoint and the existing monolithic baseline checkpoint without retraining.

Overall test composition result:

| model | pred AB/BA mismatch RMSE | AB target RMSE | BA target RMSE |
| --- | ---: | ---: | ---: |
| local_operator_eventmask | 0.000000 | 0.013881 | 0.013881 |
| monolithic_baseline | 0.000343 | 0.056710 | 0.056712 |

Robustness check across three pair-generation seeds:

| seed | model | pred AB/BA mismatch RMSE | AB target RMSE | BA target RMSE |
| ---: | --- | ---: | ---: | ---: |
| 0 | local_operator_eventmask | 0.000000 | 0.014308 | 0.014308 |
| 0 | monolithic_baseline | 0.000359 | 0.056584 | 0.056585 |
| 1 | local_operator_eventmask | 0.000000 | 0.014343 | 0.014343 |
| 1 | monolithic_baseline | 0.000370 | 0.059694 | 0.059709 |
| 2 | local_operator_eventmask | 0.000000 | 0.013881 | 0.013881 |
| 2 | monolithic_baseline | 0.000343 | 0.056710 | 0.056712 |

The composition advantage is stable across these seeds. The local operator has effectively exact AB/BA prediction consistency at evaluator precision, and substantially lower AB/BA target error.

## Composition Scope Caveat

The composition result must not be conflated with the original 3-5 node single-event sandbox.

The independent pair generator uses 8-10 node chain graphs because the conservative expanded event scope policy makes disjoint independent event pairs infeasible in 3-5 node graphs. This is a deliberate composition-only variant, not the same distribution as the original single-event dataset.

This does not invalidate the composition result, but it bounds the claim: the result says the revised local operator composes well when independent oracle-scoped events are feasible and explicitly constructed on larger chain graphs.

## What Is Established

The sandbox line now establishes:

- the structured graph/event data contract works
- oracle scope copy-by-construction is mechanically usable
- direct event-target identity is necessary for this local operator to beat monolithic on single-event prediction
- after adding direct event-target masks, the local operator is clearly better than the monolithic baseline on the clean single-event sandbox
- the revised local operator composes independent oracle-scoped events much more cleanly than the monolithic baseline in the 8-10 node composition-only variant
- the composition advantage is stable across three pair-generation seeds

The sandbox line does not establish:

- proposal discovery
- noisy observation robustness
- rendered perception
- backend transfer
- real-data readiness
- Step33 readiness
- need for training-time commutativity loss

## Decision

Preserve this sandbox line as a positive feasibility result.

Do not expand to training-time commutativity loss now. The current local operator already shows strong composition behavior under oracle masks, so a commutativity loss would add complexity before there is evidence that composition consistency is the bottleneck.

Do not keep iterating the sandbox further now. The line has answered the intended bounded question: local event-conditioned rewrite can work and compose under oracle event target and scope metadata. Further sandbox iteration risks turning a preserved feasibility result into architecture shopping.

## Preserve

Preserve these artifacts and docs as the sandbox result bundle:

- `docs/SANDBOX_MVP_LOCAL_EVENT_IMPLEMENTATION_OUTLINE.md`
- `docs/SANDBOX_MVP_FIRST_RESULT_ANALYSIS.md`
- `docs/SANDBOX_MVP_COMMUTATIVITY_RESULT_ANALYSIS.md`
- `docs/SANDBOX_MVP_COMMUTATIVITY_ROBUSTNESS_CHECK.md`
- `docs/SANDBOX_MVP_CONSOLIDATED_STATUS.md`
- `checkpoints/sandbox_local_event_mvp_local_operator_eventmask/best.pt`
- `checkpoints/sandbox_local_event_mvp_local_operator_eventmask/history.json`
- `checkpoints/sandbox_local_event_mvp_monolithic_baseline/best.pt`
- `checkpoints/sandbox_local_event_mvp_monolithic_baseline/history.json`
- `artifacts/sandbox_local_event_mvp_eval_eventmask/local_operator_test_summary.json`
- `artifacts/sandbox_local_event_mvp_eval_eventmask/local_operator_event_type_breakdown.csv`
- `artifacts/sandbox_local_event_pairs/train_summary.json`
- `artifacts/sandbox_local_event_pairs/val_summary.json`
- `artifacts/sandbox_local_event_pairs/test_summary.json`
- `artifacts/sandbox_commutativity_eval/eventmask_test/`
- `artifacts/sandbox_local_event_pairs_robustness/`
- `artifacts/sandbox_commutativity_eval_robustness/`

Also preserve the implemented sandbox-only files that produced the result:

- `data/generate_sandbox_local_event_mvp.py`
- `data/sandbox_local_event_dataset.py`
- `data/generate_sandbox_local_event_pairs.py`
- `models/sandbox_local_event_operator.py`
- `models/sandbox_monolithic_baseline.py`
- `train/train_sandbox_local_event_mvp.py`
- `train/eval_sandbox_local_event_mvp.py`
- `train/eval_sandbox_commutativity.py`

## What Not To Do Next

Keep closed:

- no training-time commutativity loss
- no further sandbox architecture iteration
- no proposal discovery work in this sandbox line
- no pair-event training
- no rendered observations or pixels
- no backend transfer
- no real data
- no raw image input
- no hypergraphs
- no LLM integration
- no Step33 reopening
- no Step22-29 local tweaking
- no README update
- no repo-wide default changes
- no official phase/status change

Final recommendation: preserve and stop. The sandbox MVP is now a clean positive feasibility result for oracle-targeted local event rewriting and composition, not a mandate to keep expanding the sandbox.
