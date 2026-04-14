# Sandbox MVP Commutativity Result Analysis

## Scope

This is a composition-only sandbox check. It is separate from the earlier single-event 3-5 node result. The pair generator uses 8-10 node chain spring graphs because disjoint independent pairs are not feasible under the conservative expanded event scopes on the original 3-5 node graphs.

No model was retrained for this check. The comparison is:

- revised local operator: `checkpoints/sandbox_local_event_mvp_local_operator_eventmask/best.pt`
- existing monolithic baseline: `checkpoints/sandbox_local_event_mvp_monolithic_baseline/best.pt`

## Pair Generation

Pair generation succeeded.

Generated splits:

| split | accepted pairs | attempts | pair types |
| --- | ---: | ---: | --- |
| train | 1000 | 1000 | impulse+impulse 485, impulse+break 447, break+break 68 |
| val | 200 | 200 | impulse+impulse 93, impulse+break 96, break+break 11 |
| test | 200 | 200 | impulse+impulse 106, impulse+break 82, break+break 12 |

The test summary records candidate-pair rejections inside accepted worlds:

- `scope_node_overlap`: 1769
- `scope_edge_overlap`: 172

These are not failed dataset-level attempts; they are rejected candidate pairings while searching within worlds. Dataset-level generation accepted 200 test pairs in 200 worlds. Oracle AB/BA discrepancy was exactly `0.0` for the accepted test pairs.

This should be treated as a composition-only subtask on a larger pair-feasible sandbox variant, not as a direct continuation of the 3-5 node single-event distribution.

## Overall Comparison

| model | pred AB/BA mismatch RMSE | AB target RMSE | BA target RMSE |
| --- | ---: | ---: | ---: |
| local_operator_eventmask | 0.000000 | 0.013881 | 0.013881 |
| monolithic_baseline | 0.000343 | 0.056710 | 0.056712 |

The revised local operator is clearly better on this composition-only check. It has zero measured AB/BA prediction mismatch at evaluator precision and much lower AB/BA target error.

## Pair-Type Breakdown

### impulse + impulse

| model | pairs | pred mismatch | AB target RMSE | BA target RMSE |
| --- | ---: | ---: | ---: | ---: |
| local_operator_eventmask | 106 | 0.000000 | 0.002347 | 0.002347 |
| monolithic_baseline | 106 | 0.000207 | 0.020635 | 0.020633 |

### impulse + break

| model | pairs | pred mismatch | AB target RMSE | BA target RMSE |
| --- | ---: | ---: | ---: | ---: |
| local_operator_eventmask | 82 | 0.000000 | 0.023676 | 0.023676 |
| monolithic_baseline | 82 | 0.000446 | 0.092564 | 0.092572 |

### break + break

| model | pairs | pred mismatch | AB target RMSE | BA target RMSE |
| --- | ---: | ---: | ---: | ---: |
| local_operator_eventmask | 12 | 0.000000 | 0.048830 | 0.048830 |
| monolithic_baseline | 12 | 0.000839 | 0.130363 | 0.130375 |

The local operator wins across all available pair types. The smallest sample is break+break with only 12 test pairs, so that bucket should be read cautiously, but its direction matches the other two buckets.

## Conservative Conclusion

Choose: **1. local is clearly better on composition**.

This conclusion is bounded to the composition-only sandbox variant: independent pairs on 8-10 node chain graphs with oracle scopes, oracle direct event masks, and no proposal discovery. It does not imply solved proposal discovery, rendered perception, backend transfer, real data readiness, or Step33 progress.

The mechanism is also unsurprising: the revised local operator applies exactly scoped, direct-event-conditioned updates and copies outside event scope, so two disjoint events compose cleanly. The monolithic model has no copy-by-construction guarantee and accumulates small whole-graph prediction errors when applied sequentially.

## Recommendation

Choose: **b. keep composition as evaluation-only for now**.

Do not move to training-time commutativity loss yet. The current result already shows strong composition behavior for the revised local operator under oracle masks. The next useful question is whether this advantage remains stable under more pair samples or slightly harder but still structured sandbox conditions, not whether to add a new training objective immediately.

Keep closed:

- no training-time commutativity loss
- no proposal discovery
- no pair-event training
- no rendered observations
- no backend transfer
- no real data
- no Step33 reopening
- no Step22-29 local tweaking
- no repo-wide status or phase change
