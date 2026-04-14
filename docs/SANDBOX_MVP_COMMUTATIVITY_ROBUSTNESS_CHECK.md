# Sandbox MVP Commutativity Robustness Check

## Scope

This is a bounded robustness check for the composition-only sandbox result. It does not change models, training, losses, or evaluation code. It is separate from the earlier 3-5 node single-event sandbox. The pair generator uses the already implemented 8-10 node chain-graph setting because conservative expanded scopes make independent event pairs infeasible in the original tiny 3-5 node graphs.

Checkpoints:

- revised local operator: `checkpoints/sandbox_local_event_mvp_local_operator_eventmask/best.pt`
- monolithic baseline: `checkpoints/sandbox_local_event_mvp_monolithic_baseline/best.pt`

## Pair Generation

Pair generation succeeded for all three seeds. Each seed produced 200 accepted test pairs in 200 worlds.

| seed | accepted pairs | impulse+impulse | impulse+break | break+break |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 200 | 108 | 81 | 11 |
| 1 | 200 | 93 | 96 | 11 |
| 2 | 200 | 106 | 82 | 12 |

The pair type counts are reasonable and stable enough for the overall robustness check. `break+break` remains the smallest bucket, so pair-type conclusions for that bucket should stay cautious.

## Overall Robustness Results

| seed | model | pred AB/BA mismatch RMSE | AB target RMSE | BA target RMSE |
| ---: | --- | ---: | ---: | ---: |
| 0 | local_operator_eventmask | 0.000000 | 0.014308 | 0.014308 |
| 0 | monolithic_baseline | 0.000359 | 0.056584 | 0.056585 |
| 1 | local_operator_eventmask | 0.000000 | 0.014343 | 0.014343 |
| 1 | monolithic_baseline | 0.000370 | 0.059694 | 0.059709 |
| 2 | local_operator_eventmask | 0.000000 | 0.013881 | 0.013881 |
| 2 | monolithic_baseline | 0.000343 | 0.056710 | 0.056712 |

The local advantage survived across pair-generation seeds.

Direction was stable for all three requested quantities:

- mismatch: local was lower on every seed, with zero measured AB/BA mismatch at evaluator precision
- AB target error: local was lower on every seed
- BA target error: local was lower on every seed

The magnitude was also stable. Local target RMSE stayed around `0.014`; monolithic target RMSE stayed around `0.057-0.060`. There is no sign in these three seeds that the composition advantage is a one-seed artifact.

## Interpretation

This robustness check strengthens the composition-only result: under oracle independent pairs, direct event masks, and conservative event scopes, the revised local operator composes more cleanly than the monolithic baseline.

The result should not be overextended. It does not test proposal discovery, noisy observations, rendered inputs, real data, backend transfer, or Step33. It also does not prove that training-time commutativity loss is needed. The current local model already achieves effectively exact order consistency on this evaluation setup without such a loss.

## Recommendation

Choose: **1. keep composition evaluation-only for now**.

Do not move to training-time commutativity loss yet. The robust result says the local architecture and oracle masks already give the desired composition behavior in this bounded sandbox. A training-time commutativity objective would add a new moving part before there is evidence that composition consistency is currently the bottleneck.

Keep closed:

- no training-time commutativity loss
- no retraining for this result
- no proposal discovery
- no rendered observations or pixels
- no backend transfer
- no real data
- no Step33 reopening
- no Step22-29 local tweaking
- no repo-wide default or phase/status change
