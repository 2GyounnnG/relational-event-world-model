# Step 23: Noisy Interaction-Aware Proposal Adaptation

## Scope

Step 23 asks whether proposal-only adaptation can help on the combined Step 5/Step 6 substrate:

- Step 5 multievent interaction complexity.
- Step 6 noisy structured observations.
- Proposal initialized from noisy P2.
- Rewrite kept fixed at RFT1 for the main run.
- Thresholds kept fixed: `node_threshold = 0.15`, `edge_threshold = 0.10`.

This is still the structured synthetic graph-event world. No perception, real-world data, hypergraphs, LLM components, rewrite retraining, threshold retuning, or joint proposal/rewrite training were added.

中文简述：本步骤只测试“proposal 在 noisy + interaction 条件下单独微调是否足够”。rewrite 不动，阈值不动。

## Data

Step 23 reuses the Step 22 noisy Step 5 corruption format and flattens multievent rollouts into single-transition proposal-supervision samples.

| Split | Sequence Samples | Transition Samples | Dependency Buckets | Corruption Settings |
|---|---:|---:|---|---|
| train | 2700 | 8100 | 2700 each: fully, partial, strong | 2700 each: N1, N2, N3 |
| val | 900 | 2700 | 900 each: fully, partial, strong | 900 each: N1, N2, N3 |
| test | 1800 | 5400 transitions evaluated | 600 each final bucket | 600 each final N1/N2/N3 |

Generated files:

- `data/graph_event_step22_noisy_step5_train.pkl`
- `data/graph_event_step23_noisy_step5_train_transitions.pkl`
- `data/graph_event_step23_noisy_step5_val_transitions.pkl`

## Training Setup

The Step 23 candidate is:

- `proposal_training_regime = noisy_interaction_aware_P2`
- init proposal: `checkpoints/proposal_noisy_obs_p2/best.pt`
- save dir: `checkpoints/step23_noisy_interaction_proposal`
- train input: `obs_node_feats`, `obs_adj`
- target: clean oracle event-scope node/edge masks
- proposal architecture unchanged
- rewrite not trained

One fixed interaction-aware exposure mechanism was used inside this single regime: a deterministic weighted sampler over dependency buckets.

| Bucket | Sampler Weight |
|---|---:|
| fully_independent | 1.0 |
| partially_dependent | 1.5 |
| strongly_interacting | 2.0 |

Validation selection score:

`0.40 * overall_edge_f1 + 0.20 * overall_node_f1 + 0.25 * strong_changed_edge_recall + 0.15 * strong_edge_scope_recall`

Best checkpoint:

- best epoch: `1`
- best validation selection score: `0.350953`
- best checkpoint: `checkpoints/step23_noisy_interaction_proposal/best.pt`

## Main Test Results

Final-step noisy Step 5 test metrics:

| Run | Full Edge | Context Edge | Changed Edge | Add | Delete | Proposal Edge Recall | Proposal Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + W012 | 0.8749 | 0.8818 | 0.2360 | 0.1027 | 0.3885 | 0.7095 | 0.5481 |
| noisy P2 + RFT1 | 0.8754 | 0.8821 | 0.2573 | 0.1111 | 0.4245 | 0.7095 | 0.5481 |
| Step23 proposal + RFT1 | 0.8813 | 0.8882 | 0.2349 | 0.0797 | 0.4125 | 0.6653 | 0.5951 |

Delta, Step23 proposal + RFT1 vs noisy P2 + RFT1:

| Metric | Delta |
|---|---:|
| full-edge | +0.0059 |
| context-edge | +0.0061 |
| changed-edge | -0.0224 |
| add | -0.0314 |
| delete | -0.0120 |
| proposal edge recall | -0.0442 |
| proposal out-of-scope miss | +0.0470 |

Interpretation: the candidate preserves context better but does this by becoming more conservative on edge scope. That is the wrong direction for the Step 22 bottleneck.

## Interaction Bucket Breakdown

Final-step bucket metrics:

| Run | Bucket | Full Edge | Context Edge | Changed Edge | Add | Delete | Proposal Edge Recall | Out-of-Scope Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + RFT1 | fully_independent | 0.8765 | 0.8855 | 0.2624 | 0.0821 | 0.4167 | 0.5896 | 0.5319 |
| Step23 + RFT1 | fully_independent | 0.8830 | 0.8923 | 0.2411 | 0.0462 | 0.4079 | 0.5481 | 0.5768 |
| noisy P2 + RFT1 | partially_dependent | 0.8728 | 0.8800 | 0.3099 | 0.1667 | 0.4389 | 0.7085 | 0.4532 |
| Step23 + RFT1 | partially_dependent | 0.8789 | 0.8865 | 0.2807 | 0.1111 | 0.4333 | 0.6610 | 0.5205 |
| noisy P2 + RFT1 | strongly_interacting | 0.8771 | 0.8808 | 0.1008 | 0.0833 | 0.3333 | 0.7741 | 0.8527 |
| Step23 + RFT1 | strongly_interacting | 0.8820 | 0.8858 | 0.0930 | 0.0917 | 0.1111 | 0.7306 | 0.8527 |

Strongly interacting sequences were the intended target. Step 23 did not reduce strongly-interacting changed-edge out-of-scope miss: it stayed at `0.8527`. Delete on the strongly-interacting slice dropped sharply from `0.3333` to `0.1111`.

中文结论：强交互 bucket 的核心 miss 没有下降；这说明单独微调 proposal 没有真正打开 Step 22 暴露出的瓶颈。

## Corruption Breakdown

Final-step corruption metrics for noisy P2 + RFT1 vs Step23 + RFT1:

| Corruption | Run | Full Edge | Context Edge | Changed Edge | Add | Delete | Proposal Edge Recall | Out-of-Scope Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| N1 | noisy P2 + RFT1 | 0.9007 | 0.9078 | 0.2517 | 0.1006 | 0.4245 | 0.7314 | 0.5369 |
| N1 | Step23 + RFT1 | 0.9063 | 0.9136 | 0.2282 | 0.0692 | 0.4101 | 0.6985 | 0.5805 |
| N2 | noisy P2 + RFT1 | 0.8736 | 0.8804 | 0.2349 | 0.1069 | 0.3813 | 0.7219 | 0.5067 |
| N2 | Step23 + RFT1 | 0.8807 | 0.8878 | 0.2148 | 0.0755 | 0.3741 | 0.6699 | 0.5705 |
| N3 | noisy P2 + RFT1 | 0.8521 | 0.8581 | 0.2852 | 0.1258 | 0.4676 | 0.6752 | 0.6007 |
| N3 | Step23 + RFT1 | 0.8569 | 0.8632 | 0.2617 | 0.0943 | 0.4532 | 0.6274 | 0.6342 |

Across N1/N2/N3, Step 23 improves full/context-edge but consistently lowers changed-edge, add, and proposal edge recall.

## Event Family Notes

All-step event-family metrics:

| Event Family | Run | Changed Edge | Add | Delete | Proposal Changed-Edge Recall | Out-of-Scope Miss |
|---|---|---:|---:|---:|---:|---:|
| edge_add | noisy P2 + RFT1 | 0.1120 | 0.1120 | NA | 0.1457 | 0.8543 |
| edge_add | Step23 + RFT1 | 0.0987 | 0.0987 | NA | 0.1176 | 0.8824 |
| edge_delete | noisy P2 + RFT1 | 0.4266 | NA | 0.4266 | 0.8076 | 0.1924 |
| edge_delete | Step23 + RFT1 | 0.4229 | NA | 0.4229 | 0.7724 | 0.2276 |

The proposal-only fine-tune did not fix edge_add. It also slightly weakened edge_delete coverage.

## Decision Answers

1. Does proposal-only noisy interaction-aware adaptation materially improve strongly-interacting proposal coverage?

No. Strongly-interacting proposal edge recall fell from `0.7741` to `0.7306`, and strongly-interacting changed-edge out-of-scope miss stayed at `0.8527`.

2. Does it improve strongly-interacting changed-edge/delete downstream?

No. Strongly-interacting changed-edge fell from `0.1008` to `0.0930`; delete fell from `0.3333` to `0.1111`. Add moved slightly from `0.0833` to `0.0917`, but that is not enough to offset the delete/coverage failure.

3. Does it preserve broad full-edge/context-edge enough to be a viable noisy interaction-aware branch?

It preserves, and even improves, broad full/context-edge: full-edge `+0.0059`, context-edge `+0.0061`. However this appears to come from conservative proposal shrinkage, not better interaction coverage. It is not a viable interaction-aware branch because it gives back changed-edge/add/delete behavior.

4. If it fails, is the next justified step joint noisy interaction-aware coupling rather than more proposal-only tweaking?

Yes. This result suggests proposal-only adaptation with the existing architecture/training target is not enough. The next justified mechanism question should be joint noisy interaction-aware coupling, because the current proposal/rewrite interface may need co-adaptation to use broader or more interaction-sensitive scopes without simply trading edits for context stability.

## Outcome

Step 23 does not replace any stable default.

- Stable noisy default remains: `RFT1 + calibrated P2`.
- Step 23 proposal candidate should not be promoted.
- Step 9c remains only a separate proposal-side branch candidate.
- The next interaction/noise line should not be more proposal-only micro-tuning; it should test joint noisy interaction-aware coupling if the project continues along this substrate.

Artifacts:

- `artifacts/step23_noisy_interaction_aware_proposal/noisy_step23_p2_rft1.json`
- `artifacts/step23_noisy_interaction_aware_proposal/noisy_step23_p2_rft1.csv`
- `artifacts/step23_noisy_interaction_aware_proposal/summary.json`
- `artifacts/step23_noisy_interaction_aware_proposal/summary.csv`
