# Step 22: Noisy Multievent Interaction

## Scope

Step 22 introduces one new benchmark variable:

- `observation_regime = clean | noisy`

The substrate is the existing Step 5 three-event multievent interaction regime. The noisy side uses Step 6a-style structured observation corruption on the current graph observation while keeping the clean latent target graph unchanged.

This is an evaluation-first pack. No proposal, rewrite, completion, chooser, or interaction model was trained.

中文说明：这一步不是重新打开 Step 2-6 的调参线，也不是继续 Step 17-21 chooser line。它只问一个新的 substrate 问题：Step 5 的交互复杂性遇到 Step 6 的 noisy structured observation 后会怎样。

## Benchmark Design

New files:

- `data/generate_step22_noisy_step5_data.py`
- `train/eval_step22_noisy_multievent_interaction.py`

Generated data:

- `data/graph_event_step22_noisy_step5_val.pkl`
- `data/graph_event_step22_noisy_step5_test.pkl`

Dataset status:

| split | clean source | noisy samples | sequence length | dependency buckets | corruption settings |
|---|---|---:|---:|---|---|
| val | `graph_event_step5_val.pkl` | 900 | 3 | 300 each | N1/N2/N3, 300 each |
| test | `graph_event_step5_test.pkl` | 1800 | 3 | 600 each | N1/N2/N3, 600 each |

Evaluation mode:

- The evaluator uses the clean current graph for `observation_regime=clean`.
- The evaluator uses `obs_graph_inputs[k]` for `observation_regime=noisy`.
- Targets are always the clean Step 5 `graph_steps[k]`.
- Metrics are computed per transition, with a final-step summary on transition 3.
- This isolates noisy observation on the Step 5 substrate; it is not an autoregressive feedback benchmark.

## Runs

Required clean side:

| run | proposal | rewrite | data |
|---|---|---|---|
| clean W012 | `scope_proposal_node_edge_flipw2` | `fp_keep_w012` | `graph_event_step5_test.pkl` |
| clean I1520 | `scope_proposal_node_edge_flipw2` | `step5_interaction_i1520` | `graph_event_step5_test.pkl` |

Required noisy side:

| run | proposal | thresholds | rewrite | data |
|---|---|---|---|---|
| noisy P2 + W012 | `proposal_noisy_obs_p2` | node `0.15`, edge `0.10` | `fp_keep_w012` | `graph_event_step22_noisy_step5_test.pkl` |
| noisy P2 + RFT1 | `proposal_noisy_obs_p2` | node `0.15`, edge `0.10` | `step6_noisy_rewrite_rft1` | `graph_event_step22_noisy_step5_test.pkl` |

Optional run completed because the checkpoint and wiring were already available:

| run | proposal | thresholds | rewrite | data |
|---|---|---|---|---|
| noisy P2 + I1520 | `proposal_noisy_obs_p2` | node `0.15`, edge `0.10` | `step5_interaction_i1520` | `graph_event_step22_noisy_step5_test.pkl` |

Step 9c branch candidate was not run; this pack intentionally does not continue the chooser/internal-completion line.

## Artifacts

Machine-readable outputs:

- `artifacts/step22_noisy_multievent_interaction/clean_w012.json`
- `artifacts/step22_noisy_multievent_interaction/clean_i1520.json`
- `artifacts/step22_noisy_multievent_interaction/noisy_p2_w012.json`
- `artifacts/step22_noisy_multievent_interaction/noisy_p2_rft1.json`
- `artifacts/step22_noisy_multievent_interaction/noisy_p2_i1520.json`
- `artifacts/step22_noisy_multievent_interaction/summary.json`
- `artifacts/step22_noisy_multievent_interaction/summary.csv`

Each run also has a matching per-run `.csv`.

## Overall Final-Step Metrics

| run | full-edge | context-edge | changed-edge | add | delete | state MAE |
|---|---:|---:|---:|---:|---:|---:|
| clean W012 | 0.9741 | 0.9842 | 0.0537 | 0.0126 | 0.1007 | 0.0706 |
| clean I1520 | 0.9574 | 0.9666 | 0.1141 | 0.0126 | 0.2302 | 0.0619 |
| noisy P2 + W012 | 0.8749 | 0.8818 | 0.2360 | 0.1027 | 0.3885 | 0.1328 |
| noisy P2 + RFT1 | 0.8754 | 0.8821 | 0.2573 | 0.1111 | 0.4245 | 0.1177 |
| noisy P2 + I1520 | 0.8361 | 0.8410 | 0.3714 | 0.1090 | 0.6715 | 0.1230 |

The clean-to-noisy comparison is not a pure same-checkpoint ablation because the noisy side uses the prescribed calibrated P2 front-end. It is the intended system-stack comparison for Step 22.

## Clean To Noisy Changes

| comparison | full-edge delta | context-edge delta | changed-edge delta | add delta | delete delta | state MAE delta |
|---|---:|---:|---:|---:|---:|---:|
| W012 clean -> noisy P2+W012 | -0.0993 | -0.1024 | +0.1823 | +0.0901 | +0.2878 | +0.0622 |
| I1520 clean -> noisy P2+I1520 | -0.1213 | -0.1256 | +0.2573 | +0.0964 | +0.4412 | +0.0611 |

Noisy P2 strongly increases edit exposure, especially delete, but the cost is a large context/full-edge drop. This is the same broad tradeoff pattern seen earlier, now on the Step 5 interaction substrate.

## Noisy Stack Comparison

| comparison | full-edge delta | context-edge delta | changed-edge delta | add delta | delete delta | state MAE delta |
|---|---:|---:|---:|---:|---:|---:|
| noisy W012 -> noisy RFT1 | +0.0006 | +0.0003 | +0.0213 | +0.0084 | +0.0360 | -0.0151 |
| noisy W012 -> noisy I1520 | -0.0388 | -0.0408 | +0.1353 | +0.0063 | +0.2830 | -0.0098 |
| noisy RFT1 -> noisy I1520 | -0.0394 | -0.0411 | +0.1141 | -0.0021 | +0.2470 | +0.0053 |

`RFT1 + calibrated P2` is the best broad noisy stack here: it improves changed-edge, add, delete, and state MAE over noisy W012 without giving back context/full-edge. `I1520 + calibrated P2` is much more edit/delete-sensitive, but it is not broad-safe.

## Dependency-Bucket Final Metrics

### Noisy P2 + RFT1

| bucket | full-edge | context-edge | changed-edge | add | delete | proposal out-of-scope miss edge |
|---|---:|---:|---:|---:|---:|---:|
| fully independent | 0.8765 | 0.8855 | 0.2624 | 0.0821 | 0.4167 | 0.5319 |
| partially dependent | 0.8728 | 0.8800 | 0.3099 | 0.1667 | 0.4389 | 0.4532 |
| strongly interacting | 0.8771 | 0.8808 | 0.1008 | 0.0833 | 0.3333 | 0.8527 |

Strongly interacting sequences remain the clearest changed-edge/proposal-coverage bottleneck. Their context/full-edge does not collapse relative to the other buckets, but their changed-edge performance and proposal changed-edge coverage are much worse.

### Noisy P2 + I1520

| bucket | full-edge | context-edge | changed-edge | add | delete | proposal out-of-scope miss edge |
|---|---:|---:|---:|---:|---:|---:|
| fully independent | 0.8385 | 0.8445 | 0.4137 | 0.0872 | 0.6930 | 0.5319 |
| partially dependent | 0.8320 | 0.8370 | 0.4269 | 0.1605 | 0.6667 | 0.4532 |
| strongly interacting | 0.8378 | 0.8414 | 0.0853 | 0.0750 | 0.2222 | 0.8527 |

I1520 preserves an edit/delete-sensitive style under noise, but it does not preserve a strong interaction-bucket advantage. On the strongly interacting bucket, it is not better than RFT1 on changed-edge and pays a large context cost.

## Event-Family Breakdown

Noisy P2 + RFT1, all steps:

| event type | full-edge | context-edge | changed-edge | add | delete | proposal out-of-scope miss edge |
|---|---:|---:|---:|---:|---:|---:|
| edge_add | 0.8588 | 0.8756 | 0.1120 | 0.1120 | NA | 0.8543 |
| edge_delete | 0.8750 | 0.8850 | 0.4266 | NA | 0.4266 | 0.1924 |
| motif_type_flip | 0.8803 | 0.8803 | NA | NA | NA | NA |
| node_state_update | 0.8793 | 0.8793 | NA | NA | NA | NA |

The combined substrate keeps the earlier asymmetry: `edge_add` is still heavily proposal-limited, while `edge_delete` is much more reachable by the noisy P2 front-end and benefits more from edit-sensitive rewrite variants.

## Proposal-Side Findings

For noisy P2 final-step overall:

- proposal edge-scope recall: `0.7095`
- proposal edge-scope excess ratio: `0.9133`
- proposal changed-edge recall: `0.4519`
- proposal out-of-scope miss edge: `0.5481`

By dependency bucket:

- fully independent out-of-scope miss: `0.5319`
- partially dependent out-of-scope miss: `0.4532`
- strongly interacting out-of-scope miss: `0.8527`

中文解释：噪声下 P2 的 edge scope 很大，oracle edge-scope recall 也高，但 strongly_interacting 的真正 changed edges 仍大量落在 scope 外。这里的核心不是 “scope 不够大”，而是 “复杂交互里的关键 changed edge 没被正确覆盖”。

## Decision Answers

1. Does noise disproportionately amplify the Step 5 interaction bottleneck?

Yes for the edit/proposal bottleneck, but not as broad context collapse. Strongly interacting sequences under noisy P2 + RFT1 have changed-edge `0.1008` versus `0.2624` / `0.3099` for fully/partially dependent, and proposal out-of-scope miss `0.8527` versus `0.5319` / `0.4532`. Context-edge is similar across buckets, so the amplification is specifically in changed-edge coverage/editability.

2. On the noisy multievent substrate, is RFT1 + calibrated P2 still the safest broad default?

Yes. It has the best broad noisy balance among evaluated stacks: full-edge `0.8754`, context-edge `0.8821`, changed-edge `0.2573`, add `0.1111`, delete `0.4245`. It improves over noisy W012 without sacrificing stability, while I1520 is too context-costly for broad default use.

3. Does I1520 preserve any meaningful interaction-aware edge under noise?

Partially. I1520 preserves a strong edit/delete-sensitive behavior overall, especially delete (`0.6715` final, `0.6791` on edge_delete all-steps). But it does not preserve a clean strongly-interacting bucket advantage under noise: strongly-interacting changed-edge is `0.0853`, below RFT1’s `0.1008`, with much lower context-edge.

4. Based on the result, is the next training line justified as noisy interaction-aware adaptation, or is the current default family already robust enough to carry forward?

The current default family is robust enough to carry forward as a baseline, but not enough to close the combined Step 5 + Step 6 bottleneck. If Step 22 becomes the next target substrate, the justified training line is noisy interaction-aware adaptation, starting from `RFT1 + calibrated P2` as the broad baseline and treating I1520 as an edit-sensitive reference, not a default.

## Interpretation

Step 22 confirms that the project now has a meaningful next-phase substrate: noisy structured observation plus multievent interaction complexity. The failure is not catastrophic; the family remains usable. But the strongly-interacting changed-edge gap is severe and proposal-side out-of-scope miss remains high in that bucket. The next useful line should not be Step 2-6 micro-tuning and should not continue the parked chooser-interface line. It should ask how to adapt the proposal/rewrite interface to noisy interaction-heavy sequences directly.

Stable defaults remain unchanged.
