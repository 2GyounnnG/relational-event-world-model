# Step 21: Two-Path Local-Subgraph Chooser

## Scope

Step 21 tests one contained chooser-line representation jump:

- `chooser_representation = shallow_probe | learned_two_path_local_subgraph`
- proposal backbone frozen
- Step 9 completion scorer frozen
- rewrite backbone frozen
- Step 9c rescued-edge set fixed
- rescue budget fixed at `0.10`
- keep-fraction frontier fixed at `0.02, 0.05, 0.10, 0.20`

This is still the structured synthetic graph-event world. It does not change the stable defaults.

中文总结：这一步只问 “更强一点的 rescued-edge 局部表示能不能让 chooser 的 top tail 变好”，不是重新训练 proposal/rewrite。

## Implementation

New scripts:

- `train/train_step21_two_path_local_subgraph_chooser.py`
- `train/eval_step21_two_path_local_subgraph_chooser.py`

The Step 21 chooser builds a small candidate-centric local pooling encoder around each rescued edge. For each rescued-edge pair, it uses:

- compact two-path interface scores from Step 17
- frozen proposal node latents and node-scope scores
- base-path and Step9c-path node predictions
- endpoints
- one-hop neighbors inside predicted node scope
- common neighbors inside predicted node scope
- local/common pooled node embeddings
- local degree/count summaries

The target remains the Step 17 chooser target:

- positive iff the Step9c path is strictly better than the base path on the rescued edge against clean GT next adjacency
- ties fall back to base

## Run

Required noisy run completed:

- proposal: `checkpoints/proposal_noisy_obs_p2/best.pt`
- completion: `checkpoints/step9_edge_completion_noisy_p2/best.pt`
- rewrite: `checkpoints/step6_noisy_rewrite_rft1/best.pt`
- data: `data/graph_event_step6a_train.pkl`, `data/graph_event_step6a_val.pkl`, `data/graph_event_step6a_test.pkl`
- thresholds: node `0.15`, edge `0.10`

Training summary:

| item | value |
|---|---:|
| best epoch | 2 |
| best validation AP / selection score | 0.1125 |
| pair feature dim | 22 |
| node feature dim | 157 |
| node embed dim | 48 |
| hidden dim | 96 |

Artifacts:

- `checkpoints/step21_two_path_local_subgraph_noisy_p2_rft1/best.pt`
- `checkpoints/step21_two_path_local_subgraph_noisy_p2_rft1/training_summary.json`
- `artifacts/step21_two_path_local_subgraph_chooser/noisy_p2_rft1.json`
- `artifacts/step21_two_path_local_subgraph_chooser/noisy_p2_rft1.csv`

Optional clean `W012` run was not executed; the required noisy default-stack run was the focus.

## Ranking Diagnostics

Noisy P2 + RFT1, rescued-edge chooser target:

| mode | AP | AUROC |
|---|---:|---:|
| Step 18 vanilla BCE compact gate | 0.1114 | 0.7094 |
| Step 19 pairwise compact chooser | 0.1086 | 0.7128 |
| Step 20 compact probe | 0.1084 | 0.7098 |
| Step 20 enriched shallow probe | 0.1159 | 0.7210 |
| Step 21 two-path local-subgraph chooser | 0.1100 | 0.7172 |

Step 21 did not materially beat the Step 20 enriched shallow probe on chooser top-tail quality. It improved AUROC over the compact baselines but gave back the AP gain that made Step 20 interesting.

## Fixed Keep Frontier

Noisy P2 + RFT1:

| mode | keep frac | full-edge | context-edge | changed-edge | add | delete |
|---|---:|---:|---:|---:|---:|---:|
| base | NA | 0.8742 | 0.8836 | 0.2710 | 0.0958 | 0.4370 |
| Step 9c completion only | 0.10 rescue budget | 0.8409 | 0.8482 | 0.3650 | 0.2815 | 0.4440 |
| Step 20 enriched probe | 0.10 | 0.8704 | 0.8795 | 0.2841 | 0.1139 | 0.4454 |
| Step 21 local-subgraph chooser | 0.10 | 0.8704 | 0.8795 | 0.2829 | 0.1105 | 0.4463 |
| Step 20 enriched probe | 0.20 | 0.8660 | 0.8747 | 0.3027 | 0.1520 | 0.4454 |
| Step 21 local-subgraph chooser | 0.20 | 0.8659 | 0.8747 | 0.3017 | 0.1500 | 0.4454 |
| oracle false-scope fallback | oracle | 0.8756 | 0.8835 | 0.3650 | 0.2815 | 0.4440 |
| oracle choose-better | oracle | 0.8773 | 0.8852 | 0.3664 | 0.2820 | 0.4463 |

Step 21 does not create a new useful keep-fraction point. At `0.10`, it is slightly worse than Step 20 enriched on changed-edge and add, with essentially the same context-edge. At `0.20`, it is also slightly worse than Step 20 enriched.

## Rescued-Edge Selection

| mode | keep frac | chooser precision | chooser recall | kept changed precision | kept false-scope frac |
|---|---:|---:|---:|---:|---:|
| Step 20 enriched | 0.10 | 0.1324 | 0.2456 | 0.0393 | 0.9287 |
| Step 21 local-subgraph | 0.10 | 0.1353 | 0.2508 | 0.0358 | 0.9280 |
| Step 20 enriched | 0.20 | 0.1090 | 0.4043 | 0.0471 | 0.9343 |
| Step 21 local-subgraph | 0.20 | 0.1066 | 0.3952 | 0.0466 | 0.9343 |
| oracle choose-better | 0.0539 actual | 1.0000 | 1.0000 | 0.4649 | 0.4958 |

The local-subgraph chooser slightly improves chooser-target precision/recall at `0.10`, but this does not translate into better changed-edge/add metrics. The kept false-scope fraction remains very high, so the core context-cost problem remains.

## Decision Answers

1. Does the learned two-path local-subgraph chooser materially beat the Step 20 enriched shallow probe on chooser top-tail quality?

No. Step 21 AP is `0.1100` versus Step 20 enriched AP `0.1159`; AUROC is also lower (`0.7172` versus `0.7210`). It is not a material top-tail improvement.

2. Does it materially improve the keep frontier over Step 18 / Step 19 / Step 20 at the same keep fractions?

No. It is close to the previous frontier but does not beat Step 20 enriched at the practically relevant `0.10` or `0.20` keep fractions. The small local-subgraph representation does not convert the oracle gap into a deployable branch.

3. Is there now a small keep_fraction that becomes a plausible active interface branch candidate?

No. `0.10` remains too conservative to recover Step9c edit gains, and `0.20` gives more changed/add but still carries context cost and does not outperform Step 20. Stable noisy default remains `RFT1 + calibrated P2`.

4. If it still fails, does that mean this interface line now requires a larger representation jump than is justified for the current phase, and should therefore be deprioritized?

Yes. Step 21 was the first substantial but still contained chooser-representation test. It did not materially improve the real frontier. The evidence says the current interface line likely needs a larger representation jump than this phase justifies. 建议：park this chooser line for now rather than continuing small tweaks.

## Interpretation

The oracle choose-better gap remains large, so the interface idea is not invalid. But the learnable chooser variants tried so far cannot reliably identify the rare rescued edges where Step9c should be kept. Step 21 shows that simply adding a small local-subgraph pooling encoder is not enough. The remaining problem is not keep-fraction tuning, not compact-objective choice, and not a tiny representation tweak. It is either a substantially harder representation problem or a signal/label sparsity problem that needs a larger phase decision.

Stable defaults remain unchanged.
