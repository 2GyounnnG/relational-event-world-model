# Step 24: Light Joint Noisy Interaction-Aware Coupling

## Scope

Step 24 tests one new mechanism variable:

`coupling_regime = frozen_rewrite_baseline | light_joint_noisy_interaction`

The goal is to see whether jointly fine-tuning proposal and rewrite on the noisy Step 5 multievent substrate can improve the strongly-interacting bottleneck that Step 22 exposed and Step 23 did not solve.

Fixed constraints:

- proposal init: `checkpoints/proposal_noisy_obs_p2/best.pt`
- rewrite init: `checkpoints/step6_noisy_rewrite_rft1/best.pt`
- node threshold: `0.15`
- edge threshold: `0.10`
- no threshold sweep
- no proposal-only or rewrite-only side tweak
- no extra loss-family sweep
- noisy observation input, clean target next state

中文简述：这一步只测试轻量 joint coupling。proposal 和 rewrite 都更新，但结构、阈值、Step 9c/chooser 线都不动。

## Data

Step 24 uses the Step 23 transition version of the Step 22 noisy Step 5 substrate.

| Split | Transition Samples | Dependency Buckets | Corruption Settings |
|---|---:|---|---|
| train | 8100 | 2700 each: fully, partial, strong | 2700 each: N1, N2, N3 |
| val | 2700 | 900 each: fully, partial, strong | 900 each: N1, N2, N3 |
| test | Step 22 noisy sequence test | 600 each final bucket | 600 each final N1/N2/N3 |

Training files:

- `data/graph_event_step23_noisy_step5_train_transitions.pkl`
- `data/graph_event_step23_noisy_step5_val_transitions.pkl`

Test file:

- `data/graph_event_step22_noisy_step5_test.pkl`

## Training Setup

Trainer:

- `train/train_step24_noisy_interaction_joint.py`

Run:

- save dir: `checkpoints/step24_noisy_interaction_joint`
- best epoch: `3`
- best validation selection score: `0.715592`
- joint proposal loss weight: `1.0`
- fp edge keep weight: `0.12`
- lr: `1e-4`
- early stopping: epoch `5`

Validation selection metric:

`0.35 * full_edge_acc + 0.35 * context_edge_acc + 0.15 * changed_edge_acc + 0.15 * delete`

Best checkpoints:

- proposal: `checkpoints/step24_noisy_interaction_joint/proposal_best.pt`
- rewrite: `checkpoints/step24_noisy_interaction_joint/rewrite_best.pt`

## Main Test Results

Final-step noisy Step 5 test metrics:

| Run | Full Edge | Context Edge | Changed Edge | Add | Delete | Proposal Edge Recall | Proposal Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + W012 | 0.8749 | 0.8818 | 0.2360 | 0.1027 | 0.3885 | 0.7095 | 0.5481 |
| noisy P2 + RFT1 | 0.8754 | 0.8821 | 0.2573 | 0.1111 | 0.4245 | 0.7095 | 0.5481 |
| Step23 proposal-only + RFT1 | 0.8813 | 0.8882 | 0.2349 | 0.0797 | 0.4125 | 0.6653 | 0.5951 |
| Step24 joint candidate | 0.8762 | 0.8829 | 0.2472 | 0.1195 | 0.3933 | 0.7159 | 0.5347 |
| noisy P2 + I1520 | 0.8361 | 0.8410 | 0.3714 | 0.1090 | 0.6715 | 0.7095 | 0.5481 |

Delta, Step24 vs noisy P2 + RFT1:

| Metric | Delta |
|---|---:|
| full-edge | +0.0007 |
| context-edge | +0.0009 |
| changed-edge | -0.0101 |
| add | +0.0084 |
| delete | -0.0312 |
| proposal edge recall | +0.0064 |
| proposal out-of-scope miss | -0.0134 |

Step 24 improves proposal coverage a little and improves add, but does not improve the overall changed-edge/delete tradeoff over the stable noisy default.

## Interaction Breakdown

Final-step bucket metrics:

| Run | Bucket | Full Edge | Context Edge | Changed Edge | Add | Delete | Proposal Edge Recall | Out-of-Scope Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + RFT1 | fully_independent | 0.8765 | 0.8855 | 0.2624 | 0.0821 | 0.4167 | 0.5896 | 0.5319 |
| Step23 + RFT1 | fully_independent | 0.8830 | 0.8923 | 0.2411 | 0.0462 | 0.4079 | 0.5481 | 0.5768 |
| Step24 joint | fully_independent | 0.8732 | 0.8818 | 0.2742 | 0.1077 | 0.4167 | 0.6296 | 0.4941 |
| noisy P2 + RFT1 | partially_dependent | 0.8728 | 0.8800 | 0.3099 | 0.1667 | 0.4389 | 0.7085 | 0.4532 |
| Step23 + RFT1 | partially_dependent | 0.8789 | 0.8865 | 0.2807 | 0.1111 | 0.4333 | 0.6610 | 0.5205 |
| Step24 joint | partially_dependent | 0.8761 | 0.8839 | 0.2749 | 0.1605 | 0.3778 | 0.7040 | 0.4708 |
| noisy P2 + RFT1 | strongly_interacting | 0.8771 | 0.8808 | 0.1008 | 0.0833 | 0.3333 | 0.7741 | 0.8527 |
| Step23 + RFT1 | strongly_interacting | 0.8820 | 0.8858 | 0.0930 | 0.0917 | 0.1111 | 0.7306 | 0.8527 |
| Step24 joint | strongly_interacting | 0.8793 | 0.8831 | 0.0853 | 0.0833 | 0.1111 | 0.7701 | 0.8372 |

The target slice remains unresolved. Step 24 slightly reduces strongly-interacting out-of-scope miss from `0.8527` to `0.8372`, but strongly-interacting changed-edge falls to `0.0853`, and delete stays low at `0.1111`.

中文结论：joint coupling 确实让 proposal coverage 有一点恢复，但强交互 downstream edit 没跟上；瓶颈没有被真正解开。

## Corruption Breakdown

Step24 final-step metrics by corruption regime:

| Corruption | Full Edge | Context Edge | Changed Edge | Add | Delete | Proposal Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| N1 | 0.9013 | 0.9084 | 0.2416 | 0.1132 | 0.3885 | 0.7548 | 0.5235 |
| N2 | 0.8767 | 0.8837 | 0.2349 | 0.1195 | 0.3669 | 0.7229 | 0.5034 |
| N3 | 0.8505 | 0.8568 | 0.2651 | 0.1258 | 0.4245 | 0.6699 | 0.5772 |

Step24 remains usable under all N1/N2/N3 regimes, but the strongest noise level still carries the expected broad degradation.

## Event Family Notes

Step24 all-step event-family metrics:

| Event Family | Changed Edge | Add | Delete | Context Edge | Proposal Changed Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|
| edge_add | 0.1317 | 0.1317 | NA | 0.8768 | 0.1562 | 0.8438 |
| edge_delete | 0.4060 | NA | 0.4060 | 0.8838 | 0.8142 | 0.1858 |
| motif_type_flip | NA | NA | NA | 0.8824 | NA | NA |
| node_state_update | NA | NA | NA | 0.8795 | NA | NA |

Compared with noisy P2 + RFT1, Step24 improves edge-add proposal recall (`0.1457 -> 0.1562`) and add (`0.1120 -> 0.1317`) all-steps, but weakens delete (`0.4266 -> 0.4060`).

## Decision Answers

1. Does light joint noisy interaction-aware coupling materially improve strongly-interacting proposal coverage relative to noisy P2 + RFT1 and Step23?

Only slightly relative to noisy P2 + RFT1, and clearly relative to Step23. Strongly-interacting out-of-scope miss improves from `0.8527` to `0.8372` vs P2/RFT1, and from `0.8527` to `0.8372` vs Step23. Proposal edge recall is `0.7701`, slightly below P2/RFT1 `0.7741` but above Step23 `0.7306`.

2. Does it improve strongly-interacting changed-edge/delete downstream?

No. Strongly-interacting changed-edge falls from `0.1008` to `0.0853`, and delete falls from `0.3333` to `0.1111`. This is the main reason Step24 should not be promoted.

3. Does it preserve broad full-edge/context-edge enough to be viable?

Broad stability is preserved. Overall full-edge is `0.8762` vs `0.8754`; context-edge is `0.8829` vs `0.8821`. But preservation alone is not enough because the intended interaction edit metrics do not improve.

4. Does it reproduce an I1520-like aggressive tradeoff under noise?

No. I1520 under noisy P2 is much more edit-aggressive overall: changed-edge `0.3714`, delete `0.6715`, but full/context-edge are much lower (`0.8361` / `0.8410`). Step24 is a mild, stable point, not an I1520-like aggressive point.

5. If it fails, is the next justified step deeper joint noisy interaction-aware coupling rather than more proposal-only tweaking?

Yes, if this substrate remains the priority. Step24 shows that light joint coupling can slightly adjust coverage without collapsing stability, but it does not unlock strongly-interacting downstream edits. More proposal-only tweaking is already disfavored by Step23. The next justified experiment would need deeper joint coupling or a more interaction-specific joint objective, not another proposal-only line.

## Outcome

Step24 should not replace the stable noisy default.

- Stable noisy broad default remains: `RFT1 + calibrated P2`.
- Step24 is a useful diagnostic joint-coupling candidate, but not a promoted branch.
- Step23 proposal-only remains negative.
- I1520 remains an aggressive interaction-aware alternative, not the broad noisy default.

Artifacts:

- `artifacts/step24_noisy_interaction_joint/noisy_step24_joint.json`
- `artifacts/step24_noisy_interaction_joint/noisy_step24_joint.csv`
- `artifacts/step24_noisy_interaction_joint/summary.json`
- `artifacts/step24_noisy_interaction_joint/summary.csv`

Checkpoints:

- `checkpoints/step24_noisy_interaction_joint/proposal_best.pt`
- `checkpoints/step24_noisy_interaction_joint/rewrite_best.pt`
