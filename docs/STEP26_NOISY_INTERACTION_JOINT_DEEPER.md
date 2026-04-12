# Step 26: Coverage-Emphasized Noisy Interaction Joint Training

## Scope

Step 26 introduces one recipe-level mechanism variable:

`joint_recipe = light_joint_baseline | coverage_emphasized_joint`

The goal is to test whether a deeper but still controlled joint recipe can recover the noisy multievent interaction headroom exposed by Step 25. The target pressure point is strongly-interacting proposal coverage and downstream changed-edge/add behavior.

Fixed constraints:

- proposal init: `checkpoints/proposal_noisy_obs_p2/best.pt`
- rewrite init: `checkpoints/step6_noisy_rewrite_rft1/best.pt`
- node threshold: `0.15`
- edge threshold: `0.10`
- architecture unchanged
- noisy Step 22 multievent interaction substrate
- noisy structured observation input, clean latent next-state target
- no threshold sweep
- no proposal-only or rewrite-only side tweak
- no extra module or calibration variable

中文简述：这一步只测试一个 recipe 级变化，就是在 joint fine-tuning 里更明确地压 proposal coverage。它不是重新开阈值、proposal-only、rewrite-only 或 chooser 线。

## Recipe

Trainer:

- `train/train_step26_noisy_interaction_joint_deeper.py`

The Step 26 recipe keeps the Step24-style joint training path, but increases coverage pressure using one coherent recipe:

- `joint_proposal_loss_weight = 2.0`
- `edge_scope_pos_weight = 4.0`
- checkpoint selection includes proposal edge recall

Validation selection metric:

`0.25 * full_edge + 0.25 * context_edge + 0.20 * changed_edge + 0.10 * add + 0.10 * delete + 0.10 * proposal_edge_recall`

This keeps the experiment as one mechanism variable: stronger proposal-coverage emphasis inside joint noisy interaction-aware fine-tuning.

## Data

Step 26 uses the same Step 22/23 noisy Step 5 substrate already created for Step 23 and Step 24.

| Split | Samples | Dependency Buckets | Corruption Settings |
|---|---:|---|---|
| train | 8100 transitions | 2700 each: fully, partial, strong | 2700 each: N1, N2, N3 |
| val | 2700 transitions | 900 each: fully, partial, strong | 900 each: N1, N2, N3 |
| test | Step 22 noisy sequence test | 600 each final bucket | 600 each final N1/N2/N3 |

Training files:

- `data/graph_event_step23_noisy_step5_train_transitions.pkl`
- `data/graph_event_step23_noisy_step5_val_transitions.pkl`

Test file:

- `data/graph_event_step22_noisy_step5_test.pkl`

## Training Summary

Save dir:

- `checkpoints/step26_noisy_interaction_joint_deeper`

Best checkpoint:

- best epoch: `4`
- best validation selection score: `0.644949`
- proposal checkpoint: `checkpoints/step26_noisy_interaction_joint_deeper/proposal_best.pt`
- rewrite checkpoint: `checkpoints/step26_noisy_interaction_joint_deeper/rewrite_best.pt`

Best validation noisy metrics:

| Metric | Value |
|---|---:|
| full-edge | 0.8235 |
| context-edge | 0.8286 |
| changed-edge | 0.3804 |
| add | 0.3209 |
| delete | 0.4398 |
| proposal edge recall | 0.7978 |

The validation behavior already showed the intended recipe shape: much stronger changed/add and proposal recall, with weaker full/context stability.

## Main Test Results

Final-step noisy Step 5 test metrics:

| Run | Full Edge | Context Edge | Changed Edge | Add | Delete | Proposal Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + W012 | 0.8749 | 0.8818 | 0.2360 | 0.1027 | 0.3885 | 0.7095 | 0.5481 |
| noisy P2 + RFT1 | 0.8754 | 0.8821 | 0.2573 | 0.1111 | 0.4245 | 0.7095 | 0.5481 |
| Step23 proposal-only + RFT1 | 0.8813 | 0.8882 | 0.2349 | 0.0797 | 0.4125 | 0.6653 | 0.5951 |
| Step24 light joint | 0.8762 | 0.8829 | 0.2472 | 0.1195 | 0.3933 | 0.7159 | 0.5347 |
| Step26 coverage-emphasized joint | 0.8264 | 0.8313 | 0.3557 | 0.3061 | 0.4125 | 0.8100 | 0.3926 |
| noisy P2 + I1520 | 0.8361 | 0.8410 | 0.3714 | 0.1090 | 0.6715 | 0.7095 | 0.5481 |
| oracle scope + RFT1 | 0.9303 | 0.9370 | 0.3233 | 0.3690 | 0.2710 | 1.0000 | 0.0000 |

Step26 vs noisy P2 + RFT1:

| Metric | Delta |
|---|---:|
| full-edge | -0.0490 |
| context-edge | -0.0508 |
| changed-edge | +0.0984 |
| add | +0.1950 |
| delete | -0.0120 |
| proposal edge recall | +0.1005 |
| out-of-scope miss | -0.1555 |

Step26 recovers real proposal coverage and edit-sensitive behavior, especially add. The price is a large context/full-edge stability cost.

## Strongly-Interacting Slice

Final-step strongly-interacting metrics:

| Run | Full Edge | Context Edge | Changed Edge | Add | Delete | Proposal Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + RFT1 | 0.8771 | 0.8808 | 0.1008 | 0.0833 | 0.3333 | 0.7741 | 0.8527 |
| Step23 proposal-only + RFT1 | 0.8820 | 0.8858 | 0.0930 | 0.0917 | 0.1111 | 0.7306 | 0.8527 |
| Step24 light joint | 0.8793 | 0.8831 | 0.0853 | 0.0833 | 0.1111 | 0.7701 | 0.8372 |
| Step26 coverage-emphasized joint | 0.8287 | 0.8314 | 0.2403 | 0.2583 | 0.0000 | 0.8523 | 0.6744 |
| noisy P2 + I1520 | 0.8378 | 0.8414 | 0.0853 | 0.0750 | 0.2222 | 0.7741 | 0.8527 |
| oracle scope + RFT1 | 0.9324 | 0.9356 | 0.2868 | 0.2833 | 0.3333 | 1.0000 | 0.0000 |

Step26 vs noisy P2 + RFT1 on strongly-interacting examples:

| Metric | Delta |
|---|---:|
| full-edge | -0.0484 |
| context-edge | -0.0494 |
| changed-edge | +0.1395 |
| add | +0.1750 |
| delete | -0.3333 |
| proposal edge recall | +0.0782 |
| out-of-scope miss | -0.1783 |

This is the key result. Step26 does what Step24 failed to do on changed/add: strongly-interacting changed-edge rises from `0.1008` to `0.2403`, and add rises from `0.0833` to `0.2583`. But delete collapses to `0.0000`, and context-edge drops by about five points.

中文解释：coverage-emphasized joint 真的打开了强交互 add/changed 的门，但它也把 rewrite 的稳定性和 delete 处理打坏了。这个点有信号，但不是稳定默认。

## Corruption Breakdown

Step26 final-step metrics by corruption regime:

| Corruption | Full Edge | Context Edge | Changed Edge | Add | Delete | Proposal Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| N1 | 0.8493 | 0.8544 | 0.3792 | 0.3333 | 0.4317 | 0.8493 | 0.3658 |
| N2 | 0.8276 | 0.8326 | 0.3423 | 0.3082 | 0.3813 | 0.8142 | 0.3658 |
| N3 | 0.8023 | 0.8070 | 0.3456 | 0.2767 | 0.4245 | 0.7665 | 0.4463 |

The recipe remains edit-active across N1/N2/N3. Stronger corruption still lowers full/context stability and proposal recall, but changed-edge does not collapse.

## Event Family Notes

All-step Step26 event-family metrics:

| Event Family | Changed Edge | Add | Delete | Context Edge | Proposal Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|
| edge_add | 0.3445 | 0.3445 | NA | 0.8240 | 0.3985 | 0.6015 |
| edge_delete | 0.4288 | NA | 0.4288 | 0.8270 | 0.8656 | 0.1344 |
| motif_type_flip | NA | NA | NA | 0.8302 | 0.9043 | NA |
| node_state_update | NA | NA | NA | 0.8243 | 0.8970 | NA |

Compared with Step24 all-step behavior, Step26 strongly improves edge_add. The final strongly-interacting delete slice is still the clearest failure, even though aggregate edge_delete remains usable.

## Decision Answers

1. Does explicit proposal-coverage emphasis in joint training recover real strongly-interacting headroom?

Yes, for proposal coverage, changed-edge, and add. Strongly-interacting proposal edge recall improves from `0.7741` to `0.8523`, out-of-scope miss improves from `0.8527` to `0.6744`, changed-edge improves from `0.1008` to `0.2403`, and add improves from `0.0833` to `0.2583`. This is a real headroom recovery, not just noise.

2. Does it avoid the Step24 failure of improving overall coverage without improving the intended strongly-interacting downstream target?

Partially yes. Unlike Step24, Step26 substantially improves strongly-interacting changed-edge/add. However, it introduces a new failure: strongly-interacting delete drops to `0.0000`, and broad full/context-edge lose about five points. It fixes the intended changed/add target but does not produce a balanced system point.

3. Is it strong enough to become a retained noisy interaction-aware branch candidate while stable defaults remain unchanged?

Yes as an aggressive diagnostic branch candidate, not as a default. Step26 is the first joint noisy interaction recipe in this line that clearly recovers strongly-interacting changed/add behavior. It should be retained as evidence that coverage-emphasized joint coupling is a real mechanism family. Stable defaults remain unchanged because the stability/delete costs are too large.

4. If it still fails, does that justify parking this substrate line for now rather than continuing deeper joint coupling?

It is not a total failure, but it is not promotable. The result argues against continuing simple coverage pressure alone. If this substrate remains a priority, the next recipe would need a genuinely new balancing mechanism that protects delete/context while keeping the recovered add coverage. Otherwise, park the noisy interaction-aware joint line for now and keep Step26 as the aggressive headroom branch.

## Outcome

Stable defaults remain unchanged:

- noisy broad default: `RFT1 + calibrated P2`
- clean broad default: `W012`
- interaction-aware alternative: `I1520`

Step26 should be retained as a noisy interaction-aware branch candidate because it demonstrates real coverage-emphasized joint signal on strongly-interacting changed/add. It should not replace the stable noisy default because it gives back too much full/context stability and damages strongly-interacting delete.

Artifacts:

- `artifacts/step26_noisy_interaction_joint_deeper/noisy_step26_joint_deeper.json`
- `artifacts/step26_noisy_interaction_joint_deeper/noisy_step26_joint_deeper.csv`
- `artifacts/step26_noisy_interaction_joint_deeper/summary.json`
- `artifacts/step26_noisy_interaction_joint_deeper/summary.csv`

Checkpoints:

- `checkpoints/step26_noisy_interaction_joint_deeper/proposal_best.pt`
- `checkpoints/step26_noisy_interaction_joint_deeper/rewrite_best.pt`
