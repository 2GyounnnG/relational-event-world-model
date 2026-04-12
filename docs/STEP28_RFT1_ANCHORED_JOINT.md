# Step 28: RFT1-Anchored Coverage-Emphasized Joint Training

## Scope

Step 28 introduces one recipe-level mechanism variable:

`rewrite_constraint_regime = none | rft1_anchored_joint`

The goal is to keep the Step26 proposal-coverage direction while preventing the rewrite drift that Step27 isolated as harmful. No thresholds, architecture, proposal-only line, rewrite-only line, or calibration setting was changed.

Fixed constraints:

- proposal init: `checkpoints/proposal_noisy_obs_p2/best.pt`
- rewrite init: `checkpoints/step6_noisy_rewrite_rft1/best.pt`
- node threshold: `0.15`
- edge threshold: `0.10`
- noisy Step22 multievent interaction substrate
- noisy structured observation input, clean target next-state
- no threshold sweep
- no proposal-only or rewrite-only side tweak

中文简述：Step28 只问一个问题：能不能保留 Step26 proposal coverage 的收益，同时用 RFT1 anchor 防止 rewrite 把 delete/context 搞坏？

## Recipe

Trainer:

- `train/train_step28_rft1_anchored_joint.py`

The recipe keeps Step26-style coverage pressure:

- `joint_proposal_loss_weight = 2.0`
- `edge_scope_pos_weight = 4.0`
- coverage-aware checkpoint selection

The single rewrite constraint is parameter anchoring toward the RFT1 initialization:

- method: `parameter_l2_to_rft1_initialization`
- `rewrite_anchor_weight = 25.0`

Validation selection metric:

`0.25 * full_edge + 0.25 * context_edge + 0.20 * changed_edge + 0.10 * add + 0.10 * delete + 0.10 * proposal_edge_recall`

This is one coherent recipe-level variable: Step26 proposal coverage pressure plus an RFT1 rewrite parameter anchor.

## Training Summary

Save dir:

- `checkpoints/step28_rft1_anchored_joint`

Best checkpoint:

- best epoch: `4`
- best validation selection score: `0.645284`
- proposal checkpoint: `checkpoints/step28_rft1_anchored_joint/proposal_best.pt`
- rewrite checkpoint: `checkpoints/step28_rft1_anchored_joint/rewrite_best.pt`

Data:

| Split | Samples | Dependency Buckets | Corruption Settings |
|---|---:|---|---|
| train | 8100 transitions | 2700 each: fully, partial, strong | 2700 each: N1, N2, N3 |
| val | 2700 transitions | 900 each: fully, partial, strong | 900 each: N1, N2, N3 |

The anchor term stayed small but nonzero during training. At best epoch, validation still selected a Step26-like coverage/edit-active point.

## Main Test Results

Final-step noisy Step5 test metrics:

| Run | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + RFT1 | 0.8754 | 0.8821 | 0.2573 | 0.1111 | 0.4245 | 0.7095 | 0.5481 |
| Step24 light joint | 0.8762 | 0.8829 | 0.2472 | 0.1195 | 0.3933 | 0.7159 | 0.5347 |
| Step26 coverage joint | 0.8264 | 0.8313 | 0.3557 | 0.3061 | 0.4125 | 0.8100 | 0.3926 |
| Step26 proposal + RFT1 | 0.8264 | 0.8312 | 0.3758 | 0.3040 | 0.4580 | 0.8100 | 0.3926 |
| Step28 anchored joint | 0.8277 | 0.8327 | 0.3490 | 0.2935 | 0.4125 | 0.8075 | 0.4004 |
| noisy P2 + I1520 | 0.8361 | 0.8410 | 0.3714 | 0.1090 | 0.6715 | 0.7095 | 0.5481 |
| oracle scope + RFT1 | 0.9303 | 0.9370 | 0.3233 | 0.3690 | 0.2710 | 1.0000 | 0.0000 |

Step28 vs noisy P2 + RFT1:

| Metric | Delta |
|---|---:|
| full-edge | -0.0478 |
| context-edge | -0.0494 |
| changed-edge | +0.0917 |
| add | +0.1824 |
| delete | -0.0120 |
| proposal edge recall | +0.0980 |
| out-of-scope miss | -0.1477 |

Step28 preserves most of the Step26 proposal-side coverage/add gain, but it does not recover broad context/full-edge stability.

## Strongly-Interacting Slice

Final-step strongly-interacting metrics:

| Run | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + RFT1 | 0.8771 | 0.8808 | 0.1008 | 0.0833 | 0.3333 | 0.7741 | 0.8527 |
| Step24 light joint | 0.8793 | 0.8831 | 0.0853 | 0.0833 | 0.1111 | 0.7701 | 0.8372 |
| Step26 coverage joint | 0.8287 | 0.8314 | 0.2403 | 0.2583 | 0.0000 | 0.8523 | 0.6744 |
| Step26 proposal + RFT1 | 0.8254 | 0.8281 | 0.2558 | 0.2500 | 0.3333 | 0.8523 | 0.6744 |
| Step28 anchored joint | 0.8303 | 0.8331 | 0.2326 | 0.2500 | 0.0000 | 0.8515 | 0.6822 |
| oracle scope + RFT1 | 0.9324 | 0.9356 | 0.2868 | 0.2833 | 0.3333 | 1.0000 | 0.0000 |

Step28 vs noisy P2 + RFT1 on strongly-interacting examples:

| Metric | Delta |
|---|---:|
| full-edge | -0.0467 |
| context-edge | -0.0477 |
| changed-edge | +0.1318 |
| add | +0.1667 |
| delete | -0.3333 |
| proposal edge recall | +0.0774 |
| out-of-scope miss | -0.1705 |

Step28 preserves the proposal-side strong coverage/add pattern, but it repeats the Step26 strong-delete collapse. The RFT1 parameter anchor did not protect the strongly-interacting delete behavior.

中文结论：anchor 没有把 rewrite 拉回到安全的 RFT1 行为。强交互 add/changed 仍然有收益，但 delete 仍然归零，所以不能推广。

## Comparison To Factorized Reference

Step28 vs Step26 proposal + RFT1:

| Slice | Changed Edge Delta | Add Delta | Delete Delta | Context Delta | Edge Recall Delta | Out-of-Scope Delta |
|---|---:|---:|---:|---:|---:|---:|
| overall | -0.0268 | -0.0105 | -0.0456 | +0.0015 | -0.0025 | +0.0078 |
| strongly_interacting | -0.0233 | +0.0000 | -0.3333 | +0.0050 | -0.0008 | +0.0078 |

The factorized reference remains cleaner. It keeps the same proposal coverage, preserves strong add at `0.2500`, has higher strong changed-edge (`0.2558` vs `0.2326`), and avoids the delete collapse (`0.3333` vs `0.0000`).

## Event Family Notes

All-step Step28 event-family metrics:

| Event Family | Changed Edge | Add | Delete | Context Edge | Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|
| edge_add | 0.3396 | 0.3396 | NA | 0.8248 | 0.3922 | 0.6078 |
| edge_delete | 0.4288 | NA | 0.4288 | 0.8283 | 0.8671 | 0.1329 |
| motif_type_flip | NA | NA | NA | 0.8318 | 0.9051 | NA |
| node_state_update | NA | NA | NA | 0.8250 | 0.8964 | NA |

The all-step edge-delete aggregate remains usable, but this masks the strongly-interacting final-slice delete collapse. The intended failure mode is therefore slice-specific, not a global edge-delete collapse.

## Decision Answers

1. Can rewrite anchoring recover most of the Step26 proposal gains without Step26 rewrite harm?

No. Step28 keeps most of the Step26 proposal coverage gain: overall edge recall is `0.8075` vs Step26 `0.8100`, and strongly-interacting edge recall is `0.8515` vs Step26 `0.8523`. It also keeps strong add at `0.2500`. But it does not remove the Step26 rewrite harm: strongly-interacting delete remains `0.0000`.

2. Is Step28 better than Step26 proposal + RFT1, or does the factorized reference remain the cleaner point?

The factorized reference remains cleaner. `Step26 proposal + RFT1` has stronger overall changed-edge (`0.3758` vs `0.3490`), stronger overall delete (`0.4580` vs `0.4125`), stronger strongly-interacting changed-edge (`0.2558` vs `0.2326`), and preserves strongly-interacting delete (`0.3333` vs `0.0000`). Step28 only gives a tiny context/full-edge improvement, not enough to offset the edit loss.

3. Is Step28 strong enough to become a retained noisy interaction-aware branch candidate while stable defaults remain unchanged?

No. Step28 is informative, but not a retained branch candidate beyond Step26/Step27. It confirms that the proposal direction is useful, but this simple RFT1 parameter anchor does not solve the rewrite drift problem. Stable defaults remain unchanged.

4. If it still fails, does that imply the next clean move is to keep the Step26 proposal direction but abandon joint rewrite updating on this substrate?

Yes. The next clean move is to keep the Step26 proposal direction and abandon unconstrained or weakly anchored rewrite updating for this substrate. If future work continues here, rewrite should either remain RFT1-like or require a much stronger, explicitly behavior-preserving constraint. More simple Step26-style joint tweaks are low value.

## Outcome

Stable defaults remain unchanged:

- noisy broad default: `RFT1 + calibrated P2`
- clean broad default: `W012`
- interaction-aware alternative: `I1520`

Step28 does not replace Step26 or the Step27 factorized reference. The clean retained signal remains:

- keep: Step26 proposal direction for noisy interaction coverage/add
- reject: Step26/Step28 rewrite drift for strongly-interacting delete

Artifacts:

- `artifacts/step28_rft1_anchored_joint/noisy_step28_rft1_anchored_joint.json`
- `artifacts/step28_rft1_anchored_joint/noisy_step28_rft1_anchored_joint.csv`
- `artifacts/step28_rft1_anchored_joint/summary.json`
- `artifacts/step28_rft1_anchored_joint/summary.csv`

Checkpoints:

- `checkpoints/step28_rft1_anchored_joint/proposal_best.pt`
- `checkpoints/step28_rft1_anchored_joint/rewrite_best.pt`
