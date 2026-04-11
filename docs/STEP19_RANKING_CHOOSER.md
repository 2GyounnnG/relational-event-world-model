# Step 19: Ranking-Aligned Rescue Fallback Chooser

## Scope

Step 19 tests one training-objective change for the compact Step 17 fallback chooser.

Fixed:

- proposal operating mode: Step 9c fixed-budget internal completion
- rescue budget: `rescue_budget_fraction = 0.10`
- proposal backbone frozen
- Step 9 completion scorer frozen
- rewrite backbone frozen
- chooser feature family unchanged from Step 17
- keep-fraction grid only reused for evaluation: `0.02`, `0.05`, `0.10`, `0.20`

New mechanism variable:

- `chooser_objective = vanilla_bce | pairwise_keep_ranking`

中文说明: 这一步只换 chooser 的训练目标，不换特征、不换模型大小、不动 proposal/rewrite。目标是确认“当前 compact chooser 是否只是训练目标没对齐”。

## Implementation

Added:

- `train/train_step19_ranking_chooser.py`
- `train/eval_step19_ranking_chooser.py`

Artifacts:

- `artifacts/step19_ranking_chooser/noisy_p2_rft1.json`
- `artifacts/step19_ranking_chooser/noisy_p2_rft1.csv`

Checkpoint:

- `checkpoints/step19_ranking_chooser_noisy_p2_rft1/best.pt`

Training objective:

- positives: rescued edges where Step9c-path rewrite output is strictly better than base-path rewrite output against GT next-state
- negatives: all other rescued edges
- tie rule: fallback/base wins ties
- loss: pairwise logistic ranking loss over positive-vs-negative rescued-edge pairs in each batch
- no tuned margin, no class-weight sweep, no focal loss
- checkpoint selection: validation AP for the chooser-positive ranking target

Optional clean training/eval was skipped because the noisy P2 + RFT1 run is the required decision run and the clean run would require a separate trained ranking chooser.

## Training Summary

Noisy P2 + RFT1:

| Epoch | Val AP | Val AUROC | Val target-positive fraction |
|---:|---:|---:|---:|
| 1 | 0.1107 | 0.7243 | 0.0551 |
| 2 | 0.1086 | 0.7240 | 0.0551 |
| 3 | 0.1116 | 0.7296 | 0.0551 |
| 4 | 0.1123 | 0.7320 | 0.0551 |
| 5 | 0.1076 | 0.7261 | 0.0551 |

Best epoch: `4`

Best validation selection score: `0.1123`

## Ranking Diagnostics on Test

| Objective | AP | AUROC | Target-positive fraction |
|---|---:|---:|---:|
| Step18 vanilla BCE gate | 0.1114 | 0.7094 | 0.0539 |
| Step19 pairwise ranking gate | 0.1086 | 0.7128 | 0.0539 |

Event-type AP:

| Event type | Vanilla AP | Pairwise AP | Direction |
|---|---:|---:|---|
| edge_add | 0.1544 | 0.1632 | slightly better |
| edge_delete | 0.1314 | 0.1222 | worse |
| node_state_update | 0.1129 | 0.0994 | worse |
| motif_type_flip | 0.1163 | 0.1123 | slightly worse |

Overall, pairwise ranking marginally improves AUROC but does not improve AP on the main noisy test distribution.

## Noisy P2 + RFT1 System Results

| Mode | Keep fraction | Full edge | Context edge | Changed edge | Add | Delete |
|---|---:|---:|---:|---:|---:|---:|
| base | NA | 0.8742 | 0.8836 | 0.2710 | 0.0958 | 0.4370 |
| Step9c completion only | 1.0000 | 0.8409 | 0.8482 | 0.3650 | 0.2815 | 0.4440 |
| Step17 thresholded gate | 0.0000 | 0.8742 | 0.8836 | 0.2710 | 0.0958 | 0.4370 |
| Step18 vanilla top-k 0.02 | 0.0200 | 0.8735 | 0.8828 | 0.2732 | 0.0958 | 0.4412 |
| Step19 ranking top-k 0.02 | 0.0200 | 0.8735 | 0.8828 | 0.2729 | 0.0958 | 0.4407 |
| Step18 vanilla top-k 0.05 | 0.0500 | 0.8725 | 0.8817 | 0.2746 | 0.0953 | 0.4444 |
| Step19 ranking top-k 0.05 | 0.0500 | 0.8724 | 0.8817 | 0.2746 | 0.0958 | 0.4440 |
| Step18 vanilla top-k 0.10 | 0.1000 | 0.8704 | 0.8795 | 0.2829 | 0.1105 | 0.4463 |
| Step19 ranking top-k 0.10 | 0.1001 | 0.8703 | 0.8794 | 0.2825 | 0.1095 | 0.4463 |
| Step18 vanilla top-k 0.20 | 0.2000 | 0.8659 | 0.8747 | 0.2958 | 0.1369 | 0.4463 |
| Step19 ranking top-k 0.20 | 0.2000 | 0.8657 | 0.8745 | 0.2993 | 0.1442 | 0.4463 |
| oracle false-scope fallback | NA | 0.8756 | 0.8835 | 0.3650 | 0.2815 | 0.4440 |
| oracle choose-better | 0.0539 | 0.8773 | 0.8852 | 0.3664 | 0.2820 | 0.4463 |

## Chooser Frontier Quality

| Mode | Keep fraction | Target precision@k | Target recall@k | Kept false-scope fraction | Kept changed fraction |
|---|---:|---:|---:|---:|---:|
| Step18 vanilla top-k 0.02 | 0.0200 | 0.1563 | 0.0580 | 0.9271 | 0.0313 |
| Step19 ranking top-k 0.02 | 0.0200 | 0.1563 | 0.0580 | 0.9479 | 0.0278 |
| Step18 vanilla top-k 0.05 | 0.0500 | 0.1641 | 0.1522 | 0.9277 | 0.0236 |
| Step19 ranking top-k 0.05 | 0.0500 | 0.1613 | 0.1496 | 0.9277 | 0.0209 |
| Step18 vanilla top-k 0.10 | 0.1000 | 0.1314 | 0.2437 | 0.9291 | 0.0362 |
| Step19 ranking top-k 0.10 | 0.1001 | 0.1313 | 0.2437 | 0.9312 | 0.0347 |
| Step18 vanilla top-k 0.20 | 0.2000 | 0.1022 | 0.3791 | 0.9451 | 0.0369 |
| Step19 ranking top-k 0.20 | 0.2000 | 0.1054 | 0.3907 | 0.9407 | 0.0403 |
| oracle choose-better | 0.0539 | 1.0000 | 1.0000 | 0.4958 | 0.4649 |

The pairwise objective only helps at the largest frontier point, and only modestly.

## Rescued-Edge Behavior

| Mode | False-scope preserve | True-changed correct-edit |
|---|---:|---:|
| Step9c completion only | 0.3103 | 0.7895 |
| Step17 thresholded gate | 0.9395 | 0.0321 |
| Step18 vanilla top-k 0.10 | 0.8617 | 0.1389 |
| Step19 ranking top-k 0.10 | 0.8618 | 0.1346 |
| Step18 vanilla top-k 0.20 | 0.7676 | 0.2543 |
| Step19 ranking top-k 0.20 | 0.7680 | 0.2756 |
| oracle false-scope fallback | 0.9395 | 0.7895 |
| oracle choose-better | 0.9677 | 0.8024 |

The ranking objective does not improve the practical 10% chooser point. At 20%, it improves true-changed edit rate but does not recover context enough to become a plausible operating point.

中文解读: pairwise objective 不是完全没效果，但效果集中在更激进的 20% keep 点；而这个点的 context cost 已经明显偏高。

## Decision Answers

### 1. Does the ranking-aligned chooser objective materially improve the rescued-edge keep frontier over Step 18?

No.

The main test AP is slightly worse:

- vanilla BCE AP: `0.1114`
- pairwise ranking AP: `0.1086`

AUROC rises slightly:

- vanilla BCE AUROC: `0.7094`
- pairwise ranking AUROC: `0.7128`

But the system frontier is essentially unchanged at 2%, 5%, and 10%. The only visible gain is at 20%, where changed-edge rises from `0.2958` to `0.2993` and add rises from `0.1369` to `0.1442`, but context-edge drops slightly from `0.8747` to `0.8745`.

### 2. Is there now a small keep_fraction that becomes a plausible active interface branch candidate?

No.

The 10% point remains too weak:

- changed-edge `0.2825`, far below Step9c `0.3650`
- add `0.1095`, far below Step9c `0.2815`
- context-edge `0.8794`, below base `0.8836`

The 20% point improves edit sensitivity somewhat, but context is still too degraded and the edit recovery remains far from oracle choose-better.

### 3. Does the gain come without giving back too much context stability?

No.

The only nontrivial edit gain appears at 20%, and that point gives back context:

- ranking top-k 0.20 context-edge: `0.8745`
- base context-edge: `0.8836`
- oracle choose-better context-edge: `0.8852`

This is not a good system-level tradeoff.

### 4. If it still fails, does that make stronger chooser representation the next justified step?

Yes.

The objective-only change is not enough. The chooser target remains rare and top-tail precision remains weak:

- target-positive fraction: `0.0539`
- pairwise top-k 0.10 target precision: `0.1313`
- oracle target precision: `1.0000`

The interface direction remains promising because oracle choose-better is still very strong, but the compact feature family appears insufficient. A stronger rescue-conditioned chooser representation is the next justified step if this branch continues.

## Bottom Line

Step19 closes the “maybe just objective alignment” question for the current compact chooser.

- Pairwise ranking does not materially improve the Step18 frontier.
- No small keep-fraction becomes a convincing active branch.
- Stable defaults remain unchanged.
- The next meaningful interface test should involve a stronger chooser representation, not more objective-only tweaks or keep-fraction tuning.

