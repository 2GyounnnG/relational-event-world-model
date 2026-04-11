# Step 17: Rescue-Conditioned Fallback Gate

## Scope

Step 17 tests a proposal/rewrite interface mechanism, not a new proposal or rewrite backbone.

- Proposal operating mode is fixed to Step 9c fixed-budget internal completion at `rescue_budget_fraction = 0.10`.
- Proposal checkpoints, Step 9 completion checkpoints, rewrite checkpoints, thresholds, and budgets are unchanged.
- The learned gate only acts on rescued edges: edges added by Step 9c relative to the base proposal.
- For each rescued edge, the gate chooses between two frozen outputs:
  - base proposal + unchanged rewrite
  - Step 9c proposal + unchanged rewrite
- Non-rescued edges remain on the Step 9c output path.

中文备注: 这里不是再训练 rewrite，也不是再调 proposal。这个实验只问一个接口问题: 对 Step 9c 新救回来的边，能不能学会什么时候用 Step 9c 输出，什么时候退回 base 输出。

## Implementation

Added:

- `train/train_step17_rescue_fallback_gate.py`
- `train/eval_step17_rescue_fallback_gate.py`

Artifacts:

- `artifacts/step17_rescue_fallback_gate/clean_w012.json`
- `artifacts/step17_rescue_fallback_gate/clean_w012.csv`
- `artifacts/step17_rescue_fallback_gate/noisy_p2_rft1.json`
- `artifacts/step17_rescue_fallback_gate/noisy_p2_rft1.csv`

Gate checkpoints:

- `checkpoints/step17_fallback_gate_clean_w012/best.pt`
- `checkpoints/step17_fallback_gate_noisy_p2_rft1/best.pt`

Tie rule:

- If base-path and Step9c-path are equally correct against the clean target edge, the target chooses fallback/base.
- This intentionally makes the gate conservative unless Step9c is strictly better on that rescued edge.

## Runs

| Run | Proposal / completion | Rewrite | Dataset | Status |
|---|---|---|---|---|
| clean W012 | clean proposal + Step9 clean completion | W012 | `data/graph_event_test.pkl` | completed |
| noisy P2 + RFT1 | calibrated P2 + Step9 noisy completion | RFT1 | `data/graph_event_step6a_test.pkl` | completed |

Optional noisy P2 + W012 was skipped because the gate uses rewrite-path features and would require a separate rewrite-specific gate fit; that was not trivial enough for the optional slot.

## Training Behavior

| Run | Best epoch | Best validation score | Target choose-Step9c fraction | Predicted choose-Step9c fraction | Validation AP |
|---|---:|---:|---:|---:|---:|
| clean W012 | 1 | 0.96995 | 0.01225 | 0.00000 | 0.03626 |
| noisy P2 + RFT1 | 1 | 0.90833 | 0.05515 | 0.00000 | 0.10508 |

The learned gate collapsed to the conservative all-fallback operating point on validation and test.

中文解释: 正样本很稀疏，尤其 clean 只有约 1.2% 的 rescued edges 需要保留 Step9c 路径。当前 compact interface features + direct chooser target 没有学出足够强的 0.5 决策边界。

## System-Level Results

### Clean W012

| Mode | Full edge | Context edge | Changed edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base | 0.9689 | 0.9828 | 0.0728 | 0.0220 | 0.1208 |
| Step9c completion only | 0.9575 | 0.9707 | 0.1049 | 0.0455 | 0.1611 |
| Step9c + oracle false-scope fallback | 0.9688 | 0.9822 | 0.1049 | 0.0455 | 0.1611 |
| Step17 learned gate | 0.9689 | 0.9828 | 0.0728 | 0.0220 | 0.1208 |
| oracle choose-better-of-two-paths | 0.9694 | 0.9828 | 0.1049 | 0.0455 | 0.1611 |

### Noisy P2 + RFT1

| Mode | Full edge | Context edge | Changed edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base | 0.8742 | 0.8836 | 0.2710 | 0.0958 | 0.4370 |
| Step9c completion only | 0.8409 | 0.8482 | 0.3650 | 0.2815 | 0.4440 |
| Step9c + oracle false-scope fallback | 0.8756 | 0.8835 | 0.3650 | 0.2815 | 0.4440 |
| Step17 learned gate | 0.8742 | 0.8836 | 0.2710 | 0.0958 | 0.4370 |
| oracle choose-better-of-two-paths | 0.8773 | 0.8852 | 0.3664 | 0.2820 | 0.4463 |

## Gate Behavior

| Run | Rescued edges | Fallback fraction | Step9c-kept fraction | Target Step9c fraction | Chosen correctness on rescued |
|---|---:|---:|---:|---:|---:|
| clean W012 | 9,477 | 1.0000 | 0.0000 | 0.0089 | 0.9722 |
| noisy P2 + RFT1 | 28,760 | 1.0000 | 0.0000 | 0.0539 | 0.9082 |

The oracle choose-better upper bound keeps only a small fraction of rescued edges on the Step9c path:

- clean: 0.0089 kept on Step9c
- noisy: 0.0539 kept on Step9c

That small kept subset carries nearly all of the Step9c edit gain.

## Rescued-Edge Behavior

### Clean W012

| Mode | False-scope preserve | True-changed correct-edit |
|---|---:|---:|
| Step9c completion only | 0.7686 | 0.3194 |
| oracle false-scope fallback | 1.0000 | 0.3194 |
| Step17 learned gate | 1.0000 | 0.0000 |
| oracle choose-better | 1.0000 | 0.3194 |

### Noisy P2 + RFT1

| Mode | False-scope preserve | True-changed correct-edit |
|---|---:|---:|
| Step9c completion only | 0.3103 | 0.7895 |
| oracle false-scope fallback | 0.9395 | 0.7895 |
| Step17 learned gate | 0.9395 | 0.0321 |
| oracle choose-better | 0.9677 | 0.8024 |

The learned gate successfully recovers context only by falling back everywhere. That destroys the rescued true-changed edit value.

中文结论: 学到的不是“保留真救援、回退假救援”，而是“全部回退”。所以它看起来修复了 context，但其实放弃了 Step9c 的主要收益。

## Decision Answers

### 1. Can the learned fallback gate recover a substantial share of the Step 16 oracle-fallback context gain?

Yes on context metrics, but only in the degenerate all-fallback sense.

Noisy context-edge goes from Step9c `0.8482` back to `0.8836`, matching the base context level. However, this is not the desired learned interface behavior; it recovers context by rejecting every rescued edge.

### 2. Does it preserve most of Step9c's changed-edge / add / delete gains?

No.

On noisy P2 + RFT1:

- changed-edge drops from Step9c `0.3650` to `0.2710`
- add drops from `0.2815` to `0.0958`
- delete drops from `0.4440` to `0.4370`

This essentially reverts to the base proposal behavior.

### 3. Is it strong enough to become the next active interface branch candidate while stable defaults remain unchanged?

No.

The oracle choose-better result proves the interface has a high ceiling, but the learned compact gate does not reach it. Step17 should be treated as a diagnostic failure of the current compact learned chooser, not as a promoted branch.

Stable defaults remain unchanged:

- broad clean default: W012
- noisy default: RFT1 + calibrated P2
- proposal-side branch candidate: Step9c fixed-budget internal completion remains a reference branch, not a default

### 4. If it fails, what is the main issue?

The main issue is insufficient gate separability / operating-point learnability from the compact interface features.

It is not evidence that the base-vs-Step9c chooser is inherently unlearnable, because the oracle choose-better upper bound is very strong:

- noisy oracle choose-better reaches full-edge `0.8773`, context-edge `0.8852`, changed-edge `0.3664`, add `0.2820`, delete `0.4463`
- this beats both base and Step9c on the relevant combined tradeoff

But the current learned gate cannot identify the small subset of rescued edges where Step9c should be kept.

## Minimal Next Mechanism

The next mechanism should not be rewrite backbone retraining yet.

The clean next probe is a better rescue-conditioned chooser representation or training objective specifically aligned to the rare “Step9c strictly better than base” target.

One minimal candidate:

- a rescue-edge chooser trained with class-balanced / utility-weighted supervision for the base-vs-Step9c decision
- still frozen proposal
- still frozen rewrite
- still fixed Step9c budget

中文: 下一步不该直接大改 rewrite。更干净的问题是: 能不能让 chooser 学会那 5% 左右真正该保留 Step9c 的 rescued edges。

