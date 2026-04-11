# Step 12: Guard Ranking Interaction

## Scope

Step 12 is a zero-training diagnostic. It tests whether the Step 11 event-scope guard contains useful residual signal that was wasted by the current `completion_score * guard_score` ranking rule.

No models were retrained. Stable defaults are unchanged.

Fixed setup:

| Setting | Value |
|---|---:|
| rescue budget fraction | 0.10 |
| candidate definition | both endpoints in predicted node scope, not already in base edge scope |
| rewrite conditioning after admission | unchanged Step 9 completion confidence |
| budget sweep | none |
| threshold sweep | none |

Ranking modes:

| Mode | Ranking rule |
|---|---|
| `completion_only` | rank by Step 9 completion score |
| `guard_only` | rank by Step 11 guard score |
| `product` | rank by completion score * guard score |
| `guard_then_completion` | primary sort by guard score, tie-break by completion score |
| `oracle_guard_then_completion` | GT event-scope primary sort, completion tie-break upper bound |

中文备注: 这一步只换排序规则，不训练新模型。目标是判断问题是不是 `completion * guard` 组合方式浪费了 guard 信号。

## Files And Artifacts

| Item | Path |
|---|---|
| evaluator | `train/eval_step12_guard_ranking_interaction.py` |
| clean metrics | `artifacts/step12_guard_ranking_interaction/clean_w012.json` |
| noisy RFT1 metrics | `artifacts/step12_guard_ranking_interaction/noisy_p2_rft1.json` |
| optional noisy W012 metrics | `artifacts/step12_guard_ranking_interaction/noisy_p2_w012.json` |
| summary CSVs | `artifacts/step12_guard_ranking_interaction/*.csv` |

## Overall Noisy Default Result

Main run: calibrated P2 + RFT1 on `graph_event_step6a_test.pkl`.

| Mode | Edge recall | Out-of-scope miss | Event-scope precision@B | Changed precision@B | False-scope frac | Changed-edge | Context-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.4810 | 0.5190 | NA | NA | NA | 0.2710 | 0.8836 | 0.0958 | 0.4370 |
| completion_only | 0.5922 | 0.4078 | 0.0507 | 0.0325 | 0.9493 | 0.3650 | 0.8482 | 0.2815 | 0.4440 |
| guard_only | 0.5913 | 0.4087 | 0.0513 | 0.0323 | 0.9487 | 0.3633 | 0.8482 | 0.2781 | 0.4440 |
| product | 0.5932 | 0.4068 | 0.0509 | 0.0328 | 0.9491 | 0.3642 | 0.8481 | 0.2810 | 0.4431 |
| guard_then_completion | 0.5913 | 0.4087 | 0.0513 | 0.0323 | 0.9487 | 0.3631 | 0.8482 | 0.2776 | 0.4440 |
| oracle_guard_then_completion | 0.9324 | 0.0676 | 0.2006 | 0.1320 | 0.7994 | 0.6051 | 0.8543 | 0.8069 | 0.4139 |
| naive closure | 0.9479 | 0.0521 | NA | NA | NA | 0.3084 | 0.7992 | 0.3412 | 0.2773 |

The learned guard has only a very small event-scope precision@B advantage over completion-only: 0.0507 to 0.0513. That small gain does not transfer into better changed-edge/add behavior. `product` gives the best proposal changed-edge recall by a tiny margin, but downstream it is still slightly worse than `completion_only`.

## Clean Result

Run: clean proposal + W012 on `graph_event_test.pkl`.

| Mode | Edge recall | Out-of-scope miss | Event-scope precision@B | Changed precision@B | False-scope frac | Changed-edge | Context-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.2011 | 0.7989 | NA | NA | NA | 0.0728 | 0.9828 | 0.0220 | 0.1208 |
| completion_only | 0.2949 | 0.7051 | 0.1201 | 0.0278 | 0.8799 | 0.1049 | 0.9707 | 0.0455 | 0.1611 |
| guard_only | 0.2956 | 0.7044 | 0.1169 | 0.0280 | 0.8831 | 0.1056 | 0.9713 | 0.0499 | 0.1583 |
| product | 0.2953 | 0.7047 | 0.1187 | 0.0279 | 0.8813 | 0.1056 | 0.9712 | 0.0484 | 0.1597 |
| guard_then_completion | 0.2956 | 0.7044 | 0.1169 | 0.0280 | 0.8831 | 0.1056 | 0.9713 | 0.0499 | 0.1583 |
| oracle_guard_then_completion | 0.7753 | 0.2247 | 0.4122 | 0.1699 | 0.5878 | 0.4379 | 0.9729 | 0.6012 | 0.2833 |
| naive closure | 0.8231 | 0.1769 | NA | NA | NA | 0.2846 | 0.8461 | 0.3856 | 0.1889 |

Clean has a tiny guard-only lift in changed-edge/add, but it is far too small to justify changing the branch. The oracle upper bound again shows that a good event-scope primary signal would be valuable.

## Top-Tail Diagnostics

### Noisy calibrated P2 + RFT1

| Mode | Event-scope AP | Event-scope AUROC | Changed AP | Changed AUROC | Event precision@B | Changed precision@B | Event recall@B | Changed recall@B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| completion_only | 0.0424 | 0.7024 | 0.0287 | 0.7141 | 0.0507 | 0.0325 | 0.2439 | 0.2383 |
| guard_only | 0.0415 | 0.7015 | 0.0285 | 0.7126 | 0.0513 | 0.0323 | 0.2471 | 0.2363 |
| product | 0.0423 | 0.7035 | 0.0288 | 0.7150 | 0.0509 | 0.0328 | 0.2451 | 0.2403 |
| guard_then_completion | 0.0415 | 0.7015 | 0.0285 | 0.7126 | 0.0513 | 0.0323 | 0.2471 | 0.2363 |
| oracle_guard_then_completion | 1.0000 | 1.0000 | 0.6687 | 0.9968 | 0.2006 | 0.1320 | 0.9657 | 0.9666 |

The learned guard is not outperforming completion-only in AP/AUROC. It has a tiny precision@B gain for event-scope, but it gives back changed-edge precision and recall. This points to weak top-tail precision, not merely product-rule waste.

中文解释: guard 的总体排序信号不比 completion 更好。它能稍微多抓一点 event-scope context，但没有更好地抓 changed edge，所以对下游 edit 指标没有帮助。

## Event-Type Breakdown

### Noisy calibrated P2 + RFT1

| Event type | Mode | Edge recall | Event precision@B | Changed precision@B | False-scope frac | Changed-edge | Context-edge | Add | Delete |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| edge_add | completion_only | 0.3933 | 0.0949 | 0.0869 | 0.9051 | 0.3121 | 0.8498 | 0.2815 | 0.5128 |
| edge_add | guard_only | 0.3921 | 0.0944 | 0.0863 | 0.9056 | 0.3087 | 0.8491 | 0.2781 | 0.5096 |
| edge_add | product | 0.3955 | 0.0963 | 0.0880 | 0.9037 | 0.3113 | 0.8493 | 0.2810 | 0.5096 |
| edge_add | guard_then_completion | 0.3921 | 0.0944 | 0.0863 | 0.9056 | 0.3083 | 0.8491 | 0.2776 | 0.5096 |
| edge_delete | completion_only | 0.7805 | 0.0304 | 0.0246 | 0.9696 | 0.4219 | 0.8509 | 0.2692 | 0.4440 |
| edge_delete | guard_only | 0.7830 | 0.0324 | 0.0258 | 0.9676 | 0.4239 | 0.8511 | 0.2853 | 0.4440 |
| edge_delete | product | 0.7826 | 0.0317 | 0.0256 | 0.9683 | 0.4231 | 0.8510 | 0.2853 | 0.4431 |
| edge_delete | guard_then_completion | 0.7830 | 0.0324 | 0.0258 | 0.9676 | 0.4239 | 0.8511 | 0.2853 | 0.4440 |

The guard helps edge-delete slightly, but hurts edge-add. Since edge_add remains the most proposal-limited slice, this tradeoff is not enough.

## Optional Noisy W012 Contrast

The optional W012 noisy contrast is consistent with RFT1:

| Mode | Changed-edge | Context-edge | Add | Delete |
|---|---:|---:|---:|---:|
| completion_only | 0.3547 | 0.8493 | 0.2688 | 0.4361 |
| guard_only | 0.3533 | 0.8496 | 0.2664 | 0.4356 |
| product | 0.3540 | 0.8493 | 0.2683 | 0.4352 |
| guard_then_completion | 0.3533 | 0.8496 | 0.2664 | 0.4356 |
| oracle_guard_then_completion | 0.5849 | 0.8554 | 0.7708 | 0.4088 |

No reranking mode with the learned guard beats completion-only in a useful way for W012 either.

## Decision Answers

### 1. Does any zero-retraining ranking mode materially outperform Step 9c `completion_only` at the same 10% budget?

No. In the main noisy RFT1 stack, `completion_only` remains the best practical mode overall. `product` has the highest proposal recall by a tiny amount, but downstream changed-edge/add/delete are slightly worse than `completion_only`. `guard_only` and `guard_then_completion` slightly improve event-scope precision@B and context-edge, but reduce changed-edge/add.

### 2. Does the learned guard have useful residual signal beyond the current product rule?

Only weakly, and not in a way that is operationally useful. `guard_only` improves noisy event-scope precision@B from 0.0507 to 0.0513, but changed precision@B drops from 0.0325 to 0.0323. The residual signal appears to favor some event-scope context edges, not the true changed edges needed to preserve Step 9c's edit gains.

### 3. Is Step 11's failure mainly a ranking-combination problem, or true top-tail precision weakness?

It is mainly true top-tail precision weakness. The learned guard does not dominate completion-only in AP, AUROC, precision@B, or downstream metrics. Changing the combination rule does not unlock a hidden strong mode.

### 4. What should the next training line be?

The next training line should be a stronger precision-focused guard objective / hard-negative guard, not a pure inference/reranking change with the current checkpoints.

Reason: oracle event-scope primary ranking is still excellent, so the target is valuable. But the current learned guard cannot reproduce that target in the top 10% budget. A better objective should focus on the top-tail false-scope negatives that currently dominate rescued admissions.

## Stable Defaults

Stable defaults remain unchanged:

| Role | Current choice |
|---|---|
| broad clean default | W012 |
| noisy proposal front-end | calibrated P2, node threshold 0.15, edge threshold 0.10 |
| noisy broad default | RFT1 + calibrated P2 |
| proposal-side branch candidate | Step 9c fixed-budget internal completion at 10% |
| Step 12 result | informative negative for zero-training reranking |

