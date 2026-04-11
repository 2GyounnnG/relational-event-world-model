# Step 11: Guarded Internal Edge Completion

## Scope

This is a small proposal-side evaluation/training pack inside the structured synthetic graph-event world. It does not change the rewrite model, does not retrain Step 2-6 defaults, and does not sweep budgets or thresholds.

Mechanism variable:

| Variable | Values |
|---|---|
| `guard_mode` | `off`, `learned_event_scope_guard` |

Fixed operating point:

| Setting | Value |
|---|---:|
| rescue budget fraction | 0.10 |
| clean base proposal threshold | node 0.20, edge 0.15 |
| noisy P2 threshold | node 0.15, edge 0.10 |
| completion scorer | existing Step 9 completion head |

The guard is trained only on internal rescue candidates:

1. both endpoints are already inside predicted node scope
2. the base proposal edge scope does not already include the edge

The guard target is GT event-scope edge membership, not changed-edge-only membership. Inference ranks candidates by:

```text
combined_score = completion_score * guard_score
```

Selected guarded edges keep the completion confidence as the edge proposal probability passed to rewrite. This keeps Step 11 focused on admission/ranking, not on damping rewrite confidence.

中文备注: 这一步只测试"能不能少放进假 scope 边", 不改变 rewrite，也不重新调旧阈值。

## Files And Artifacts

| Item | Path |
|---|---|
| guard trainer | `train/train_step11_internal_scope_guard.py` |
| guarded evaluator | `train/eval_step11_guarded_internal_completion.py` |
| clean guard checkpoint | `checkpoints/step11_scope_guard_clean/best.pt` |
| noisy guard checkpoint | `checkpoints/step11_scope_guard_noisy_p2/best.pt` |
| clean metrics | `artifacts/step11_guarded_internal_completion/clean_w012.json` |
| noisy RFT1 metrics | `artifacts/step11_guarded_internal_completion/noisy_p2_rft1.json` |
| optional noisy W012 metrics | `artifacts/step11_guarded_internal_completion/noisy_p2_w012.json` |

## Guard Training Summary

| Stack | Train data | Best epoch | Val AP | Val AUROC | Precision@0.5 | Recall@0.5 |
|---|---|---:|---:|---:|---:|---:|
| clean proposal + clean completion | `graph_event_train/val` | 3 | 0.1157 | 0.7858 | 0.1291 | 0.3487 |
| noisy P2 + noisy completion | `graph_event_step6a_train/val` | 2 | 0.0460 | 0.7154 | 0.0891 | 0.0360 |

Interpretation: the guard has some ranking signal, especially AUROC, but AP is low because event-scope positives among internal candidates are very sparse. The noisy guard in particular is too weak at the raw classifier level.

## Proposal-Side Overall Results

### Clean: proposal + W012

| Mode | Edge recall | Out-of-scope miss | Pred edge scope size | Recall recovery vs naive | Cost fraction vs naive |
|---|---:|---:|---:|---:|---:|
| base | 0.2011 | 0.7989 | 14,150 | NA | NA |
| Step 9c budget 10% | 0.2949 | 0.7051 | 23,627 | 0.1508 | 0.0922 |
| Step 11 guarded budget 10% | 0.2953 | 0.7047 | 23,627 | 0.1514 | 0.0922 |
| oracle event-scope guard | 0.7753 | 0.2247 | 18,056 | 0.9232 | 0.0380 |
| naive induced closure | 0.8231 | 0.1769 | 116,960 | NA | NA |

### Noisy: calibrated P2 + RFT1

| Mode | Edge recall | Out-of-scope miss | Pred edge scope size | Recall recovery vs naive | Cost fraction vs naive |
|---|---:|---:|---:|---:|---:|
| base | 0.4810 | 0.5190 | 146,566 | NA | NA |
| Step 9c budget 10% | 0.5922 | 0.4078 | 175,326 | 0.2383 | 0.0924 |
| Step 11 guarded budget 10% | 0.5932 | 0.4068 | 175,326 | 0.2403 | 0.0924 |
| oracle event-scope guard | 0.9324 | 0.0676 | 152,335 | 0.9666 | 0.0185 |
| naive induced closure | 0.9479 | 0.0521 | 457,934 | NA | NA |

The learned guard barely changes recall or miss at the fixed 10% budget. The oracle event-scope-filtered upper bound is extremely strong, which confirms the mechanism target is meaningful, but the current learned guard does not approximate it well enough.

## Rescue Composition

### Clean: proposal + W012

| Mode | True changed fraction | True-scope context fraction | False-scope fraction |
|---|---:|---:|---:|
| Step 9c budget 10% | 0.0278 | 0.0923 | 0.8799 |
| Step 11 guarded budget 10% | 0.0279 | 0.0909 | 0.8813 |
| oracle event-scope guard | 0.4122 | 0.5878 | 0.0000 |

### Noisy: calibrated P2 + RFT1

| Mode | True changed fraction | True-scope context fraction | False-scope fraction |
|---|---:|---:|---:|
| Step 9c budget 10% | 0.0325 | 0.0181 | 0.9493 |
| Step 11 guarded budget 10% | 0.0328 | 0.0181 | 0.9491 |
| oracle event-scope guard | 0.6582 | 0.3418 | 0.0000 |

Decision-relevant result: learned guard reduces noisy false-scope rescue from 0.94934 to 0.94910, effectively no material reduction. It admits 8 additional GT-changed rescued edges overall, but does not change the false-scope problem.

中文解释: 当前 guard 不是完全没信号，但它没有把"假 scope 边"明显挡掉。真正的 oracle guard 可以同时大幅提高 recall 并压低成本，说明方向对，当前小 guard 还不够强。

## Event-Type Slice

### Noisy calibrated P2 + RFT1

| Event type | Mode | Edge recall | Out-of-scope miss | False-scope fraction | Changed-edge | Context-edge | Add | Delete |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| edge_add | Step 9c budget 10% | 0.3933 | 0.6067 | 0.9051 | 0.3121 | 0.8498 | 0.2815 | 0.5128 |
| edge_add | Step 11 guarded budget 10% | 0.3955 | 0.6045 | 0.9037 | 0.3113 | 0.8493 | 0.2810 | 0.5096 |
| edge_delete | Step 9c budget 10% | 0.7805 | 0.2195 | 0.9696 | 0.4219 | 0.8509 | 0.2692 | 0.4440 |
| edge_delete | Step 11 guarded budget 10% | 0.7826 | 0.2174 | 0.9683 | 0.4231 | 0.8510 | 0.2853 | 0.4431 |

The guard gives a tiny proposal-side lift in both edge-add and edge-delete recall. Downstream effects are mixed and very small: edge-delete changed-edge improves slightly, but edge-add changed-edge and add accuracy drop slightly.

## Downstream Overall Results

### Clean: proposal + W012

| Mode | Full-edge | Changed-edge | Context-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base | 0.9689 | 0.0728 | 0.9828 | 0.0220 | 0.1208 |
| Step 9c budget 10% | 0.9575 | 0.1049 | 0.9707 | 0.0455 | 0.1611 |
| Step 11 guarded budget 10% | 0.9580 | 0.1056 | 0.9712 | 0.0484 | 0.1597 |
| oracle event-scope guard | 0.9725 | 0.4379 | 0.9808 | 0.6012 | 0.2833 |
| naive induced closure | 0.8375 | 0.2846 | 0.8461 | 0.3856 | 0.1889 |

### Noisy: calibrated P2 + RFT1

| Mode | Full-edge | Changed-edge | Context-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base | 0.8742 | 0.2710 | 0.8836 | 0.0958 | 0.4370 |
| Step 9c budget 10% | 0.8409 | 0.3650 | 0.8482 | 0.2815 | 0.4440 |
| Step 11 guarded budget 10% | 0.8408 | 0.3642 | 0.8481 | 0.2810 | 0.4431 |
| oracle event-scope guard | 0.8804 | 0.6051 | 0.8847 | 0.8069 | 0.4139 |
| naive induced closure | 0.7917 | 0.3084 | 0.7992 | 0.3412 | 0.2773 |

The learned guard does not recover context-edge stability in the noisy default stack. It slightly underperforms Step 9c on changed-edge, context-edge, add, and delete, despite a tiny proposal recall increase.

## Context-Source Decomposition

### Noisy calibrated P2 + RFT1

| Mode | False-scope extra context error share | True-scope context extra share | Base-scope spillover drop |
|---|---:|---:|---:|
| Step 9c budget 10% | 0.8958 | 0.0019 | 0.0000 |
| Step 11 guarded budget 10% | 0.8953 | 0.0017 | 0.0000 |

The Step 10 diagnosis remains unchanged: context loss is overwhelmingly from false-scope rescued edges. Step 11 does not reduce that source enough to matter.

## Decision Answers

### 1. Does the learned event-scope guard materially reduce rescued false-scope fraction at the same 10% budget?

No. In the noisy default stack, false-scope fraction changes from 0.94934 to 0.94910. This is directionally correct but far too small to matter. Clean is slightly worse: 0.87992 to 0.88129.

### 2. Does it preserve most of Step 9c's edit-sensitive gain while recovering context-edge stability?

It preserves most of the Step 9c behavior because it barely changes the selected set. It does not recover context stability. Noisy context-edge changes from 0.84823 to 0.84814, while Step 9c already gave up substantial context relative to base 0.88356.

### 3. Is it strong enough to become the next main proposal-side branch candidate, while leaving stable defaults unchanged?

No. Step 9c remains the stronger branch candidate than this guarded variant. Stable defaults remain unchanged: noisy broad default is still RFT1 + calibrated P2, and guarded internal completion should not replace it.

### 4. If it fails, what is the remaining problem?

The failure is mainly insufficient guard precision / weak event-scope separability among internal candidates. It is not insufficient rescue recall among true-scope edges in the abstract, because the oracle event-scope guard upper bound is excellent. It is also not primarily a rewrite utilization issue: the oracle guard proves rewrite can use correctly admitted internal edges well.

## Current Interpretation

The Step 11 guard target is correct, but this small guard head is not yet strong enough. The gap between learned guard and oracle guard is the important signal:

| Stack | Learned guarded recall recovery | Oracle guarded recall recovery | Learned cost | Oracle cost |
|---|---:|---:|---:|---:|
| clean | 0.1514 | 0.9232 | 0.0922 | 0.0380 |
| noisy RFT1 | 0.2403 | 0.9666 | 0.0924 | 0.0185 |

This says proposal-side precision guarding remains the right mechanism family, but the current guard is too weak. A stronger event-scope guard or guard-completion joint ranking objective is more justified than moving to rewrite-side don't-touch gating at this point.

## Stable Defaults

Do not promote Step 11 to default.

| Role | Current choice |
|---|---|
| broad clean default | W012 |
| noisy proposal front-end | calibrated P2, node threshold 0.15, edge threshold 0.10 |
| noisy broad default | RFT1 + calibrated P2 |
| proposal-side branch candidate | Step 9c fixed-budget internal completion at 10% |
| Step 11 status | informative negative / guard upper-bound confirms target, learned guard too weak |

