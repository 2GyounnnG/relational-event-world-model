# Step 13: Tail-Focused Scope Guard

## Scope

Step 13 tests one new mechanism variable:

| Variable | Values |
|---|---|
| `guard_objective` | `vanilla_scope_bce`, `tail_focused_hard_negative_scope_bce` |

Everything else is fixed:

| Setting | Value |
|---|---:|
| rescue budget fraction | 0.10 |
| tail fraction | 0.10 |
| completion scorer | existing Step 9 head |
| guard architecture | existing Step 11 small guard |
| proposal backbone | frozen |
| completion head | frozen |
| rewrite model | unchanged |
| inference ranking | `completion_score * guard_score` |

Tail-focused objective:

```text
loss = BCE(all internal candidates)
     + BCE(top 10% internal candidates ranked by frozen completion score)
```

The target remains GT event-scope membership, not changed-edge-only membership.

中文备注: 这一步只改 guard 的训练目标，专门压实际 10% rescue 操作区间里的 hard negatives；不改模型结构、不改 budget、不训练 rewrite。

## Files And Artifacts

| Item | Path |
|---|---|
| trainer | `train/train_step13_tailfocused_scope_guard.py` |
| evaluator | `train/eval_step13_tailfocused_guard.py` |
| clean tail guard | `checkpoints/step13_tailfocused_scope_guard_clean/best.pt` |
| noisy tail guard | `checkpoints/step13_tailfocused_scope_guard_noisy_p2/best.pt` |
| clean metrics | `artifacts/step13_tailfocused_scope_guard/clean_w012.json` |
| noisy RFT1 metrics | `artifacts/step13_tailfocused_scope_guard/noisy_p2_rft1.json` |
| optional noisy W012 metrics | `artifacts/step13_tailfocused_scope_guard/noisy_p2_w012.json` |

## Training Summary

| Stack | Best epoch | Selection metric | Val event precision@B | Val event recall@B | Val guard AP | Val guard AUROC |
|---|---:|---|---:|---:|---:|---:|
| clean proposal + clean completion | 2 | event precision@B | 0.1245 | 0.2665 | 0.1123 | 0.7825 |
| noisy P2 + noisy completion | 3 | event precision@B | 0.0541 | 0.2659 | 0.0421 | 0.7024 |

The tail objective improves the validation budget-tail precision signal relative to the Step 11 noisy guard line, but the test set is the real decision point.

## Main Noisy Result: Calibrated P2 + RFT1

| Mode | Edge recall | Out-of-scope miss | Event precision@B | Event recall@B | False-scope frac | Changed frac | Context frac | Changed-edge | Context-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.4810 | 0.5190 | NA | NA | NA | NA | NA | 0.2710 | 0.8836 | 0.0958 | 0.4370 |
| Step 9c completion-only | 0.5922 | 0.4078 | 0.0507 | 0.2439 | 0.9493 | 0.0325 | 0.0181 | 0.3650 | 0.8482 | 0.2815 | 0.4440 |
| Step 11 vanilla guard | 0.5932 | 0.4068 | 0.0509 | 0.2451 | 0.9491 | 0.0328 | 0.0181 | 0.3642 | 0.8481 | 0.2810 | 0.4431 |
| Step 13 tail guard | 0.5893 | 0.4107 | 0.0505 | 0.2429 | 0.9495 | 0.0317 | 0.0188 | 0.3645 | 0.8481 | 0.2801 | 0.4444 |
| oracle event-scope guard | 0.9324 | 0.0676 | 0.2006 | 0.9657 | 0.7994 | 0.1320 | 0.0686 | 0.6051 | 0.8543 | 0.8069 | 0.4139 |
| naive closure | 0.9479 | 0.0521 | NA | NA | NA | NA | NA | 0.3084 | 0.7992 | 0.3412 | 0.2773 |

Result: the tail-focused guard does not improve test false-scope precision. It slightly worsens false-scope fraction and gives back changed-edge recall compared with both Step 9c and Step 11.

## Clean Result: Clean Proposal + W012

| Mode | Edge recall | Out-of-scope miss | Event precision@B | Event recall@B | False-scope frac | Changed frac | Context frac | Changed-edge | Context-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.2011 | 0.7989 | NA | NA | NA | NA | NA | 0.0728 | 0.9828 | 0.0220 | 0.1208 |
| Step 9c completion-only | 0.2949 | 0.7051 | 0.1201 | 0.2571 | 0.8799 | 0.0278 | 0.0923 | 0.1049 | 0.9707 | 0.0455 | 0.1611 |
| Step 11 vanilla guard | 0.2953 | 0.7047 | 0.1187 | 0.2542 | 0.8813 | 0.0279 | 0.0909 | 0.1056 | 0.9712 | 0.0484 | 0.1597 |
| Step 13 tail guard | 0.2978 | 0.7022 | 0.1173 | 0.2512 | 0.8827 | 0.0286 | 0.0887 | 0.1077 | 0.9708 | 0.0469 | 0.1653 |
| oracle event-scope guard | 0.7753 | 0.2247 | 0.4122 | 0.8825 | 0.5878 | 0.1699 | 0.2423 | 0.4379 | 0.9729 | 0.6012 | 0.2833 |
| naive closure | 0.8231 | 0.1769 | NA | NA | NA | NA | NA | 0.2846 | 0.8461 | 0.3856 | 0.1889 |

Clean gets a small changed-edge/delete improvement, but it still does not reduce false-scope fraction. This is not the primary noisy target and is not enough to promote the line.

## Event-Type Breakdown

### Noisy calibrated P2 + RFT1

| Event type | Mode | Edge recall | Event precision@B | False-scope frac | Changed frac | Downstream changed | Context | Add | Delete |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| edge_add | Step 9c | 0.3933 | 0.0949 | 0.9051 | 0.0869 | 0.3121 | 0.8498 | 0.2815 | 0.5128 |
| edge_add | Step 11 | 0.3955 | 0.0963 | 0.9037 | 0.0880 | 0.3113 | 0.8493 | 0.2810 | 0.5096 |
| edge_add | Step 13 | 0.3880 | 0.0928 | 0.9072 | 0.0842 | 0.3104 | 0.8495 | 0.2801 | 0.5096 |
| edge_delete | Step 9c | 0.7805 | 0.0304 | 0.9696 | 0.0246 | 0.4219 | 0.8509 | 0.2692 | 0.4440 |
| edge_delete | Step 11 | 0.7826 | 0.0317 | 0.9683 | 0.0256 | 0.4231 | 0.8510 | 0.2853 | 0.4431 |
| edge_delete | Step 13 | 0.7814 | 0.0321 | 0.9679 | 0.0250 | 0.4235 | 0.8510 | 0.2788 | 0.4444 |

Tail focusing helps the edge-delete event-scope precision slice a little, but it hurts edge_add, which remains the most important proposal-limited slice.

## Context Decomposition

### Noisy calibrated P2 + RFT1

| Mode | False-scope extra context errors | True-scope context extra errors | Spillover drop | False-scope error share |
|---|---:|---:|---:|---:|
| Step 9c | 17,178 | 37 | 0.0000 | 0.8958 |
| Step 11 | 17,212 | 33 | 0.0000 | 0.8953 |
| Step 13 | 17,204 | 35 | 0.0000 | 0.8946 |

The Step 10 diagnosis remains intact: context loss still comes overwhelmingly from rescued false-scope edges. Step 13 does not reduce that source enough to matter.

## Optional Noisy W012 Contrast

The optional W012 noisy run matches the RFT1 conclusion.

| Mode | Event precision@B | False-scope frac | Changed-edge | Context-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|---:|
| Step 9c | 0.0507 | 0.9493 | 0.3547 | 0.8493 | 0.2688 | 0.4361 |
| Step 11 | 0.0509 | 0.9491 | 0.3540 | 0.8493 | 0.2683 | 0.4352 |
| Step 13 | 0.0505 | 0.9495 | 0.3538 | 0.8494 | 0.2674 | 0.4356 |
| oracle guard | 0.2006 | 0.7994 | 0.5849 | 0.8554 | 0.7708 | 0.4088 |

## Decision Answers

### 1. Does the tail-focused hard-negative guard materially reduce rescued false-scope fraction at the same fixed 10% budget?

No. In the main noisy stack, false-scope fraction is:

| Mode | False-scope fraction |
|---|---:|
| Step 9c | 0.9493 |
| Step 11 | 0.9491 |
| Step 13 | 0.9495 |

Step 13 slightly worsens the overall false-scope fraction on test.

### 2. Does it recover context-edge stability while preserving most changed-edge/add gains?

No. Context-edge remains essentially unchanged and still far below base:

| Mode | Context-edge | Changed-edge | Add |
|---|---:|---:|---:|
| base | 0.8836 | 0.2710 | 0.0958 |
| Step 9c | 0.8482 | 0.3650 | 0.2815 |
| Step 13 | 0.8481 | 0.3645 | 0.2801 |

It preserves most Step 9c behavior, but only because it does not meaningfully change the rescued set.

### 3. Is it strong enough to replace Step 11 as the active guard line, while keeping stable defaults unchanged?

No. Step 13 is not better than Step 11 on the main noisy objective, and neither is better than Step 9c as a practical branch. Stable defaults remain unchanged.

### 4. If it still fails, is the bottleneck guard representation weakness or insufficiently targeted objective design?

The evidence points more toward guard representation weakness, or at least toward this small guard architecture being unable to express the event-scope boundary needed in the top tail. The objective was budget-tail-aligned and improved validation precision@B, but did not generalize to test. The oracle guard remains very strong, so the target is still meaningful; this particular minimal objective is not enough.

中文解释: 这不是“目标完全错了”，而是当前小 guard 的表达/泛化能力不够。hard-negative tail BCE 没能把 top 10% 里最致命的 false-scope 边真正排出去。

## Stable Defaults

Do not promote Step 13.

| Role | Current choice |
|---|---|
| broad clean default | W012 |
| noisy proposal front-end | calibrated P2, node threshold 0.15, edge threshold 0.10 |
| noisy broad default | RFT1 + calibrated P2 |
| proposal-side branch candidate | Step 9c fixed-budget internal completion at 10% |
| Step 13 status | informative negative for minimal tail-focused guard objective |

## Minimal Next Read

If continuing this mechanism family, the next move should not be another tiny objective tweak on the same guard head. The more plausible next line is a stronger internal candidate representation or hard-negative guard with richer pair features, while keeping the current defaults unchanged until it clearly beats Step 9c under noisy evaluation.

