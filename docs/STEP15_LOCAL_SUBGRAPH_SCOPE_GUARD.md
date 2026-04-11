# Step 15: Local-Subgraph Scope Guard

## Purpose

Step 15 tests the last minimal representation jump for the internal edge-completion guard line.

The fixed question:

> Can a minimally richer learned local-subgraph guard convert the Step 14 admission-quality signal into a real system-level gain at the same fixed 10% rescue budget?

The project scope is unchanged: structured synthetic graph-event world modeling with learned-scope proposal followed by oracle-local rewrite. No rewrite retraining, proposal backbone retraining, threshold sweeps, budget sweeps, new guard-loss variants, real-world data, perception modules, hypergraphs, or LLM components were added.

## Mechanism

The only new mechanism variable is:

| Variable | Compared values |
|---|---|
| `guard_representation` | `current_small_scalar_guard` vs `learned_local_subgraph_guard` |

The Step 15 guard keeps the same target as Step 11-14:

- positive target: GT event-scope membership among internal rescue candidates
- not changed-edge-only

Internal rescue candidates remain:

- both endpoints already inside predicted node scope
- base proposal edge scope does not already include the edge

Inference remains:

- fixed `rescue_budget_fraction = 0.10`
- rank internal candidates by guard score
- admit top-ranked candidates up to budget
- once admitted, pass the unchanged Step 9 completion confidence into rewrite

中文说明：Step 15 只改变 guard 的候选边表示，不改变 proposal backbone、不改变 completion scorer、不改变 rewrite、不调 budget。

## Local-Subgraph Guard Representation

The Step 15 guard is a tiny candidate-centric local encoder over frozen proposal outputs. It uses:

- endpoint proposal latents
- endpoint-neighborhood pooled latents inside predicted node scope
- common-neighbor pooled latents inside predicted node scope
- current observed edge bit
- base proposal edge score
- Step 9 completion score
- endpoint node-scope scores
- local degree and scope-density summaries

The proposal backbone and Step 9 completion head are frozen. Only the local-subgraph guard head is trained with vanilla BCE.

## Files And Artifacts

Scripts:

- `train/train_step15_local_subgraph_scope_guard.py`
- `train/eval_step15_local_subgraph_scope_guard.py`

Checkpoints:

- `checkpoints/step15_local_subgraph_guard_clean/best.pt`
- `checkpoints/step15_local_subgraph_guard_noisy_p2/best.pt`

Metrics:

- `artifacts/step15_local_subgraph_scope_guard/clean_w012.json`
- `artifacts/step15_local_subgraph_scope_guard/clean_w012.csv`
- `artifacts/step15_local_subgraph_scope_guard/noisy_p2_rft1.json`
- `artifacts/step15_local_subgraph_scope_guard/noisy_p2_rft1.csv`
- `artifacts/step15_local_subgraph_scope_guard/noisy_p2_w012.json`
- `artifacts/step15_local_subgraph_scope_guard/noisy_p2_w012.csv`

Compile check passed for the new Step 15 train/eval scripts and reused proposal/rewrite modules.

## Training Summary

| Stack | Best epoch | Best val precision@B | Notes |
|---|---:|---:|---|
| clean proposal + W012 | 5 | 0.1285 | Below Step 14 clean enriched val precision `0.1342` |
| noisy calibrated P2 + RFT1 | 5 | 0.0645 | Essentially tied with Step 14 noisy enriched val precision `0.0648` |

The validation result did not show a decisive local-subgraph advantage over Step 14 enriched frozen features.

## Main Noisy Result: Calibrated P2 + RFT1

| Mode | Edge recall | Out-of-scope miss | Event precision@B | Rescued false-scope frac | Context-edge | Changed-edge | Add | Delete | Full-edge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base proposal | 0.4810 | 0.5190 | NA | NA | 0.8836 | 0.2710 | 0.0958 | 0.4370 | 0.8742 |
| Step 9c completion_only | 0.5922 | 0.4078 | 0.0507 | 0.9493 | 0.8482 | 0.3650 | 0.2815 | 0.4440 | 0.8409 |
| Step 14 enriched probe | 0.6001 | 0.3999 | 0.0625 | 0.9375 | 0.8486 | 0.3709 | 0.2859 | 0.4514 | 0.8413 |
| Step 15 local-subgraph guard | 0.5974 | 0.4026 | 0.0613 | 0.9387 | 0.8491 | 0.3683 | 0.2791 | 0.4528 | 0.8418 |
| Oracle event-scope guard @ 10% | 0.9324 | 0.0676 | 0.2006 | 0.7994 | 0.8543 | 0.6051 | 0.8069 | 0.4139 | 0.8505 |

Interpretation:

- Step 15 beats Step 9c on the intended precision axis: event precision@B `0.0507 -> 0.0613`.
- Step 15 does not beat Step 14 enriched: event precision@B `0.0625 -> 0.0613`.
- Step 15 recovers slightly more context-edge than Step 14: `0.8486 -> 0.8491`.
- Step 15 gives back changed-edge and add versus Step 14: changed-edge `0.3709 -> 0.3683`, add `0.2859 -> 0.2791`.
- Step 15 slightly improves delete: `0.4514 -> 0.4528`.

So the richer local-subgraph encoder is not a decisive system-level improvement over the simpler Step 14 enriched probe.

## Event-Type Breakdown

### Noisy P2 + RFT1

| Event type | Mode | Edge recall | Event P@B | False-scope frac | Context-edge | Changed-edge | Add | Delete |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| edge_add | Step 9c completion_only | 0.3933 | 0.0949 | 0.9051 | 0.8498 | 0.3121 | 0.2815 | 0.5128 |
| edge_add | Step 14 enriched | 0.3902 | 0.0971 | 0.9029 | 0.8499 | 0.3172 | 0.2859 | 0.5224 |
| edge_add | Step 15 local-subgraph | 0.3851 | 0.0953 | 0.9047 | 0.8500 | 0.3104 | 0.2791 | 0.5160 |
| edge_delete | Step 9c completion_only | 0.7805 | 0.0304 | 0.9696 | 0.8509 | 0.4219 | 0.2692 | 0.4440 |
| edge_delete | Step 14 enriched | 0.8008 | 0.0449 | 0.9551 | 0.8518 | 0.4296 | 0.2788 | 0.4514 |
| edge_delete | Step 15 local-subgraph | 0.8008 | 0.0449 | 0.9551 | 0.8525 | 0.4300 | 0.2724 | 0.4528 |

The best Step 15 slice is noisy `edge_delete`: it matches Step 14 on event precision@B/recall while slightly improving context-edge, changed-edge, and delete. But it does not improve `edge_add`, which was one of the motivating slices for internal completion.

## Clean Result: Clean Proposal + W012

| Mode | Edge recall | Out-of-scope miss | Event precision@B | Rescued false-scope frac | Context-edge | Changed-edge | Add | Delete | Full-edge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base proposal | 0.2011 | 0.7989 | NA | NA | 0.9828 | 0.0728 | 0.0220 | 0.1208 | 0.9689 |
| Step 9c completion_only | 0.2949 | 0.7051 | 0.1201 | 0.8799 | 0.9707 | 0.1049 | 0.0455 | 0.1611 | 0.9575 |
| Step 14 enriched probe | 0.2874 | 0.7126 | 0.1234 | 0.8766 | 0.9733 | 0.1006 | 0.0440 | 0.1542 | 0.9600 |
| Step 15 local-subgraph guard | 0.2935 | 0.7065 | 0.1200 | 0.8800 | 0.9711 | 0.1113 | 0.0616 | 0.1583 | 0.9580 |
| Oracle event-scope guard @ 10% | 0.7753 | 0.2247 | 0.4122 | 0.5878 | 0.9729 | 0.4379 | 0.6012 | 0.2833 | 0.9647 |

Clean Step 15 improves changed-edge and add relative to Step 14 enriched, but loses on event precision@B, false-scope fraction, and context-edge. It behaves more like a slightly edit-heavier tradeoff than a cleaner guard.

## Optional Noisy W012 Contrast

| Mode | Context-edge | Changed-edge | Add | Delete | Full-edge |
|---|---:|---:|---:|---:|---:|
| base proposal | 0.8850 | 0.2622 | 0.0894 | 0.4259 | 0.8755 |
| Step 9c completion_only | 0.8493 | 0.3547 | 0.2688 | 0.4361 | 0.8418 |
| Step 14 enriched probe | 0.8500 | 0.3602 | 0.2722 | 0.4435 | 0.8426 |
| Step 15 local-subgraph guard | 0.8504 | 0.3578 | 0.2654 | 0.4454 | 0.8429 |

The optional W012 contrast matches the RFT1 conclusion: Step 15 gains a little stability/context but gives back some edit-sensitive add/changed behavior compared with Step 14.

## Context Decomposition

### Noisy P2 + RFT1

| Mode | Extra context errors | False-scope error share | True-scope-context error share | Spillover share |
|---|---:|---:|---:|---:|
| Step 9c completion_only | 19176 | 0.8958 | 0.0019 | 0.0000 |
| Step 14 enriched probe | 18956 | 0.8462 | 0.0052 | 0.0000 |
| Step 15 local-subgraph guard | 18690 | 0.8848 | 0.0056 | 0.0000 |

Step 15 reduces total extra context errors versus Step 14, but a larger share again comes from false-scope rescued edges. The same Step 10 diagnosis remains true: context cost is still dominated by false-scope admissions, not broad rewrite spillover.

## Decision Answers

### 1. Does the learned local-subgraph guard materially beat Step 14 enriched at fixed 10% budget?

No.

On noisy P2 + RFT1:

- event precision@B is slightly worse: `0.0625 -> 0.0613`
- rescued false-scope fraction is slightly worse: `0.9375 -> 0.9387`
- downstream context-edge is slightly better: `0.8486 -> 0.8491`

The context-edge improvement is small and comes with losses in changed-edge and add.

### 2. Does it materially beat Step 9c on proposal precision without giving back too much changed-edge / add?

It beats Step 9c on proposal precision, but not enough to become a clear win.

Compared with Step 9c on noisy P2 + RFT1:

- event precision@B improves: `0.0507 -> 0.0613`
- false-scope fraction improves: `0.9493 -> 0.9387`
- changed-edge improves: `0.3650 -> 0.3683`
- delete improves: `0.4440 -> 0.4528`
- add decreases slightly: `0.2815 -> 0.2791`
- context-edge improves slightly: `0.8482 -> 0.8491`

This is a real but modest proposal-side precision improvement over Step 9c. It is not a decisive system-level improvement.

### 3. Is it strong enough to replace Step 14 probe as the active guard line?

No.

Step 14 enriched remains the cleaner active guard/probe reference because it has the stronger top-tail admission quality and better edit-sensitive noisy gains:

- Step 14 event precision@B: `0.0625`
- Step 15 event precision@B: `0.0613`
- Step 14 changed-edge: `0.3709`
- Step 15 changed-edge: `0.3683`
- Step 14 add: `0.2859`
- Step 15 add: `0.2791`

Step 15 is useful evidence, but not a branch promotion.

### 4. If it still fails, does the guard line now require a larger representation jump than justified for the current phase?

Yes, for the current phase.

Step 15 was intentionally the last minimal representation jump: endpoint neighborhoods, common-neighbor pooling, induced local structure, and frozen proposal/completion signals. It did not materially beat Step 14. The oracle guard gap remains large, but the next improvement likely requires either a substantially stronger local-subgraph/pair representation or a different proposal-side design. That is bigger than the current minimal branch budget.

中文结论：更丰富的小型 local-subgraph guard 没有把 Step 14 的信号转成明确系统收益。现在继续微调 guard loss 或小幅加特征，性价比不高。稳定默认保持不变，Step 9c/14 作为 proposal-side 分支候选保留即可。

## Recommendation

Keep stable defaults unchanged:

- clean broad default: W012
- noisy broad default: RFT1 + calibrated P2
- active proposal-side branch candidate: Step 9c fixed-budget internal completion @ 10%
- guard/probe reference: Step 14 enriched local-context probe

Do not promote Step 15. Treat it as evidence that the minimal guard-representation path has largely been exhausted for now.

