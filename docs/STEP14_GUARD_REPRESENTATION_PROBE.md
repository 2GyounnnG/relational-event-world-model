# Step 14: Guard Representation Probe

## Purpose

Step 14 tests whether the small Step 11 / Step 13 guard line failed mainly because its candidate-edge representation was too weak.

The project scope is unchanged: structured synthetic graph-event world modeling with a proposal local region followed by oracle-local rewrite. No perception, real-world data, hypergraphs, LLMs, rewrite retraining, budget sweeps, or Step 2-6 retuning were introduced.

The only new analysis / modeling variable is:

| Variable | Values |
|---|---|
| `feature_bundle` | `scores_only`, `enriched_local_context` |

Both bundles use the same tiny candidate-level MLP probe and the same vanilla BCE event-scope target. The probe is trained only on internal rescue candidates:

- both endpoints are already inside predicted node scope
- the base proposal edge scope does not already include the edge

The operating point remains fixed at `rescue_budget_fraction = 0.10`.

中文摘要：这一步不是重训主模型，而是问一个很窄的问题：候选边的局部结构特征是否足以改善 guard 的 top-tail 精度。

## Files And Artifacts

Scripts:

- `train/train_step14_guard_representation_probe.py`
- `train/eval_step14_guard_representation_probe.py`

Probe checkpoints:

- `checkpoints/step14_guard_probe_clean_scores_only/best.pt`
- `checkpoints/step14_guard_probe_clean_enriched/best.pt`
- `checkpoints/step14_guard_probe_noisy_p2_scores_only/best.pt`
- `checkpoints/step14_guard_probe_noisy_p2_enriched/best.pt`

Metrics:

- `artifacts/step14_guard_representation_probe/clean_w012.json`
- `artifacts/step14_guard_representation_probe/clean_w012.csv`
- `artifacts/step14_guard_representation_probe/noisy_p2_rft1.json`
- `artifacts/step14_guard_representation_probe/noisy_p2_rft1.csv`
- `artifacts/step14_guard_representation_probe/noisy_p2_w012.json`
- `artifacts/step14_guard_representation_probe/noisy_p2_w012.csv`

Compile check passed for the new train/eval scripts and related proposal/rewrite modules.

## Feature Bundles

`scores_only` uses scalar signals already available from the existing system:

- completion score
- base edge proposal score
- endpoint node proposal scores
- simple endpoint score combinations
- reference small-guard score when available

`enriched_local_context` adds frozen local structured features:

- current edge-exists bit
- endpoint input node features
- endpoint feature difference and product
- endpoint degrees inside predicted node scope
- common-neighbor count inside predicted node scope
- predicted-scope density

For noisy runs, these structural features are derived from the noisy observed input graph, matching the actual model input.

## Training Summary

| Stack | Feature bundle | Best epoch | Best val precision@B |
|---|---:|---:|---:|
| clean proposal + W012 | scores_only | 2 | 0.1267 |
| clean proposal + W012 | enriched_local_context | 5 | 0.1342 |
| noisy calibrated P2 + RFT1 | scores_only | 1 | 0.0554 |
| noisy calibrated P2 + RFT1 | enriched_local_context | 5 | 0.0648 |

The validation result already showed the key signal: enriched local context improves top-tail event-scope precision in both clean and noisy settings, with the more important noisy gain moving from `0.0554` to `0.0648`.

## Main Noisy Result: Calibrated P2 + RFT1

| Mode | Edge recall | Out-of-scope miss | Event precision@B | Rescued false-scope frac | Context-edge | Changed-edge | Add | Delete | Full-edge |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base proposal | 0.4810 | 0.5190 | NA | NA | 0.8836 | 0.2710 | 0.0958 | 0.4370 | 0.8742 |
| Step 9c completion_only | 0.5922 | 0.4078 | 0.0507 | 0.9493 | 0.8482 | 0.3650 | 0.2815 | 0.4440 | 0.8409 |
| Step 11 vanilla guard product | 0.5932 | 0.4068 | 0.0509 | 0.9491 | 0.8481 | 0.3642 | 0.2810 | 0.4431 | 0.8408 |
| Step 14 scores_only probe | 0.5868 | 0.4132 | 0.0505 | 0.9495 | 0.8491 | 0.3621 | 0.2722 | 0.4472 | 0.8417 |
| Step 14 enriched_local_context probe | 0.6001 | 0.3999 | 0.0625 | 0.9375 | 0.8486 | 0.3709 | 0.2859 | 0.4514 | 0.8413 |
| Oracle event-scope guard @ 10% | 0.9324 | 0.0676 | 0.2006 | 0.7994 | 0.8543 | 0.6051 | 0.8069 | 0.4139 | 0.8505 |

Interpretation:

- `enriched_local_context` beats `scores_only` on event precision@B: `0.0625` vs `0.0505`.
- It reduces rescued false-scope fraction: `0.9375` vs `0.9495`.
- It improves changed-edge, add, and delete over Step 9c and Step 11.
- Context-edge is only marginally better than Step 9c (`0.8486` vs `0.8482`) and slightly worse than scores_only (`0.8491`).
- The oracle guard remains far stronger, so representation helps but does not close the guard gap.

中文解释：丰富局部结构特征确实让 guard 更会挑边，但 false-scope 仍然占绝大多数。这个结果支持“表示不足”这个诊断，但还不足以升级稳定默认线。

## Top-Tail Diagnostics

| Stack | Mode | Event P@B | Event R@B | Event AP | Event AUROC | Changed P@B | Changed R@B |
|---|---|---:|---:|---:|---:|---:|---:|
| clean W012 | Step 9c completion_only | 0.1201 | 0.2571 | 0.1117 | 0.7777 | 0.0278 | 0.1508 |
| clean W012 | Step 14 scores_only | 0.1193 | 0.2555 | 0.1127 | 0.7816 | 0.0308 | 0.1674 |
| clean W012 | Step 14 enriched | 0.1234 | 0.2641 | 0.1189 | 0.8017 | 0.0255 | 0.1388 |
| noisy P2 + RFT1 | Step 9c completion_only | 0.0507 | 0.2439 | 0.0424 | 0.7024 | 0.0325 | 0.2383 |
| noisy P2 + RFT1 | Step 14 scores_only | 0.0505 | 0.2432 | 0.0437 | 0.7030 | 0.0309 | 0.2266 |
| noisy P2 + RFT1 | Step 14 enriched | 0.0625 | 0.3010 | 0.0555 | 0.7366 | 0.0348 | 0.2551 |

The noisy top-tail result is the clearest Step 14 signal. Enriched local context improves event-scope AP, AUROC, precision@B, recall@B, changed precision@B, and changed recall@B.

## Event-Type Breakdown

### Noisy P2 + RFT1

| Event type | Mode | Edge recall | Event P@B | False-scope frac | Context-edge | Changed-edge | Add | Delete |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| edge_add | Step 9c completion_only | 0.3933 | 0.0949 | 0.9051 | 0.8498 | 0.3121 | 0.2815 | 0.5128 |
| edge_add | Step 14 scores_only | 0.3793 | 0.0879 | 0.9121 | 0.8501 | 0.3045 | 0.2722 | 0.5160 |
| edge_add | Step 14 enriched | 0.3902 | 0.0971 | 0.9029 | 0.8499 | 0.3172 | 0.2859 | 0.5224 |
| edge_delete | Step 9c completion_only | 0.7805 | 0.0304 | 0.9696 | 0.8509 | 0.4219 | 0.2692 | 0.4440 |
| edge_delete | Step 14 scores_only | 0.7860 | 0.0339 | 0.9661 | 0.8519 | 0.4280 | 0.2949 | 0.4472 |
| edge_delete | Step 14 enriched | 0.8008 | 0.0449 | 0.9551 | 0.8518 | 0.4296 | 0.2788 | 0.4514 |

The enriched probe is especially helpful on noisy `edge_delete`, improving event precision@B from `0.0304` to `0.0449` and edge recall from `0.7805` to `0.8008`.

### Clean W012

| Event type | Mode | Edge recall | Event P@B | False-scope frac | Context-edge | Changed-edge | Add | Delete |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| edge_add | Step 9c completion_only | 0.1260 | 0.0719 | 0.9281 | 0.9731 | 0.0611 | 0.0455 | 0.1635 |
| edge_add | Step 14 scores_only | 0.1247 | 0.0667 | 0.9333 | 0.9744 | 0.0636 | 0.0499 | 0.1538 |
| edge_add | Step 14 enriched | 0.1075 | 0.0670 | 0.9330 | 0.9761 | 0.0585 | 0.0440 | 0.1538 |
| edge_delete | Step 9c completion_only | 0.4575 | 0.1037 | 0.8963 | 0.9716 | 0.1468 | 0.0481 | 0.1611 |
| edge_delete | Step 14 scores_only | 0.4739 | 0.1134 | 0.8866 | 0.9716 | 0.1505 | 0.0481 | 0.1653 |
| edge_delete | Step 14 enriched | 0.4527 | 0.1078 | 0.8922 | 0.9737 | 0.1408 | 0.0481 | 0.1542 |

Clean results are mixed: enriched local context improves context preservation but gives back some edit-sensitive performance relative to scores_only.

## Optional Noisy W012 Contrast

The optional noisy P2 + W012 run shows the same proposal-side ranking result because proposal admission is unchanged by rewrite checkpoint:

| Mode | Context-edge | Changed-edge | Add | Delete | Full-edge |
|---|---:|---:|---:|---:|---:|
| base proposal | 0.8850 | 0.2622 | 0.0894 | 0.4259 | 0.8755 |
| Step 9c completion_only | 0.8493 | 0.3547 | 0.2688 | 0.4361 | 0.8418 |
| Step 14 scores_only | 0.8504 | 0.3519 | 0.2590 | 0.4398 | 0.8428 |
| Step 14 enriched | 0.8500 | 0.3602 | 0.2722 | 0.4435 | 0.8426 |

This supports the same conclusion as RFT1: enriched features improve admission quality and edit-sensitive metrics, but context cost remains substantial.

## Decision Answers

### 1. Does `enriched_local_context` materially beat `scores_only` at fixed 10% budget?

Partially yes.

On the critical noisy P2 + RFT1 stack, enriched local context clearly improves proposal-side top-tail quality:

- event precision@B: `0.0505 -> 0.0625`
- event recall@B: `0.2432 -> 0.3010`
- rescued false-scope fraction: `0.9495 -> 0.9375`
- changed-edge recall: `0.5868 -> 0.6001`

But downstream context-edge does not materially improve over scores_only:

- scores_only context-edge: `0.8491`
- enriched context-edge: `0.8486`

So the answer is: enriched local context improves admission/ranking, but does not yet solve context stability.

### 2. Does `enriched_local_context` materially beat Step 9c and the current small-guard baseline?

On noisy P2 + RFT1, yes on the proposal-side and edit-sensitive metrics:

- event precision@B improves over Step 9c: `0.0507 -> 0.0625`
- changed-edge improves: `0.3650 -> 0.3709`
- add improves: `0.2815 -> 0.2859`
- delete improves: `0.4440 -> 0.4514`
- false-scope fraction drops: `0.9493 -> 0.9375`

However, the improvement is not large enough to challenge the stable noisy default because context-edge remains far below the base proposal path:

- base context-edge: `0.8836`
- enriched completion context-edge: `0.8486`

### 3. If yes, is the next training line best framed as a richer local-subgraph guard with a simple vanilla event-scope objective?

Yes, as an evaluation branch, not as a new stable default.

Step 13 already suggested that small loss tweaks were not enough. Step 14 shows that richer local structural representation gives a stronger signal than scores alone under noisy observation. That points to a richer local-subgraph guard as the clean next proposal-side branch.

The objective should remain event-scope membership for now. The issue is not that the target is wrong; it is that the representation is still too weak to approach the oracle top tail.

### 4. If no, does the evidence suggest that the guard line now requires a larger representation jump than justified?

The result is not a dead end. It does not justify abandoning the guard line, but it also does not justify a broad architecture jump.

The most faithful next move is a modest richer local-subgraph guard: more structured candidate context than Step 11 / Step 13, but still proposal-side, local, and evaluation-first. Stable defaults remain unchanged.

中文结论：Step 14 说明“更丰富的局部表示”确实有用，但还不是压倒性成功。下一步如果继续 proposal-side branch，应该做一个更强但仍局部的小型 subgraph guard，而不是继续调阈值、调 loss，或直接替换稳定默认模型。

## Current Recommendation

Do not replace the stable noisy default (`RFT1 + calibrated P2`).

Preserve Step 9c fixed-budget internal completion and Step 14 enriched local-context probing as an active proposal-side branch candidate.

The next mechanism, if pursued, should be:

- richer local-subgraph event-scope guard
- fixed 10% rescue budget
- no rewrite retraining initially
- no new Step 2-6 tuning

