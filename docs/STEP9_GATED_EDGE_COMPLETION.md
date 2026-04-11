# Step 9: Gated Internal Edge Completion

## Scope

Step 9 tests exactly one new proposal-side mechanism variable:

`edge_completion_mode = off | naive_induced_closure | learned_gated_internal_completion`

The project remains in the structured synthetic graph-event setting. Node proposal is unchanged, rewrite is unchanged, and no Step 2-6 tuning loops are reopened.

## Motivation

Step 8 showed that most missed GT changed edges already have both endpoints inside predicted node scope:

- Clean overall: 77.9% of missed changed edges.
- Noisy overall: 90.0% of missed changed edges.
- Noisy `edge_add`: 91.9%.
- Noisy `edge_delete`: 84.8%.

Naive induced closure over predicted node scope nearly solves edge coverage, but makes the edge scope far too large. 中文总结：节点锚点已经大多找到了，问题主要是“节点 scope 内部的边没有闭合”，不是边界扩张或更深节点召回。

## Implemented Mechanism

The learned mode adds a small residual edge-completion head:

- It freezes the existing proposal checkpoint.
- It reuses frozen proposal node latents and base edge logits.
- It only considers candidate edge slots where both endpoints are already inside predicted node scope.
- It only rescues slots not already selected by the base edge proposal.
- The final edge scope is `base_edge_scope UNION learned_rescued_internal_edges`.
- The training target is still `event_scope_union_edges`, not only changed edges.
- Completion threshold is fixed at `0.5`; no threshold sweep was run.

New scripts:

- `train/train_step9_edge_completion.py`
- `train/eval_step9_gated_edge_completion.py`

Artifacts:

- `artifacts/step9_gated_edge_completion/clean_w012.json`
- `artifacts/step9_gated_edge_completion/noisy_p2_rft1.json`
- `artifacts/step9_gated_edge_completion/noisy_p2_w012.json`

## Training Runs

| Stack | Proposal checkpoint | Data | Best epoch | Best val score | Completion checkpoint |
|---|---:|---:|---:|---:|---|
| Clean | `scope_proposal_node_edge_flipw2` | `graph_event_train/val` | 2 | 0.1437 | `checkpoints/step9_edge_completion_clean/best.pt` |
| Noisy | calibrated P2 | `graph_event_step6a_train/val` | 2 | 0.0306 | `checkpoints/step9_edge_completion_noisy_p2/best.pt` |

Selection score is `recall_recovery_fraction_vs_naive - cost_fraction_vs_naive` on validation.

## Proposal-Side Results

### Overall

| Stack | Mode | Changed-edge recall | Out-of-scope miss | Excess ratio | Edge scope size | Recall recovery vs naive | Cost fraction vs naive |
|---|---|---:|---:|---:|---:|---:|---:|
| Clean W012 | off | 0.201 | 0.799 | 0.960 | 14,150 | NA | NA |
| Clean W012 | naive closure | 0.823 | 0.177 | 0.980 | 116,960 | NA | NA |
| Clean W012 | learned gated | 0.397 | 0.603 | 0.966 | 32,826 | 0.314 | 0.182 |
| Noisy P2 | off | 0.481 | 0.519 | 0.972 | 146,566 | NA | NA |
| Noisy P2 | naive closure | 0.948 | 0.052 | 0.983 | 457,934 | NA | NA |
| Noisy P2 | learned gated | 0.495 | 0.505 | 0.972 | 149,052 | 0.031 | 0.008 |

Clean learned completion recovers about 31% of the naive recall gain for about 18% of the naive scope-size cost. Noisy learned completion recovers only about 3% of the naive gain for less than 1% of the cost.

### Edge Add / Edge Delete

| Stack | Event type | Mode | Changed-edge recall | Out-of-scope miss | Edge scope size | Recovery | Cost |
|---|---|---|---:|---:|---:|---:|---:|
| Clean W012 | edge_add | off | 0.074 | 0.926 | 4,414 | NA | NA |
| Clean W012 | edge_add | naive | 0.782 | 0.218 | 37,542 | NA | NA |
| Clean W012 | edge_add | learned | 0.202 | 0.798 | 10,462 | 0.181 | 0.183 |
| Clean W012 | edge_delete | off | 0.319 | 0.681 | 4,888 | NA | NA |
| Clean W012 | edge_delete | naive | 0.869 | 0.131 | 39,710 | NA | NA |
| Clean W012 | edge_delete | learned | 0.593 | 0.407 | 11,086 | 0.499 | 0.178 |
| Noisy P2 | edge_add | off | 0.222 | 0.778 | 47,228 | NA | NA |
| Noisy P2 | edge_add | naive | 0.937 | 0.063 | 147,576 | NA | NA |
| Noisy P2 | edge_add | learned | 0.246 | 0.754 | 47,952 | 0.033 | 0.007 |
| Noisy P2 | edge_delete | off | 0.731 | 0.269 | 49,792 | NA | NA |
| Noisy P2 | edge_delete | naive | 0.959 | 0.041 | 157,398 | NA | NA |
| Noisy P2 | edge_delete | learned | 0.737 | 0.263 | 50,672 | 0.025 | 0.008 |

The learned head is strongest on clean `edge_delete`, where it recovers roughly half of the naive recall gain at under one fifth of the naive cost. Under noisy observations it is too conservative for both `edge_add` and `edge_delete`.

## End-to-End Results

### Clean W012

| Mode | Full edge | Changed edge | Context edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| off | 0.969 | 0.073 | 0.983 | 0.022 | 0.121 |
| naive closure | 0.837 | 0.285 | 0.846 | 0.386 | 0.189 |
| learned gated | 0.945 | 0.146 | 0.958 | 0.079 | 0.208 |

Clean learned completion improves changed-edge, add, and delete behavior while preserving much more full/context stability than naive closure.

### Noisy P2 + RFT1

| Mode | Full edge | Changed edge | Context edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| off | 0.874 | 0.271 | 0.884 | 0.096 | 0.437 |
| naive closure | 0.792 | 0.308 | 0.799 | 0.341 | 0.277 |
| learned gated | 0.872 | 0.283 | 0.881 | 0.120 | 0.438 |

Noisy learned completion gives small downstream gains on changed edge and add, with little effect on delete and a small context/full-edge cost.

### Noisy P2 + W012

| Mode | Full edge | Changed edge | Context edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| off | 0.875 | 0.262 | 0.885 | 0.089 | 0.426 |
| naive closure | 0.741 | 0.364 | 0.747 | 0.458 | 0.275 |
| learned gated | 0.873 | 0.274 | 0.882 | 0.112 | 0.427 |

The optional W012 noisy contrast matches the RFT1 pattern: learned completion is safe but weak; naive closure improves edit-sensitive metrics while damaging context stability.

## Decision Questions

### 1. Does learned gated internal completion recover a meaningful share of the naive-closure gain?

Partially. Clean learned completion recovers a meaningful share: 31.4% overall, 49.9% on clean `edge_delete`, and 18.1% on clean `edge_add`.

Noisy learned completion does not recover enough: only 3.1% overall, 3.3% on noisy `edge_add`, and 2.5% on noisy `edge_delete`.

### 2. Does it do so at much lower scope inflation cost?

Yes. The learned head is much cheaper than naive closure:

- Clean cost fraction: 18.2% of naive scope inflation.
- Noisy cost fraction: 0.8% of naive scope inflation.

However, in the noisy setting this low cost mostly reflects insufficient rescue power rather than a fully successful efficient frontier.

### 3. Is this strong enough to become the next main proposal-side branch?

Not yet.

The mechanism direction is validated by Step 8 and by the clean Step 9 result, but the current learned implementation is too conservative under the noisy default stack. The current best noisy system should remain `RFT1 + calibrated P2` for now.

### 4. If not, what failed?

The failure is primarily insufficient rescue power under noisy observation, not too much excess.

Evidence:

- Noisy learned completion rescues only 2,486 edge slots versus 311,368 slots for naive closure.
- It rescues only 120 changed edges overall.
- Noisy recall improves from 0.481 to only 0.495, while naive closure reaches 0.948.
- Downstream changed-edge improves modestly, but the proposal-side coverage gain is too small to change the system-level bottleneck.

中文判断：不是“补太多边导致污染”，而是“门控太保守，真正该补的内部边没有被救出来”。clean 上说明机制有信号；noisy 上说明当前 head/训练目标/固定操作点还不够有力。

## Current Step 9 Conclusion

Step 9 supports edge-completion / closure as the right mechanism family, but the first learned gated implementation is not yet strong enough to become the main proposal-side branch.

Recommended next move, if continuing this line, is still within the same mechanism variable: improve learned gated internal completion so it rescues more true internal event-scope edges without approaching naive closure cost. Do not switch to boundary expansion or deeper node-anchor recall based on these results.

