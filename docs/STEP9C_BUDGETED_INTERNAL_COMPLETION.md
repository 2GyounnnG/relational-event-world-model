# Step 9c: Budgeted Internal Completion Operating Mode

## Goal

Step 9c turns the Step 9b rescue-frontier diagnostic into one reusable proposal-side operating mode:

`fixed-budget top-k learned internal completion`

This is not a training pack. It reuses the existing Step 9 completion heads:

- `checkpoints/step9_edge_completion_clean/best.pt`
- `checkpoints/step9_edge_completion_noisy_p2/best.pt`

Stable defaults remain unchanged:

- broad clean default: W012
- noisy broad default: RFT1 + calibrated P2

## Operating Mode

Budgeted internal completion uses:

`rescue_budget_fraction = 0.10`

Candidate edge slots must satisfy:

- both endpoints are already inside predicted node scope,
- base proposal edge scope did not already include the edge.

Candidates are ranked by the learned completion score, and the top candidates are added until the budget is exhausted.

Budget definition:

`added_edges_budget = 0.10 * (naive_induced_closure_scope_size - base_scope_size)`

Implementation detail: because naive closure adds exactly all internal candidates not already in base edge scope, this is implemented per sample as:

`k = floor(0.10 * internal_candidate_count)`

中文说明：Step 9c 不再扫 budget，也不调 threshold；它把 10% top-k rescue 固定成一个可复用 proposal-side operating mode。

## Implemented Script

New evaluator:

- `train/eval_step9c_budgeted_internal_completion.py`

It reports exactly these modes:

1. `off`
2. `learned_default_threshold_0.5`
3. `learned_budget_0.10`
4. `naive_induced_closure`

Artifacts:

- `artifacts/step9c_budgeted_internal_completion/clean_w012.json`
- `artifacts/step9c_budgeted_internal_completion/noisy_p2_rft1.json`
- `artifacts/step9c_budgeted_internal_completion/noisy_p2_w012.json`
- matching `.csv` summaries in the same directory

## Clean W012 Results

### Proposal Side

| Mode | Edge recall | Out-of-scope miss | Edge scope size | Recovery vs naive | Cost vs naive |
|---|---:|---:|---:|---:|---:|
| base | 0.201 | 0.799 | 14,150 | NA | NA |
| threshold 0.5 | 0.397 | 0.603 | 32,826 | 0.314 | 0.182 |
| budget 0.10 | 0.295 | 0.705 | 23,627 | 0.151 | 0.092 |
| naive closure | 0.823 | 0.177 | 116,960 | NA | NA |

### Downstream

| Mode | Full edge | Changed edge | Context edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base | 0.969 | 0.073 | 0.983 | 0.022 | 0.121 |
| threshold 0.5 | 0.945 | 0.146 | 0.958 | 0.079 | 0.208 |
| budget 0.10 | 0.957 | 0.105 | 0.971 | 0.045 | 0.161 |
| naive closure | 0.837 | 0.285 | 0.846 | 0.386 | 0.189 |

Clean 10% budget is more conservative than the thresholded Step 9 mode. It preserves context better, but gives up some edit recovery.

## Noisy P2 + RFT1 Results

This is the main Step 9c target stack.

### Proposal Side

| Mode | Edge recall | Out-of-scope miss | Edge scope size | Recovery vs naive | Cost vs naive |
|---|---:|---:|---:|---:|---:|
| base | 0.481 | 0.519 | 146,566 | NA | NA |
| threshold 0.5 | 0.495 | 0.505 | 149,052 | 0.031 | 0.008 |
| budget 0.10 | 0.592 | 0.408 | 175,326 | 0.238 | 0.092 |
| naive closure | 0.948 | 0.052 | 457,934 | NA | NA |

### Proposal Side by Event Type

| Event type | Mode | Edge recall | Recovery vs naive | Cost vs naive |
|---|---|---:|---:|---:|
| edge_add | base | 0.222 | NA | NA |
| edge_add | threshold 0.5 | 0.246 | 0.033 | 0.007 |
| edge_add | budget 0.10 | 0.393 | 0.239 | 0.092 |
| edge_add | naive closure | 0.937 | NA | NA |
| edge_delete | base | 0.731 | NA | NA |
| edge_delete | threshold 0.5 | 0.737 | 0.025 | 0.008 |
| edge_delete | budget 0.10 | 0.781 | 0.217 | 0.092 |
| edge_delete | naive closure | 0.959 | NA | NA |

### Downstream

| Mode | Full edge | Changed edge | Context edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base | 0.874 | 0.271 | 0.884 | 0.096 | 0.437 |
| threshold 0.5 | 0.872 | 0.283 | 0.881 | 0.120 | 0.438 |
| budget 0.10 | 0.841 | 0.365 | 0.848 | 0.282 | 0.444 |
| naive closure | 0.792 | 0.308 | 0.799 | 0.341 | 0.277 |

Budgeted 10% completion clearly outperforms the thresholded Step 9 operating point on noisy changed-edge and add behavior. It also beats naive closure on changed-edge while preserving substantially better context-edge accuracy.

## Optional Noisy P2 + W012 Contrast

| Mode | Full edge | Changed edge | Context edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base | 0.875 | 0.262 | 0.885 | 0.089 | 0.426 |
| threshold 0.5 | 0.873 | 0.274 | 0.882 | 0.112 | 0.427 |
| budget 0.10 | 0.842 | 0.355 | 0.849 | 0.269 | 0.436 |
| naive closure | 0.741 | 0.364 | 0.747 | 0.458 | 0.275 |

The same pattern holds with W012: budgeted completion strongly improves edit-sensitive metrics relative to thresholded completion, but costs context/full-edge stability.

## Decision Questions

### 1. Does fixed-budget 10% internal completion outperform the current thresholded Step 9 operating point in the noisy default stack?

Yes, on the intended edit-sensitive axes.

For noisy P2 + RFT1:

- Proposal edge recall improves from 0.495 to 0.592.
- Out-of-scope miss improves from 0.505 to 0.408.
- Downstream changed-edge improves from 0.283 to 0.365.
- Add improves from 0.120 to 0.282.
- Delete is essentially preserved, 0.438 to 0.444.

The tradeoff is context/full-edge stability:

- Context edge drops from 0.881 to 0.848.
- Full edge drops from 0.872 to 0.841.

### 2. Does it provide a good gain/cost tradeoff relative to naive closure?

Yes, but only as an edit-sensitive proposal branch, not as a universal default.

At 10% budget:

- It recovers 23.8% of the naive recall gain for 9.2% of the naive scope-size cost.
- It keeps context-edge at 0.848 versus 0.799 for naive closure.
- It beats naive closure on downstream changed-edge, 0.365 versus 0.308, under RFT1.

中文判断：10% budget 的性价比明显好于 naive closure。它不是“把所有内部边都打开”，而是利用 learned score 做较小范围的有序补边。

### 3. Is it strong enough to preserve as the next proposal-side branch candidate, while keeping the stable noisy default unchanged?

Yes.

`learned_budget_0.10` should be preserved as the next proposal-side branch candidate because it converts the Step 9b ranking signal into a reusable operating mode and gives a meaningful noisy edit-sensitive gain.

But it should not replace the stable noisy default yet. RFT1 + calibrated P2 remains safer because the 10% budget mode gives back about 3.5 points of context-edge accuracy and 3.3 points of full-edge accuracy relative to base.

### 4. If not fully promoted, what is still limiting it?

The remaining limiter is downstream context cost, not insufficient ranking quality.

Evidence:

- Ranking is useful enough to recover meaningful recall at 10% budget.
- Proposal-side recall improves substantially.
- Downstream changed-edge and add improve substantially.
- The cost is that extra internal edges create enough context exposure for rewrite to degrade full/context edge accuracy.

So the next branch should not be a stronger scorer first. The next minimal practical step should evaluate whether the rewrite can tolerate this budgeted proposal mode, or whether the budgeted mode needs a stability guard before promotion.

## Step 9c Conclusion

The 10% fixed-budget internal completion mode is a real and reusable proposal-side operating mode.

It should be kept as the next proposal-side branch candidate:

- better than thresholded Step 9 under noisy observation,
- much cheaper and less context-damaging than naive closure,
- especially useful for `edge_add`,
- still not safe enough to replace the stable noisy default.

Stable defaults remain unchanged.

