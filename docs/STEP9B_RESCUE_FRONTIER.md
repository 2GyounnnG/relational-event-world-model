# Step 9b: Rescue Frontier Diagnostic

## Question

Under noisy observation, is the current learned edge-completion head:

- **A. score-separable but overly conservative at its current operating point**, or
- **B. fundamentally weak in ranking / separability?**

This pack does not train a new model. It reuses the Step 9 completion checkpoints and evaluates ranked internal candidate rescue.

## Candidate Definition

Internal rescue candidates are edge slots where:

- both endpoints are already inside predicted node scope,
- the base proposal edge scope did not already include the edge.

These are exactly the edge slots that naive induced closure would add.

中文说明：这里只看“节点 scope 内部但 base edge proposal 漏掉的边”。如果这个集合里分数排序有用，就说明问题更像 operating point；如果排序也没信号，就需要更强 pair scorer。

## New Evaluation Variable

`rescue_budget_fraction`

Budget is measured relative to the naive closure cost gap:

`added_edges_budget = rescue_budget_fraction * (naive_scope_size - base_scope_size)`

Implementation detail: budget is applied per sample as top-k over internal candidates, with:

`k = floor(rescue_budget_fraction * candidate_count)`

Fixed budget grid:

- `0.02`
- `0.05`
- `0.10`
- `0.20`

No threshold sweep, no retraining, no temperature scaling.

## Runs

| Run | Proposal | Rewrite | Completion checkpoint | Dataset |
|---|---|---|---|---|
| Clean W012 | clean proposal `scope_proposal_node_edge_flipw2` | W012 | `step9_edge_completion_clean/best.pt` | `graph_event_test.pkl` |
| Noisy P2 + RFT1 | calibrated P2 | RFT1 | `step9_edge_completion_noisy_p2/best.pt` | `graph_event_step6a_test.pkl` |
| Noisy P2 + W012 | calibrated P2 | W012 | `step9_edge_completion_noisy_p2/best.pt` | `graph_event_step6a_test.pkl` |

Artifacts:

- `artifacts/step9b_rescue_frontier/clean_w012.json`
- `artifacts/step9b_rescue_frontier/noisy_p2_rft1.json`
- `artifacts/step9b_rescue_frontier/noisy_p2_w012.json`
- matching `.csv` summaries in the same directory

## Score Separability

### Overall

| Stack | Oracle AP | Oracle AUROC | Changed-edge AP | Changed-edge AUROC | Pos median score | Neg median score |
|---|---:|---:|---:|---:|---:|---:|
| Clean W012 | 0.1117 | 0.7777 | 0.0274 | 0.6690 | 0.498 | 0.172 |
| Noisy P2 | 0.0424 | 0.7024 | 0.0287 | 0.7141 | 0.214 | 0.121 |

Noisy AP is low because positives are very sparse, but AUROC is clearly above random and positive scores are shifted upward. This is weak-but-real ranking signal, not total separability failure.

### Edge Add / Edge Delete

| Stack | Event type | Oracle AP | Oracle AUROC | Changed AP | Changed AUROC |
|---|---|---:|---:|---:|---:|
| Clean | edge_add | 0.0649 | 0.6465 | 0.0370 | 0.5800 |
| Clean | edge_delete | 0.1041 | 0.8023 | 0.0717 | 0.7860 |
| Noisy | edge_add | 0.0878 | 0.7313 | 0.0817 | 0.7312 |
| Noisy | edge_delete | 0.0263 | 0.6799 | 0.0214 | 0.6753 |

The noisy head ranks `edge_add` candidates better than the default 0.5 threshold suggested. `edge_delete` ranking is weaker, but still above random.

## Proposal Frontier

### Clean W012

| Mode | Edge recall | Out-of-scope miss | Edge scope size | Recovery vs naive | Cost vs naive |
|---|---:|---:|---:|---:|---:|
| base | 0.201 | 0.799 | 14,150 | NA | NA |
| default threshold 0.5 | 0.397 | 0.603 | 32,826 | 0.314 | 0.182 |
| budget 0.02 | 0.215 | 0.785 | 15,261 | 0.023 | 0.011 |
| budget 0.05 | 0.252 | 0.748 | 18,432 | 0.081 | 0.042 |
| budget 0.10 | 0.295 | 0.705 | 23,627 | 0.151 | 0.092 |
| budget 0.20 | 0.392 | 0.608 | 33,919 | 0.307 | 0.192 |
| naive closure | 0.823 | 0.177 | 116,960 | NA | NA |

Clean behavior is smooth and coherent: more budget yields more recall, and the 20% budget nearly matches the default-threshold completion operating point.

### Noisy P2

| Mode | Edge recall | Out-of-scope miss | Edge scope size | Recovery vs naive | Cost vs naive |
|---|---:|---:|---:|---:|---:|
| base | 0.481 | 0.519 | 146,566 | NA | NA |
| default threshold 0.5 | 0.495 | 0.505 | 149,052 | 0.031 | 0.008 |
| budget 0.02 | 0.493 | 0.507 | 149,836 | 0.025 | 0.011 |
| budget 0.05 | 0.533 | 0.467 | 159,458 | 0.111 | 0.041 |
| budget 0.10 | 0.592 | 0.408 | 175,326 | 0.238 | 0.092 |
| budget 0.20 | 0.683 | 0.317 | 206,482 | 0.433 | 0.192 |
| naive closure | 0.948 | 0.052 | 457,934 | NA | NA |

Noisy budgeted rescue reveals much stronger utility than the default 0.5 operating point. At 10% budget, it recovers 23.8% of the naive recall gain for 9.2% of the naive cost. At 20% budget, it recovers 43.3% of the gain for 19.2% of the cost.

### Noisy Edge Add / Edge Delete

| Event type | Mode | Edge recall | Recovery vs naive | Cost vs naive |
|---|---|---:|---:|---:|
| edge_add | base | 0.222 | NA | NA |
| edge_add | budget 0.05 | 0.302 | 0.112 | 0.041 |
| edge_add | budget 0.10 | 0.393 | 0.239 | 0.092 |
| edge_add | budget 0.20 | 0.536 | 0.440 | 0.193 |
| edge_add | naive | 0.937 | NA | NA |
| edge_delete | base | 0.731 | NA | NA |
| edge_delete | budget 0.05 | 0.756 | 0.110 | 0.042 |
| edge_delete | budget 0.10 | 0.781 | 0.217 | 0.092 |
| edge_delete | budget 0.20 | 0.822 | 0.401 | 0.192 |
| edge_delete | naive | 0.959 | NA | NA |

The noisy frontier is especially important for `edge_add`: default threshold barely moved recall, but ranked budgeted rescue moves from 0.222 to 0.393 at 10% budget and 0.536 at 20% budget.

## Downstream Rewrite Frontier

### Noisy P2 + RFT1

| Mode | Changed edge | Context edge | Add | Delete |
|---|---:|---:|---:|---:|
| base | 0.271 | 0.884 | 0.096 | 0.437 |
| default threshold 0.5 | 0.283 | 0.881 | 0.120 | 0.438 |
| budget 0.02 | 0.286 | 0.878 | 0.128 | 0.436 |
| budget 0.05 | 0.322 | 0.866 | 0.198 | 0.439 |
| budget 0.10 | 0.365 | 0.848 | 0.282 | 0.444 |
| budget 0.20 | 0.431 | 0.810 | 0.413 | 0.448 |
| naive closure | 0.308 | 0.799 | 0.341 | 0.277 |

Budgeted rescue beats naive closure on changed-edge at 10% and 20% budget while keeping substantially better context-edge accuracy. This is a strong sign that ranking is useful even though the default 0.5 operating point was too conservative.

### Noisy P2 + W012

| Mode | Changed edge | Context edge | Add | Delete |
|---|---:|---:|---:|---:|
| base | 0.262 | 0.885 | 0.089 | 0.426 |
| default threshold 0.5 | 0.274 | 0.882 | 0.112 | 0.427 |
| budget 0.05 | 0.312 | 0.868 | 0.189 | 0.429 |
| budget 0.10 | 0.355 | 0.849 | 0.269 | 0.436 |
| budget 0.20 | 0.421 | 0.812 | 0.400 | 0.440 |
| naive closure | 0.364 | 0.747 | 0.458 | 0.275 |

The optional W012 contrast tells the same story: budgeted top-k rescue exposes useful score ordering and avoids the severe context collapse caused by naive closure.

## Decision Answers

### Does the noisy learned completion head have useful ranking signal beyond its default operating point?

Yes. The noisy head is score-separable but overly conservative at the default threshold.

Evidence:

- Overall noisy changed-edge AUROC is 0.714.
- Noisy `edge_add` changed-edge AUROC is 0.731.
- Budgeted top-k rescue at 10% improves proposal recall from 0.481 to 0.592.
- The same 10% budget improves downstream changed-edge from 0.271 to 0.365 and add from 0.096 to 0.282 for RFT1.

AP is low, so the scorer is not strong in an absolute sparse-ranking sense. But it is not fundamentally useless.

### At what small rescue budget does it start to recover a meaningful share of the naive-closure gain?

For noisy P2:

- 2% budget is too small.
- 5% budget starts to matter: recall recovery 11.1%, changed-edge downstream 0.322.
- 10% budget is the first clearly useful point: recall recovery 23.8%, cost 9.2%, downstream changed-edge 0.365.
- 20% budget is stronger but more tradeoff-heavy: recall recovery 43.3%, cost 19.2%, context-edge drops to 0.810.

The best diagnostic budget is 10%. The best aggressive edit-sensitive budget is 20%.

### Is Step 9 weakness mainly operating-point conservatism, or true score-separability weakness?

Mostly operating-point conservatism, with moderate score weakness.

中文判断：不是“分数完全排不出好边”，而是默认 0.5 阈值太保守；但 AP 很低，说明 scorer 本身还不够锋利，不能只靠一个固定阈值解决。

The current default operating point rescued only 2,486 edges and recovered 3.1% of naive recall gain. Budgeted ranking at 10% rescued more useful candidates and recovered 23.8% of the gain.

### Next implementation choice

Choose **(i) budgeted/top-k learned internal completion** as the next implementation.

Reason:

- It directly uses the existing signal without retraining.
- It materially improves noisy changed-edge/add behavior.
- It preserves far more context stability than naive closure.
- It answers the Step 9 failure mode: the learned head has signal, but fixed-threshold operation is too conservative.

Do not switch yet to a stronger internal pair-scoring head as the immediate next move. A stronger scorer may be needed later, but the next minimal mechanism should first formalize budgeted/top-k internal completion and validate it as a stable proposal-side operating mode.

