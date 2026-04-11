# Step 10: Rescued Scope Rewrite Decomposition

## Question

Step 9c showed that fixed-budget 10% internal completion improves edit-sensitive metrics, especially `changed_edge` and `add`, but hurts `context_edge`.

Step 10 asks:

Where exactly does that context loss come from?

Candidate explanations:

1. false rescue: the proposal adds edge slots that are not in GT event scope,
2. over-edit on correctly rescued true-scope context,
3. spillover onto context edges that were already inside the original base proposal scope.

中文说明：这一步不是训练，不改 scorer；只把 10% budget 打开的边拆开看，看 context 损失到底是“救错边”还是 rewrite 在原 scope 里被扰乱。

## Evaluator

New script:

- `train/eval_step10_rescued_scope_rewrite_decomp.py`

Artifacts:

- `artifacts/step10_rescued_scope_rewrite_decomp/noisy_p2_rft1.json`
- `artifacts/step10_rescued_scope_rewrite_decomp/clean_w012.json`
- `artifacts/step10_rescued_scope_rewrite_decomp/noisy_p2_w012.json`
- matching `.csv` summaries in the same directory

Comparison:

- base proposal
- fixed-budget internal completion at `rescue_budget_fraction = 0.10`

No retraining, no budget sweep, no threshold sweep.

## Definitions

Base scope:

- original proposal edge mask before budgeted rescue

Rescued edges:

- edges added by fixed-budget internal completion relative to base

Rescued edge classes:

- `rescued_true_changed`: edge is in GT changed set
- `rescued_true_scope_context`: edge is in GT event scope but not changed
- `rescued_false_scope`: edge is not in GT event scope

Spillover set:

- original base-scope context edges
- measured under both base proposal and budgeted proposal

## Noisy P2 + RFT1 Main Result

### Rescued Edge Composition

| Class | Count | Fraction |
|---|---:|---:|
| rescued true changed | 936 | 0.033 |
| rescued true-scope context | 521 | 0.018 |
| rescued false scope | 27,303 | 0.949 |
| total rescued | 28,760 | 1.000 |

The 10% budgeted mode opens many internal edges, but about 95% of them are not in GT event scope.

### Rewrite Behavior on Rescued Edges

| Group | Base correctness/preserve | Budget correctness/preserve | Delta / drop |
|---|---:|---:|---:|
| rescued true changed correct edit | 0.032 | 0.790 | +0.757 |
| rescued true-scope context preserve | 0.841 | 0.770 | -0.071 |
| rescued false-scope preserve | 0.939 | 0.310 | -0.629 |

The budgeted proposal helps true changed rescued edges substantially. But the false-scope rescued edges are often corrupted once exposed to rewrite.

### Spillover

| Context set | Preserve under base | Preserve under budget | Drop |
|---|---:|---:|---:|
| original base-scope context | 0.662 | 0.662 | 0.000 |
| all context | 0.884 | 0.848 | 0.035 |
| rescued context only | 0.938 | 0.319 | 0.619 |

There is no measurable spillover onto original base-scope context edges. The context drop is local to the newly rescued context edges.

### Extra Context Error Attribution

| Source | Extra errors | Share of total extra context errors |
|---|---:|---:|
| rescued true-scope context | 37 | 0.002 |
| rescued false scope | 17,178 | 0.896 |
| original base-scope spillover | 0 | 0.000 |

The remaining unassigned share comes from context edges outside these named groups / accounting boundaries, but the dominant named source is false rescue.

## Clean W012 Result

| Metric | Value |
|---|---:|
| rescued total | 9,477 |
| rescued true changed fraction | 0.028 |
| rescued true-scope context fraction | 0.092 |
| rescued false-scope fraction | 0.880 |
| correct edit rate on rescued true changed | 0.319 |
| preserve rate on rescued true-scope context | 0.879 |
| preserve rate on rescued false-scope | 0.769 |
| spillover context drop | 0.000 |
| all context drop | 0.012 |
| extra context error share from false scope | 0.881 |

Clean behaves similarly but less severely: false rescue is still the dominant source of context loss, but W012 preserves false-scope rescued edges better in the clean setting than RFT1 does under noisy input.

## Optional Noisy P2 + W012 Contrast

| Metric | Value |
|---|---:|
| rescued false-scope fraction | 0.949 |
| correct edit rate on rescued true changed | 0.784 |
| preserve rate on rescued true-scope context | 0.756 |
| preserve rate on rescued false-scope | 0.303 |
| spillover context drop | 0.000 |
| all context drop | 0.036 |
| extra context error share from false scope | 0.899 |

The W012 noisy contrast matches RFT1: the problem is not specific to RFT1. False-scope rescued edges dominate the context loss under noisy observation.

## Event-Type Breakdown

### Noisy P2 + RFT1

| Event type | Rescued total | True changed frac | True-scope context frac | False-scope frac | True changed correct | False-scope preserve | Spillover drop |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge_add | 9,282 | 0.087 | 0.008 | 0.905 | 0.869 | 0.345 | 0.000 |
| edge_delete | 9,950 | 0.025 | 0.006 | 0.970 | 0.527 | 0.300 | 0.000 |

`edge_add` gets more true changed rescues than `edge_delete`, and the rewrite uses them well. But both slices are dominated by false-scope rescued edges.

## Decision Answers

### 1. Is the budgeted-10% context loss mainly from false rescue, over-edit on true-scope context, or spillover?

It is mainly from false rescue.

For noisy P2 + RFT1:

- 94.9% of rescued edges are outside GT event scope.
- False-scope rescued edges preserve at only 0.310 under budgeted rewrite.
- False-scope rescued edges account for about 89.6% of extra context errors.
- True-scope context over-edit accounts for only about 0.2% of extra context errors.
- Spillover onto original base-scope context is exactly 0.0 in this evaluation.

中文判断：不是原来的 base scope 被污染，也不是 true-scope context 被大量误改；主要是 10% budget 里“救错”的 false-scope 边太多，一旦交给 rewrite，就容易被改坏。

### 2. Does the evidence point to the next mechanism being proposal-side or rewrite-side?

The primary evidence points back to proposal-side precision / guarding.

Step 9c showed budgeted completion has useful ranking and improves edits. Step 10 shows the context cost comes mostly from false-scope rescues, not from spillover or broad rewrite instability.

That means proposal-side work is not done: the budgeted completion branch needs a better way to avoid or flag false-scope internal edges.

### 3. If rewrite-side dominates, is the minimal next mechanism best framed as rescue-conditioned change-mask / don't-touch gating?

Rewrite-side does not dominate here.

A rescue-conditioned don't-touch gate remains conceptually relevant because false-scope rescued edges are corrupted after being exposed. But Step 10 does not support making it the immediate main mechanism by itself. If most rescued edges are false-scope, a rewrite gate would be compensating for proposal false positives rather than solving the root selection problem.

### 4. If proposal false rescue still dominates materially, why is proposal-side work not done?

Proposal-side work is not done because the 10% budget mode opens a set where only 3.3% of rescued edges are truly changed and only 1.8% are true-scope context under noisy P2 + RFT1.

The budgeted ranking is useful, but not precise enough. It improves recall and downstream edits, yet the false-scope fraction is so high that context loss becomes the limiting factor.

## Step 10 Conclusion

Budgeted internal completion remains a promising proposal-side branch candidate, but its next bottleneck is false rescue precision.

Stable defaults remain unchanged:

- noisy broad default remains RFT1 + calibrated P2.
- budgeted 10% internal completion remains a branch candidate, not a default.

The next minimal work should stay proposal-side: add a precision guard / false-rescue filter for budgeted internal completion, or evaluate a rescue confidence criterion that reduces false-scope edges before involving rewrite-side training.

