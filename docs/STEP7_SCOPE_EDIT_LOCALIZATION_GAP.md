# Step 7: Scope/Edit Localization Gap

## 1. Purpose

This Step 7 entry pack asks one focused mechanism question:

> When proposal already gives a plausible local scope, is the dominant failure mode  
> (A) proposal miss / insufficient changed-region coverage,  
> or (B) rewrite-side edit mislocalization inside scope?

This was evaluated without new training. The goal here is diagnosis, not another tuning loop.

## 2. Evaluated runs

Core runs:

- clean structured default:
  - `scope_proposal_node_edge_flipw2 + W012`
- noisy structured:
  - calibrated `P2 + W012`
  - calibrated `P2 + RFT1`

Optional clean contrast that was cheap to run:

- `scope_proposal_node_edge_flipw2 + WG025`

Artifacts:

- `artifacts/scope_edit_localization_gap/clean_w012_scope_edit_gap.json`
- `artifacts/scope_edit_localization_gap/clean_wg025_scope_edit_gap.json`
- `artifacts/scope_edit_localization_gap/noisy_w012_p2_scope_edit_gap.json`
- `artifacts/scope_edit_localization_gap/noisy_rft1_p2_scope_edit_gap.json`
- `artifacts/scope_edit_localization_gap/summary_overall.csv`
- `artifacts/scope_edit_localization_gap/summary_event_type.csv`

## 3. Metric definitions

The evaluator decomposes error into proposal-side and rewrite-side buckets.

### Proposal-side

- `proposal_changed_region_recall_node / edge`
  - GT changed items covered by predicted proposal scope
- `proposal_scope_excess_ratio_node / edge`
  - predicted scope items that are actually GT context
- `out_of_scope_miss_node / edge`
  - GT changed items that fall outside predicted scope

### Rewrite-side

- `in_scope_under_edit_edge`
  - GT changed edges inside scope that the full rewrite still predicts incorrectly
- `in_scope_over_edit_edge`
  - GT context edges inside scope that the full rewrite corrupts

Node-side rewrite buckets are reported separately for type and state:

- `in_scope_under_edit_node_type`
- `in_scope_over_edit_node_type`
- `in_scope_under_edit_node_state_not_improved`
  - changed nodes inside scope whose state prediction is not even closer to target than the copy/current baseline
- `in_scope_over_edit_node_state_mae`
  - mean state MAE on unchanged nodes inside scope

中文说明：

- 边侧可以直接做清晰的二值“改对 / 改错”统计。
- 节点状态是连续值，所以这里不用伪精确的“完全正确”硬二值，而是看是否至少优于 copy/current，以及对应的 MAE。

For direct bottleneck comparison, the most useful quantities are:

- `out_of_scope_miss_edge`
- `in_scope_under_edit_edge_share_of_gt_changed`
- `in_scope_over_edit_edge`

The first measures proposal miss on GT changed edges. The second measures rewrite under-edit on GT changed edges that proposal did cover. The third measures rewrite corruption on extra context inside scope.

## 4. Overall summary

### Overall table

| run | edge recall | edge miss | edge in-scope under-edit share | edge in-scope over-edit | node recall | node miss | node state not-improved share |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean `W012` | 0.201 | 0.799 | 0.128 | 0.229 | 0.909 | 0.091 | 0.721 |
| clean `WG025` | 0.201 | 0.799 | 0.111 | 0.280 | 0.909 | 0.091 | 0.624 |
| noisy `P2 + W012` | 0.481 | 0.519 | 0.276 | 0.332 | 0.969 | 0.031 | 0.779 |
| noisy `P2 + RFT1` | 0.481 | 0.519 | 0.267 | 0.338 | 0.969 | 0.031 | 0.649 |

### Immediate reading

1. Edge-side proposal coverage is still the dominant limiter.
   - clean `W012`: edge miss `0.799` vs edge in-scope under-edit share `0.128`
   - noisy `P2 + W012`: edge miss `0.519` vs edge in-scope under-edit share `0.276`
   - noisy `P2 + RFT1`: edge miss `0.519` vs edge in-scope under-edit share `0.267`

2. Proposal scopes are plausible but still extremely context-heavy on edges.
   - clean edge scope excess ratio: `0.960`
   - noisy edge scope excess ratio: `0.972`

3. Node-side behavior looks different from edge-side behavior.
   - node proposal recall is already high: `0.909` clean, `0.969` noisy
   - but in-scope node-state under-edit remains large:
     - noisy `P2 + W012`: `0.779`
     - noisy `P2 + RFT1`: `0.649`

So the split diagnosis is:

- edge side: proposal miss dominates
- node state side: rewrite localization inside scope is still a real issue

## 5. Event-type breakdown

Important note: event-type groups use **contains-event-type** membership. Two-event samples can contribute to multiple event-type slices.

### Clean `W012`

| event type | edge miss | edge in-scope under-edit share | edge in-scope over-edit | node state not-improved share |
|---|---:|---:|---:|---:|
| `node_state_update` | 0.801 | 0.105 | 0.250 | 0.525 |
| `edge_add` | 0.926 | 0.037 | 0.210 | 0.746 |
| `edge_delete` | 0.681 | 0.210 | 0.236 | 0.665 |
| `motif_type_flip` | 0.784 | 0.144 | 0.212 | 0.949 |

### Noisy `P2 + W012`

| event type | edge miss | edge in-scope under-edit share | edge in-scope over-edit | node state not-improved share |
|---|---:|---:|---:|---:|
| `node_state_update` | 0.485 | 0.304 | 0.345 | 0.627 |
| `edge_add` | 0.778 | 0.110 | 0.326 | 0.808 |
| `edge_delete` | 0.269 | 0.439 | 0.332 | 0.743 |
| `motif_type_flip` | 0.517 | 0.296 | 0.323 | 0.950 |

### Noisy `P2 + RFT1`

| event type | edge miss | edge in-scope under-edit share | edge in-scope over-edit | node state not-improved share |
|---|---:|---:|---:|---:|
| `node_state_update` | 0.485 | 0.314 | 0.350 | 0.405 |
| `edge_add` | 0.778 | 0.090 | 0.332 | 0.679 |
| `edge_delete` | 0.269 | 0.427 | 0.338 | 0.596 |
| `motif_type_flip` | 0.517 | 0.256 | 0.331 | 0.930 |

### Event-type interpretation

- `edge_add` is still strongly proposal-limited.
  - Even under noisy `P2`, edge miss stays at `0.778`, while in-scope under-edit stays much smaller.
- `edge_delete` is the clearest rewrite-sensitive slice.
  - noisy `P2 + W012`: edge miss `0.269`, in-scope under-edit share `0.439`
  - noisy `P2 + RFT1`: edge miss `0.269`, in-scope under-edit share `0.427`
- `node_state_update` and `motif_type_flip` show that node-state editing inside scope remains difficult even when node proposal recall is already high.

So the event-type view adds an important nuance:

- overall edge bottleneck: proposal miss
- delete-heavy and node-state-heavy slices: rewrite localization still matters a lot

## 6. Clean vs noisy comparison

Moving from clean `W012` to noisy calibrated `P2 + W012` improves proposal edge coverage a lot:

- edge recall: `0.201 -> 0.481`
- edge miss: `0.799 -> 0.519`

But noisy operation also exposes stronger in-scope rewrite-side stress:

- edge in-scope over-edit: `0.229 -> 0.332`
- edge in-scope under-edit share: `0.128 -> 0.276`

`RFT1` improves the node-side in-scope state problem:

- node state not-improved share: `0.779 -> 0.649`

and slightly improves edge in-scope under-edit:

- `0.276 -> 0.267`

But `RFT1` does not change the proposal miss term, because the front-end is the same calibrated `P2`.

## 7. Decision question

### Is the dominant bottleneck proposal miss, or in-scope edit mislocalization?

**Answer: the dominant system-level bottleneck is still proposal miss on the edge side.**

Why:

1. The largest single failure term on the main changed-edge path is still `out_of_scope_miss_edge`.
   - clean `W012`: `0.799`
   - noisy `P2 + W012`: `0.519`
   - noisy `P2 + RFT1`: `0.519`

2. The corresponding rewrite-side changed-edge failure inside scope is smaller.
   - clean `W012` in-scope under-edit share: `0.128`
   - noisy `P2 + W012`: `0.276`
   - noisy `P2 + RFT1`: `0.267`

3. Rewrite adaptation changes the in-scope term but not the proposal miss term.
   - `RFT1` improves node-state and some in-scope edit behavior
   - but the dominant edge miss floor remains unchanged because it is upstream

中文结论：

**如果只允许给出一个主导瓶颈判断，当前答案应是 proposal miss，而不是 scope 内 rewrite edit mislocalization。**

More precise version:

- edge-level overall bottleneck: proposal miss
- node-state and delete-heavy secondary bottleneck: rewrite mislocalization inside scope

## 8. Why this points to a genuinely new mechanism issue

This does **not** look like another threshold-only problem:

- Step 6 threshold calibration already helped
- regime-aware thresholding did not help further
- temperature scaling did not help further

So the remaining issue is not just operating point selection. The edge-side proposal still has a structural coverage/discrimination problem:

- recall on GT changed edges is still limited
- predicted edge scope remains heavily contaminated by context

That means the next mechanism question should be about how proposal represents and isolates the true changed edge subset, not about reopening old Step 6 calibration loops.
