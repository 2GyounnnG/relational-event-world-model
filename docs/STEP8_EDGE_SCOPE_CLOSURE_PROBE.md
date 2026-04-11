# Step 8: Edge Scope Closure Probe

## 1. Purpose

Step 7 showed that the dominant overall bottleneck is edge-side proposal miss / out-of-scope miss. Step 8 asks a narrower mechanism question:

> Among GT changed edges missed by proposal, are the endpoints already inside predicted node scope?

This determines whether the next mechanism should be:

- edge-completion / closure
- frontier / boundary expansion
- deeper node-anchor recall

This is evaluation-only. No training was added.

## 2. Evaluated runs

Core runs:

- clean structured:
  - `scope_proposal_node_edge_flipw2 + W012`
- noisy structured:
  - calibrated `P2 + W012`
  - calibrated `P2 + RFT1`

Important note:

- This pack is proposal-side anatomy and closure evaluation.
- Therefore `P2 + W012` and `P2 + RFT1` have identical anatomy/closure numbers because they use the same calibrated `P2` proposal front-end.
- No downstream rewrite-under-closure check was run; that would require a separate externally modified-mask rewrite pass and was intentionally skipped to keep this pack small.

Artifacts:

- `artifacts/edge_scope_closure_probe/clean_w012_edge_scope_closure_probe.json`
- `artifacts/edge_scope_closure_probe/noisy_w012_p2_edge_scope_closure_probe.json`
- `artifacts/edge_scope_closure_probe/noisy_rft1_p2_edge_scope_closure_probe.json`
- `artifacts/edge_scope_closure_probe/summary_miss_anatomy.csv`
- `artifacts/edge_scope_closure_probe/summary_closure_overall.csv`

## 3. Metrics

For every GT changed edge outside predicted edge scope, the evaluator classifies endpoint coverage into:

- `both_endpoints_in_predicted_node_scope`
- `one_endpoint_in_predicted_node_scope`
- `neither_endpoint_in_predicted_node_scope`

It also evaluates two proposal-side closure modes:

- `none`
  - original predicted edge scope
- `induced_edge_closure_over_predicted_nodes`
  - every valid edge slot whose two endpoints are both inside predicted node scope is included in proposal edge scope

中文说明：

- 这个 probe 不改变模型参数。
- closure 是一个诊断性上界：如果两个端点都已经被 node proposal 选中，那么把它们之间的 edge slot 加进 edge scope。
- 它回答“边漏检是不是因为 edge head 没补全 node scope 内的边”，而不是直接给出最终训练方案。

## 4. Missed-edge endpoint anatomy

### Overall

| run | missed changed edges | both endpoints share | one endpoint share | neither endpoint share |
|---|---:|---:|---:|---:|
| clean `W012` | 2240 | 0.779 | 0.213 | 0.009 |
| noisy `P2 + W012` | 4366 | 0.900 | 0.093 | 0.007 |
| noisy `P2 + RFT1` | 4366 | 0.900 | 0.093 | 0.007 |

### Event-type slices

| run | event type | missed changed edges | both endpoints share | one endpoint share | neither endpoint share |
|---|---|---:|---:|---:|---:|
| clean `W012` | `edge_add` | 1456 | 0.765 | 0.228 | 0.007 |
| clean `W012` | `edge_delete` | 1122 | 0.807 | 0.182 | 0.011 |
| noisy `P2 + W012` | `edge_add` | 3668 | 0.919 | 0.077 | 0.003 |
| noisy `P2 + W012` | `edge_delete` | 1330 | 0.848 | 0.138 | 0.014 |
| noisy `P2 + RFT1` | `edge_add` | 3668 | 0.919 | 0.077 | 0.003 |
| noisy `P2 + RFT1` | `edge_delete` | 1330 | 0.848 | 0.138 | 0.014 |

### Anatomy interpretation

The missed-edge subtype is overwhelmingly:

> both endpoints already inside predicted node scope

That is true in both clean and noisy settings:

- clean overall: `77.9%`
- noisy overall: `90.0%`
- noisy `edge_add`: `91.9%`
- noisy `edge_delete`: `84.8%`

The `neither endpoint` category is tiny:

- clean overall: `0.9%`
- noisy overall: `0.7%`

So this is not primarily deeper node-anchor recall. The proposal usually found the node anchors, but the edge proposal failed to include the changed edge slot between them.

中文结论：

**漏掉的 changed edges 大多数不是因为 node scope 没找到端点，而是因为 edge scope 没在已选 node scope 内补全相关边。**

## 5. Induced-edge closure probe

### Overall closure impact

| run | closure mode | edge recall | out-of-scope miss | edge scope excess ratio | edge pred slots |
|---|---|---:|---:|---:|---:|
| clean `W012` | `none` | 0.201 | 0.799 | 0.960 | 14150 |
| clean `W012` | `induced` | 0.823 | 0.177 | 0.980 | 116960 |
| noisy `P2 + W012` | `none` | 0.481 | 0.519 | 0.972 | 146566 |
| noisy `P2 + W012` | `induced` | 0.948 | 0.052 | 0.983 | 457934 |
| noisy `P2 + RFT1` | `none` | 0.481 | 0.519 | 0.972 | 146566 |
| noisy `P2 + RFT1` | `induced` | 0.948 | 0.052 | 0.983 | 457934 |

### Event-type closure impact

| run | event type | closure | edge recall | out-of-scope miss | edge scope excess ratio |
|---|---|---|---:|---:|---:|
| clean `W012` | `edge_add` | `none` | 0.074 | 0.926 | 0.974 |
| clean `W012` | `edge_add` | `induced` | 0.782 | 0.218 | 0.967 |
| clean `W012` | `edge_delete` | `none` | 0.319 | 0.681 | 0.892 |
| clean `W012` | `edge_delete` | `induced` | 0.869 | 0.131 | 0.964 |
| noisy `P2 + W012` | `edge_add` | `none` | 0.222 | 0.778 | 0.978 |
| noisy `P2 + W012` | `edge_add` | `induced` | 0.937 | 0.063 | 0.970 |
| noisy `P2 + W012` | `edge_delete` | `none` | 0.731 | 0.269 | 0.927 |
| noisy `P2 + W012` | `edge_delete` | `induced` | 0.959 | 0.041 | 0.970 |

### Closure interpretation

Induced closure materially reduces edge out-of-scope miss:

- clean overall:
  - miss `0.799 -> 0.177`
  - recall `0.201 -> 0.823`
- noisy overall:
  - miss `0.519 -> 0.052`
  - recall `0.481 -> 0.948`
- noisy `edge_add`:
  - miss `0.778 -> 0.063`
  - recall `0.222 -> 0.937`
- noisy `edge_delete`:
  - miss `0.269 -> 0.041`
  - recall `0.731 -> 0.959`

This is a large positive diagnostic result for edge-completion / closure.

However, naive full induced closure is not free:

- clean predicted edge slots: `14150 -> 116960`
- noisy predicted edge slots: `146566 -> 457934`

The scope excess ratio rises only modestly because it was already high, but the absolute number of candidate edge slots grows substantially. That means this should be treated as evidence for a closure mechanism, not necessarily as evidence that unconditional dense closure is the final deployment choice.

中文解读：

- closure 明显补上了大量漏掉的 changed edges。
- 但 naive closure 会显著放大 edge scope 的绝对大小。
- 因此下一步更像是“有控制的 edge-completion / closure”，而不是简单地永远把 node scope 内所有边都打开。

## 6. Decision questions

### 1. Among missed GT changed edges, which endpoint-coverage subtype dominates?

**Dominant subtype: both endpoints already inside predicted node scope.**

Evidence:

- clean overall: `0.779` of missed changed edges
- noisy overall: `0.900`
- noisy `edge_add`: `0.919`
- noisy `edge_delete`: `0.848`

The one-endpoint category is secondary, and the neither-endpoint category is tiny.

### 2. Does induced-edge closure materially reduce edge out-of-scope miss at acceptable excess cost?

**It materially reduces miss, but the naive version has a large absolute scope-size cost.**

Evidence:

- noisy overall out-of-scope miss drops from `0.519` to `0.052`
- noisy edge recall rises from `0.481` to `0.948`
- predicted edge slots rise from `146566` to `457934`

So the result is acceptable as a mechanism probe and upper bound, but not necessarily acceptable as the final raw inference rule without gating, scoring, or sparsification.

### 3. Based on the result, what should the next mechanism be?

**Next mechanism: edge-completion / closure.**

Not boundary expansion:

- one-endpoint misses are only `9.3%` of noisy missed changed edges.

Not deeper node-anchor recall:

- neither-endpoint misses are only `0.7%` of noisy missed changed edges.

The cleanest next mechanism question is therefore:

> Given a predicted node scope, how should the model propose or score candidate changed edges inside that node scope without exploding context-edge load?

## 7. Minimal next mechanism direction

The next mechanism should be an explicit edge-completion / closure variable at the proposal/rewrite interface.

Recommended shape for the next phase:

- keep proposal checkpointing and thresholds fixed initially
- add an evaluation/training path that can distinguish:
  - original edge-head scope
  - node-induced candidate edge closure
  - a learned or gated edge-completion subset inside predicted node scope

Do not reopen Step 2-6 micro-tuning for this. The Step 8 result points to a new edge-scope completion mechanism, not another keep/rescue/threshold/temperature sweep.
