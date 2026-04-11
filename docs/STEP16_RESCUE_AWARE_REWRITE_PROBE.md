# Step 16: Rescue-Aware Rewrite Probe

## Purpose

Step 16 asks a proposal/rewrite interface question:

> If rescued false-scope edges were prevented from being actively rewritten, how much of Step 9c's context cost could be recovered without giving back most of its edit-sensitive gains?

This is an oracle evaluation probe, not a training run.

Fixed setup:

- proposal operating mode: Step 9c fixed-budget internal completion
- `rescue_budget_fraction = 0.10`
- completion scorer unchanged
- rewrite checkpoints unchanged
- thresholds unchanged
- no budget sweep
- no rewrite retraining

中文说明：这一步不是改 proposal，也不是训练 rewrite；它只问“如果 rewrite 对救错的 false-scope 边别动，会发生什么？”

## Mechanism Variable

| Variable | Values |
|---|---|
| `protection_mode` | `off`, `oracle_false_scope_fallback` |

`oracle_false_scope_fallback` is applied after unchanged Step 9c rewrite:

1. Run base proposal + rewrite.
2. Run Step 9c budgeted internal completion + rewrite.
3. Identify Step 9c rescued edges that are not in GT event scope.
4. For those rescued false-scope edges only, replace Step 9c edge logits with the base-mode edge logits for the same sample and edge.

This uses base-mode output fallback, not raw noisy-observation copying.

## Files And Artifacts

Script:

- `train/eval_step16_rescue_aware_rewrite_probe.py`

Artifacts:

- `artifacts/step16_rescue_aware_rewrite_probe/clean_w012.json`
- `artifacts/step16_rescue_aware_rewrite_probe/clean_w012.csv`
- `artifacts/step16_rescue_aware_rewrite_probe/noisy_p2_rft1.json`
- `artifacts/step16_rescue_aware_rewrite_probe/noisy_p2_rft1.csv`
- `artifacts/step16_rescue_aware_rewrite_probe/noisy_p2_w012.json`
- `artifacts/step16_rescue_aware_rewrite_probe/noisy_p2_w012.csv`

Compile passed for the new evaluator and reused proposal/rewrite modules.

## Main Noisy Result: Calibrated P2 + RFT1

| Mode | Full-edge | Context-edge | Changed-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base proposal | 0.8742 | 0.8836 | 0.2710 | 0.0958 | 0.4370 |
| Step 9c completion_only | 0.8409 | 0.8482 | 0.3650 | 0.2815 | 0.4440 |
| Step 9c + oracle false-scope fallback | 0.8756 | 0.8835 | 0.3650 | 0.2815 | 0.4440 |
| oracle event-scope guard @ 10% | 0.8505 | 0.8543 | 0.6051 | 0.8069 | 0.4139 |

Step 9c improves edit-sensitive metrics but damages context:

- changed-edge: `0.2710 -> 0.3650`
- add: `0.0958 -> 0.2815`
- delete: `0.4370 -> 0.4440`
- context-edge: `0.8836 -> 0.8482`

Oracle false-scope fallback keeps the Step 9c edit gains exactly while restoring context-edge almost to base:

- changed-edge stays `0.3650`
- add stays `0.2815`
- delete stays `0.4440`
- context-edge recovers `0.8482 -> 0.8835`
- full-edge recovers `0.8409 -> 0.8756`

This is the strongest evidence so far that the Step 9c branch has a valuable proposal/rewrite interface ceiling if false-scope rescues can be protected.

## Rescued-Edge Decomposition

### Noisy P2 + RFT1

| Metric | Step 9c | Step 9c + false-scope fallback |
|---|---:|---:|
| rescued total | 28,760 | 28,760 |
| rescued true changed fraction | 0.0325 | 0.0325 |
| rescued true-scope context fraction | 0.0181 | 0.0181 |
| rescued false-scope fraction | 0.9493 | 0.9493 |
| rescued true-changed correct-edit rate | 0.7895 | 0.7895 |
| rescued false-scope preserve rate | 0.3103 | 0.9395 |
| false-scope extra context errors | 17,178 | 0 |
| spillover context drop | 0.0000 | 0.0000 |

The fallback does exactly what the interface probe intended:

- It does not change the rescued set.
- It does not harm rescued true-changed edits.
- It removes the false-scope rescued-edge context-error source.
- It does not create spillover on the original base scope.

中文结论：Step 9c 的 edit gain 还在；context loss 基本可以通过“false-scope rescue 不要让 rewrite 主动改”这个接口消掉。

## Event-Type Breakdown

### Noisy P2 + RFT1

| Event type | Mode | Context-edge | Changed-edge | Add | Delete | False-scope preserve | True-changed correct |
|---|---|---:|---:|---:|---:|---:|---:|
| edge_add | Step 9c | 0.8498 | 0.3121 | 0.2815 | 0.5128 | 0.3448 | 0.8686 |
| edge_add | Step 9c + fallback | 0.8819 | 0.3121 | 0.2815 | 0.5128 | 0.9372 | 0.8686 |
| edge_delete | Step 9c | 0.8509 | 0.4219 | 0.2692 | 0.4440 | 0.3004 | 0.5265 |
| edge_delete | Step 9c + fallback | 0.8883 | 0.4219 | 0.2692 | 0.4440 | 0.9410 | 0.5265 |

Both `edge_add` and `edge_delete` show the same structure: fallback recovers context while preserving Step 9c's changed/add/delete behavior.

## Clean W012 Result

| Mode | Full-edge | Context-edge | Changed-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base proposal | 0.9689 | 0.9828 | 0.0728 | 0.0220 | 0.1208 |
| Step 9c completion_only | 0.9575 | 0.9707 | 0.1049 | 0.0455 | 0.1611 |
| Step 9c + oracle false-scope fallback | 0.9688 | 0.9822 | 0.1049 | 0.0455 | 0.1611 |
| oracle event-scope guard @ 10% | 0.9647 | 0.9729 | 0.4379 | 0.6012 | 0.2833 |

Clean behavior matches noisy behavior: context is recovered almost to base, while Step 9c edit gains are preserved.

## Optional Noisy W012 Contrast

| Mode | Full-edge | Context-edge | Changed-edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base proposal | 0.8755 | 0.8850 | 0.2622 | 0.0894 | 0.4259 |
| Step 9c completion_only | 0.8418 | 0.8493 | 0.3547 | 0.2688 | 0.4361 |
| Step 9c + oracle false-scope fallback | 0.8768 | 0.8849 | 0.3547 | 0.2688 | 0.4361 |
| oracle event-scope guard @ 10% | 0.8513 | 0.8554 | 0.5849 | 0.7708 | 0.4088 |

The optional W012 contrast confirms the result is not specific to RFT1.

## Deltas Vs Step 9c

### Noisy P2 + RFT1

| Mode | Δ full-edge | Δ context-edge | Δ changed-edge | Δ add | Δ delete |
|---|---:|---:|---:|---:|---:|
| base proposal | +0.0334 | +0.0353 | -0.0939 | -0.1857 | -0.0069 |
| Step 9c + false-scope fallback | +0.0347 | +0.0353 | +0.0000 | +0.0000 | +0.0000 |
| oracle event-scope guard @ 10% | +0.0097 | +0.0061 | +0.2401 | +0.5254 | -0.0301 |

The fallback result is unusually clean: it recovers the full context loss of Step 9c without losing edit-sensitive gains.

## Decision Answers

### 1. Does oracle false-scope fallback materially recover context-edge relative to Step 9c?

Yes, decisively.

On noisy P2 + RFT1:

- context-edge improves from `0.8482` to `0.8835`
- base context-edge is `0.8836`
- false-scope extra context errors go from `17,178` to `0`

This essentially recovers the entire Step 9c context penalty.

### 2. Does it preserve most of Step 9c's changed-edge / add / delete gains?

Yes. In this oracle probe, it preserves them exactly:

- changed-edge remains `0.3650`
- add remains `0.2815`
- delete remains `0.4440`
- rescued true-changed correct-edit rate remains `0.7895`

The fallback only changes rescued false-scope edges, so the edit-sensitive value of true rescues is not harmed.

### 3. If yes, does this justify the next training line as rescue-conditioned rewrite-side don't-touch / fallback gating?

Yes, as the next evaluation/training branch to test.

Step 10 showed false-scope rescued edges cause the context loss. Step 16 shows that if those edges can be routed to a base-output fallback / don't-touch behavior, Step 9c's system-level tradeoff becomes much stronger.

The next minimal mechanism should therefore be framed as:

- rescue-conditioned rewrite-side fallback / don't-touch gating
- trained or supervised to preserve likely false-scope rescued edges
- without changing the Step 9c budget, proposal thresholds, or completion scorer

中文建议：下一步不应该再继续小 guard 微调，而应该测试 rewrite 侧是否能学会“被 rescue 进来的边，除非真的需要改，否则走 fallback / don’t-touch”。

### 4. If not, does Step 9c have limited system-level ceiling?

This probe says the opposite: Step 9c has a higher system-level ceiling than the raw Step 9c result suggested.

The current limitation is not that Step 9c rescues are useless. True-changed rescued edges are used well. The limitation is the proposal/rewrite interface: false-scope rescued edges are admitted and then actively rewritten. An oracle fallback removes that cost.

## Recommendation

Stable defaults remain unchanged:

- noisy broad default: RFT1 + calibrated P2
- clean broad default: W012

But Step 16 creates a strong next-branch justification:

- keep Step 9c fixed-budget internal completion as the proposal-side branch
- add a rescue-conditioned rewrite-side fallback / don't-touch gate as the next mechanism to test
- do not reopen Step 2-6 tuning, Step 9c budget sweeps, or small guard tweaks

