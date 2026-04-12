# Step 25: Noisy Multievent Oracle-Scope Headroom

## Scope

Step 25 introduces one evaluation variable:

`scope_source = learned | oracle`

No model was trained. The evaluator keeps the Step 22 noisy multievent substrate fixed:

- noisy structured observation input stays noisy
- latent clean target next-state stays clean
- proposal/rewrite checkpoints stay unchanged
- node threshold stays `0.15`
- edge threshold stays `0.10`

For oracle rows, GT event-scope masks are used as the rewrite scope. If the rewrite checkpoint uses proposal conditioning, the oracle masks are also passed as oracle proposal probabilities. This makes the oracle row a clean event-scope-interface headroom probe.

中文简述：这一步问的是“如果 proposal scope 完美，rewrite 能救回多少？”不是新训练。

## Runs

Required learned-scope references:

- learned scope: noisy P2 + RFT1
- learned scope: Step24 joint candidate

Oracle-scope rows:

- oracle scope + RFT1 rewrite
- oracle scope + Step24 rewrite

Optional reference rows included because they were already wired:

- oracle scope + W012
- noisy P2 + I1520

Artifacts:

- `artifacts/step25_noisy_multievent_oracle_headroom/oracle_scope_rft1.json`
- `artifacts/step25_noisy_multievent_oracle_headroom/oracle_scope_step24_rewrite.json`
- `artifacts/step25_noisy_multievent_oracle_headroom/oracle_scope_w012.json`
- `artifacts/step25_noisy_multievent_oracle_headroom/summary.json`
- `artifacts/step25_noisy_multievent_oracle_headroom/summary.csv`

## Overall Results

Final-step noisy Step 5 test metrics:

| Run | Scope | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Scope Recall | Out-of-Scope Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + RFT1 | learned | 0.8754 | 0.8821 | 0.2573 | 0.1111 | 0.4245 | 0.7095 | 0.5481 |
| Step24 joint | learned | 0.8762 | 0.8829 | 0.2472 | 0.1195 | 0.3933 | 0.7159 | 0.5347 |
| oracle + RFT1 | oracle | 0.9303 | 0.9370 | 0.3233 | 0.3690 | 0.2710 | 1.0000 | 0.0000 |
| oracle + Step24 rewrite | oracle | 0.9296 | 0.9368 | 0.2785 | 0.3124 | 0.2398 | 1.0000 | 0.0000 |
| oracle + W012 | oracle | 0.9308 | 0.9370 | 0.3669 | 0.4738 | 0.2446 | 1.0000 | 0.0000 |
| noisy P2 + I1520 | learned | 0.8361 | 0.8410 | 0.3714 | 0.1090 | 0.6715 | 0.7095 | 0.5481 |

Learned -> oracle gap for RFT1:

| Metric | Oracle RFT1 - Learned P2+RFT1 |
|---|---:|
| full-edge | +0.0549 |
| context-edge | +0.0549 |
| changed-edge | +0.0660 |
| add | +0.2579 |
| delete | -0.1535 |
| out-of-scope miss | -0.5481 |

The oracle scope strongly improves full/context-edge and add. It does not improve delete; overall delete gets worse.

## Strongly-Interacting Slice

Final-step strongly-interacting metrics:

| Run | Scope | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Scope Recall | Out-of-Scope Miss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + RFT1 | learned | 0.8771 | 0.8808 | 0.1008 | 0.0833 | 0.3333 | 0.7741 | 0.8527 |
| Step24 joint | learned | 0.8793 | 0.8831 | 0.0853 | 0.0833 | 0.1111 | 0.7701 | 0.8372 |
| oracle + RFT1 | oracle | 0.9324 | 0.9356 | 0.2868 | 0.2833 | 0.3333 | 1.0000 | 0.0000 |
| oracle + Step24 rewrite | oracle | 0.9316 | 0.9351 | 0.2326 | 0.2417 | 0.1111 | 1.0000 | 0.0000 |
| oracle + W012 | oracle | 0.9330 | 0.9354 | 0.4341 | 0.4417 | 0.3333 | 1.0000 | 0.0000 |
| noisy P2 + I1520 | learned | 0.8378 | 0.8414 | 0.0853 | 0.0750 | 0.2222 | 0.7741 | 0.8527 |

Strongly-interacting learned -> oracle gains:

| Comparison | Changed Edge Gain | Add Gain | Delete Gain | Context Edge Gain | Out-of-Scope Miss Reduction |
|---|---:|---:|---:|---:|---:|
| oracle RFT1 - learned P2+RFT1 | +0.1860 | +0.2000 | +0.0000 | +0.0548 | -0.8527 |
| oracle Step24 rewrite - learned Step24 | +0.1473 | +0.1583 | +0.0000 | +0.0520 | -0.8372 |

This is the main Step 25 result. On strongly-interacting examples, oracle scope recovers a lot of changed-edge/add and fully removes proposal out-of-scope miss, while keeping context-edge much higher. But delete does not improve under oracle scope.

中文解释：强交互问题有明显 proposal headroom，尤其是 edge_add / changed-edge；但 delete 不是单纯 scope 覆盖问题。

## Corruption Regime Summary

Oracle RFT1 final-step metrics:

| Corruption | Full Edge | Context Edge | Changed Edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| N1 | 0.9630 | 0.9700 | 0.3221 | 0.3522 | 0.2878 |
| N2 | 0.9303 | 0.9373 | 0.2919 | 0.3585 | 0.2158 |
| N3 | 0.8977 | 0.9037 | 0.3557 | 0.3962 | 0.3094 |

Oracle scope remains useful across N1/N2/N3. The noisy degradation is still visible, but it is not catastrophic.

## Event Family Summary

All-step event-family metrics:

| Run | Event Family | Changed Edge | Add | Delete | Context Edge |
|---|---|---:|---:|---:|---:|
| learned P2 + RFT1 | edge_add | 0.1120 | 0.1120 | NA | 0.8756 |
| oracle + RFT1 | edge_add | 0.3936 | 0.3936 | NA | 0.9372 |
| oracle + Step24 rewrite | edge_add | 0.3186 | 0.3186 | NA | 0.9372 |
| oracle + W012 | edge_add | 0.5042 | 0.5042 | NA | 0.9372 |
| learned P2 + RFT1 | edge_delete | 0.4266 | NA | 0.4266 | 0.8850 |
| oracle + RFT1 | edge_delete | 0.2665 | NA | 0.2665 | 0.9388 |
| oracle + Step24 rewrite | edge_delete | 0.2599 | NA | 0.2599 | 0.9388 |
| oracle + W012 | edge_delete | 0.2636 | NA | 0.2636 | 0.9388 |

Oracle scope provides very large edge_add headroom. Delete behaves differently: context gets much better, but delete accuracy drops relative to learned P2 + RFT1. This suggests delete performance is not simply proposal-coverage-limited.

Note: the compact Step 25 evaluator reports interaction bucket and event family separately, not a crossed bucket-by-event table. The strongly-interacting aggregate and event-family aggregate are enough to answer the main headroom question without adding another analysis dimension.

## Step24 Rewrite Under Oracle Scope

Oracle Step24 rewrite vs oracle RFT1:

| Metric | Overall Delta | Strongly-Interacting Delta |
|---|---:|---:|
| full-edge | -0.0007 | -0.0008 |
| context-edge | -0.0002 | -0.0005 |
| changed-edge | -0.0447 | -0.0543 |
| add | -0.0566 | -0.0417 |
| delete | -0.0312 | -0.2222 |

Step24 rewrite does not add value under oracle scope. It is worse than RFT1 on changed-edge/add/delete, including the strongly-interacting slice.

## Decision Answers

1. On the noisy strongly-interacting slice, how much headroom is proposal-limited versus rewrite-limited?

There is substantial proposal-coverage headroom for changed-edge/add. With RFT1, strong changed-edge rises from `0.1008` to `0.2868`, and add rises from `0.0833` to `0.2833` under oracle scope. Out-of-scope miss goes from `0.8527` to `0.0000`. However, strong delete stays at `0.3333`, so delete remains rewrite-side or interaction-handling limited even when scope is oracle.

2. Does oracle scope substantially recover changed-edge/delete while keeping context-edge stable?

Oracle scope substantially recovers changed-edge/add and improves context-edge, but it does not recover delete. For RFT1 overall, context-edge improves `0.8821 -> 0.9370` and changed-edge improves `0.2573 -> 0.3233`; add improves `0.1111 -> 0.3690`; delete drops `0.4245 -> 0.2710`.

3. Under oracle scope, is Step24 rewrite actually better than RFT1?

No. Step24 rewrite is worse than RFT1 under oracle scope: overall changed-edge is `0.2785` vs `0.3233`, add is `0.3124` vs `0.3690`, delete is `0.2398` vs `0.2710`, and strongly-interacting changed-edge is `0.2326` vs `0.2868`.

4. Based on the result, what is the next justified training line?

The next line should not be more proposal-only tweaking, and Step24-style light joint coupling is not enough. The evidence points to deeper joint noisy interaction-aware coupling with explicit proposal-coverage pressure for changed-edge/add, while also preserving or improving rewrite-side delete handling. If the project wants to isolate further before training, the next evaluation should specifically separate oracle-scope add vs delete behavior under interaction buckets.

## Outcome

Stable defaults remain unchanged:

- noisy broad default: `RFT1 + calibrated P2`
- clean broad default: `W012`
- interaction-aware alternative: `I1520`

Step25 does not promote Step24. It shows the system has real proposal-scope headroom on noisy strongly-interacting changed/add behavior, but also shows that delete is not solved by oracle scope and that the Step24 rewrite update did not improve oracle-scope rewrite quality.
