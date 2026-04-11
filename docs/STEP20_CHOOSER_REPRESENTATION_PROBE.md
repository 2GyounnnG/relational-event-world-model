# Step 20: Chooser Representation Probe

## Scope

Step 20 tests whether richer rescued-edge local context improves the rescue-conditioned chooser problem.

Fixed:

- proposal operating mode: Step 9c fixed-budget internal completion
- rescue budget: `rescue_budget_fraction = 0.10`
- base proposal path unchanged
- Step9c proposal path unchanged
- proposal backbone frozen
- Step9 completion scorer frozen
- rewrite backbone frozen
- chooser target unchanged from Step 17 / Step 19
- objective fixed to plain BCE
- keep-fraction grid reused only for evaluation: `0.02`, `0.05`, `0.10`, `0.20`

New analysis variable:

- `feature_bundle = compact_interface_scores | enriched_rescue_local_context`

中文说明: 这一步不是再调目标函数，也不是调 keep fraction。它只问：加局部结构特征后，同一个浅层 probe 是否能更好地区分“该保留 Step9c 输出”的 rescued edges。

## Implementation

Added:

- `train/train_step20_chooser_representation_probe.py`
- `train/eval_step20_chooser_representation_probe.py`

Artifacts:

- `artifacts/step20_chooser_representation_probe/noisy_p2_rft1.json`
- `artifacts/step20_chooser_representation_probe/noisy_p2_rft1.csv`

Checkpoints:

- `checkpoints/step20_chooser_compact_noisy_p2_rft1/best.pt`
- `checkpoints/step20_chooser_enriched_noisy_p2_rft1/best.pt`

Feature bundles:

- `compact_interface_scores`: same compact interface-style signals used by Step17, including proposal scores, completion scores, endpoint node-scope scores, base/Step9c rewrite logits/probabilities, and disagreement.
- `enriched_rescue_local_context`: compact features plus endpoint node features, endpoint differences/products, scoped endpoint degrees, common-neighbor count in predicted node scope, predicted-scope density, local rewrite disagreement, and local rewrite probability-difference summaries.

Optional clean W012 was skipped because the noisy P2 + RFT1 run is the required decision run and the clean contrast would require two additional probe fits.

## Training Summary

| Feature bundle | Feature dim | Best epoch | Best val AP | Notes |
|---|---:|---:|---:|---|
| compact_interface_scores | 22 | 4 | 0.1112 | matches Step17-style compact signal |
| enriched_rescue_local_context | 54 | 5 | 0.1155 | modest validation-side representation gain |

Both probes used the same shallow MLP class and plain BCE objective.

## Test Ranking Diagnostics

| Model / objective | AP | AUROC |
|---|---:|---:|
| Step18 vanilla BCE gate | 0.1114 | 0.7094 |
| Step19 pairwise ranking gate | 0.1086 | 0.7128 |
| Step20 compact probe | 0.1084 | 0.7098 |
| Step20 enriched probe | 0.1159 | 0.7210 |

The enriched bundle is the best on both AP and AUROC, but the absolute gain is small.

中文解读: enriched features 确实有信息，但它们没有把 top-tail precision 拉到接近 oracle 的程度。

## Noisy P2 + RFT1 Frontier

### Keep Fraction 0.10

| Mode | Full edge | Context edge | Changed edge | Add | Delete | Target precision@k | Target recall@k | False-scope kept | Changed kept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step18 vanilla | 0.8704 | 0.8795 | 0.2829 | 0.1105 | 0.4463 | 0.1314 | 0.2437 | 0.9291 | 0.0362 |
| Step19 ranking | 0.8703 | 0.8794 | 0.2825 | 0.1095 | 0.4463 | 0.1313 | 0.2437 | 0.9312 | 0.0347 |
| Step20 compact | 0.8704 | 0.8795 | 0.2829 | 0.1105 | 0.4463 | 0.1321 | 0.2450 | 0.9298 | 0.0362 |
| Step20 enriched | 0.8704 | 0.8795 | 0.2841 | 0.1139 | 0.4454 | 0.1324 | 0.2456 | 0.9287 | 0.0393 |

At the practical 10% keep point, enriched local context is directionally better than compact, but not enough to change the system tradeoff.

### Keep Fraction 0.20

| Mode | Full edge | Context edge | Changed edge | Add | Delete | Target precision@k | Target recall@k | False-scope kept | Changed kept |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Step18 vanilla | 0.8659 | 0.8747 | 0.2958 | 0.1369 | 0.4463 | 0.1022 | 0.3791 | 0.9451 | 0.0369 |
| Step19 ranking | 0.8657 | 0.8745 | 0.2993 | 0.1442 | 0.4463 | 0.1054 | 0.3907 | 0.9407 | 0.0403 |
| Step20 compact | 0.8658 | 0.8747 | 0.2974 | 0.1403 | 0.4463 | 0.1005 | 0.3727 | 0.9430 | 0.0393 |
| Step20 enriched | 0.8660 | 0.8747 | 0.3027 | 0.1520 | 0.4454 | 0.1090 | 0.4043 | 0.9343 | 0.0471 |

At 20%, enriched local context gives the clearest edit-sensitive improvement, but context remains far below the base/oracle region.

## Reference Bounds

| Mode | Full edge | Context edge | Changed edge | Add | Delete |
|---|---:|---:|---:|---:|---:|
| base | 0.8742 | 0.8836 | 0.2710 | 0.0958 | 0.4370 |
| Step9c completion only | 0.8409 | 0.8482 | 0.3650 | 0.2815 | 0.4440 |
| oracle false-scope fallback | 0.8756 | 0.8835 | 0.3650 | 0.2815 | 0.4440 |
| oracle choose-better | 0.8773 | 0.8852 | 0.3664 | 0.2820 | 0.4463 |

Oracle still exposes a large ceiling. Step20 enriched does not approach it.

## Rescued-Edge Behavior

| Mode | False-scope preserve | True-changed correct-edit |
|---|---:|---:|
| Step9c completion only | 0.3103 | 0.7895 |
| Step18 vanilla 0.10 | 0.8617 | 0.1389 |
| Step19 ranking 0.10 | 0.8618 | 0.1346 |
| Step20 compact 0.10 | 0.8619 | 0.1389 |
| Step20 enriched 0.10 | 0.8613 | 0.1485 |
| Step18 vanilla 0.20 | 0.7676 | 0.2543 |
| Step19 ranking 0.20 | 0.7680 | 0.2756 |
| Step20 compact 0.20 | 0.7664 | 0.2692 |
| Step20 enriched 0.20 | 0.7682 | 0.3130 |
| oracle false-scope fallback | 0.9395 | 0.7895 |
| oracle choose-better | 0.9677 | 0.8024 |

The enriched probe improves true-changed correct-edit at fixed keep fractions, especially 20%, but does not solve false-scope preservation.

## Decision Answers

### 1. Does enriched_rescue_local_context materially beat compact_interface_scores on chooser top-tail quality?

It beats compact directionally, but only modestly.

Test ranking:

- compact AP / AUROC: `0.1084 / 0.7098`
- enriched AP / AUROC: `0.1159 / 0.7210`

At keep 0.10:

- target precision improves only from `0.1321` to `0.1324`
- target recall improves only from `0.2450` to `0.2456`
- false-scope kept improves from `0.9298` to `0.9287`

At keep 0.20:

- target precision improves from `0.1005` to `0.1090`
- target recall improves from `0.3727` to `0.4043`
- false-scope kept improves from `0.9430` to `0.9343`

So richer local context helps, but not enough to be called a material top-tail fix.

### 2. Does it materially improve the keep frontier over Step18 / Step19 at the same keep fractions?

No for the practical 10% point; modestly at 20%, but not enough.

At keep 0.10:

- enriched changed-edge: `0.2841` vs Step18 `0.2829`
- enriched add: `0.1139` vs Step18 `0.1105`
- enriched context-edge: `0.8795`, essentially unchanged

At keep 0.20:

- enriched changed-edge: `0.3027`, better than Step18 `0.2958` and Step19 `0.2993`
- enriched add: `0.1520`, better than Step18 `0.1369` and Step19 `0.1442`
- enriched context-edge: `0.8747`, still too far below base `0.8836`

中文结论: enriched probe 把曲线往正确方向推了一点，但没有把系统 tradeoff 推到可用分支。

### 3. If yes, is the next justified training line a stronger rescue-conditioned chooser representation?

The evidence supports representation as the right bottleneck direction, but this exact shallow enriched probe is not strong enough to become the next active branch.

A stronger chooser representation is justified only if we continue the interface line, because:

- oracle choose-better remains very strong
- enriched local context improves AP/AUROC and edit recovery
- compact/objective-only variants have plateaued

But the next representation would need to be a real jump beyond hand-built shallow local features.

### 4. If no, does this imply the current interface line requires a larger representation jump than justified for the current phase?

Yes, for the current minimal phase.

Step20 is the strongest minimal representation probe so far, and it still leaves a large gap:

- enriched keep 0.20 changed-edge `0.3027` vs oracle `0.3664`
- enriched keep 0.20 add `0.1520` vs oracle `0.2820`
- enriched keep 0.20 context-edge `0.8747` vs oracle `0.8852`

That suggests the current interface line would require a larger chooser representation jump to become system-useful. Stable defaults should remain unchanged.

## Bottom Line

Step20 answers the representation-sufficiency probe:

- richer local context helps ranking quality
- it does not materially fix the top-tail chooser problem
- no Step20 keep fraction becomes a plausible active branch
- the compact/minimal chooser line is likely exhausted for now
- further progress would require a larger rescue-conditioned chooser representation, not more tiny objective or operating-point tweaks

