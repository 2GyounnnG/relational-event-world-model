# Step 18: Fallback Gate Frontier

## Scope

Step 18 is an evaluation-only operating-point probe for the already trained Step 17 fallback gate.

Fixed:

- proposal operating mode: Step 9c fixed-budget internal completion
- rescue budget: `rescue_budget_fraction = 0.10`
- proposal / completion / rewrite checkpoints unchanged
- no new training
- no Step 9c rescue-budget sweep

New evaluation variable:

- `keep_fraction`: fraction of Step9c-rescued edges kept on the Step9c rewrite path.

All other rescued edges fall back to the base-proposal rewrite path. Non-rescued edges remain unchanged.

中文说明: 这里不是重新调 Step9c 的救援预算，而是在固定已经救出来的边集合上，问 Step17 gate 的分数排序有没有用。

## Implementation

Added:

- `train/eval_step18_fallback_gate_frontier.py`

Artifacts:

- `artifacts/step18_fallback_gate_frontier/noisy_p2_rft1.json`
- `artifacts/step18_fallback_gate_frontier/noisy_p2_rft1.csv`
- `artifacts/step18_fallback_gate_frontier/clean_w012.json`
- `artifacts/step18_fallback_gate_frontier/clean_w012.csv`

The evaluator first computes dataset-level score cutoffs over all rescued edges, then applies the fixed keep fractions:

- `0.02`
- `0.05`
- `0.10`
- `0.20`

This avoids per-sample integer rounding that would make 2% / 5% degenerate on samples with few rescued edges.

## Runs

| Run | Stack | Status |
|---|---|---|
| noisy P2 + RFT1 | calibrated P2 + Step9 noisy completion + RFT1 | completed |
| clean W012 | clean proposal + Step9 clean completion + W012 | completed |

The noisy run is the required decision run. Clean W012 was included because it was already wired and cheap.

## Noisy P2 + RFT1 Results

### System Metrics

| Mode | Keep fraction | Full edge | Context edge | Changed edge | Add | Delete |
|---|---:|---:|---:|---:|---:|---:|
| base | NA | 0.8742 | 0.8836 | 0.2710 | 0.0958 | 0.4370 |
| Step9c completion only | 1.0000 | 0.8409 | 0.8482 | 0.3650 | 0.2815 | 0.4440 |
| Step17 thresholded gate | 0.0000 | 0.8742 | 0.8836 | 0.2710 | 0.0958 | 0.4370 |
| Step18 top-k keep 0.02 | 0.0200 | 0.8735 | 0.8828 | 0.2732 | 0.0958 | 0.4412 |
| Step18 top-k keep 0.05 | 0.0500 | 0.8725 | 0.8817 | 0.2746 | 0.0953 | 0.4444 |
| Step18 top-k keep 0.10 | 0.1000 | 0.8704 | 0.8795 | 0.2829 | 0.1105 | 0.4463 |
| Step18 top-k keep 0.20 | 0.2000 | 0.8659 | 0.8747 | 0.2958 | 0.1369 | 0.4463 |
| oracle false-scope fallback | NA | 0.8756 | 0.8835 | 0.3650 | 0.2815 | 0.4440 |
| oracle choose-better | 0.0539 | 0.8773 | 0.8852 | 0.3664 | 0.2820 | 0.4463 |

### Chooser Quality

| Mode | Actual keep | Target precision among kept | Target recall | GT changed precision among kept | False-scope fraction among kept |
|---|---:|---:|---:|---:|---:|
| thresholded gate | 0.0000 | NA | 0.0000 | NA | NA |
| top-k 0.02 | 0.0200 | 0.1563 | 0.0580 | 0.0313 | 0.9271 |
| top-k 0.05 | 0.0500 | 0.1641 | 0.1522 | 0.0236 | 0.9277 |
| top-k 0.10 | 0.1000 | 0.1314 | 0.2437 | 0.0362 | 0.9291 |
| top-k 0.20 | 0.2000 | 0.1022 | 0.3791 | 0.0369 | 0.9451 |
| oracle choose-better | 0.0539 | 1.0000 | 1.0000 | 0.4649 | 0.4958 |

Ranking diagnostics over rescued edges:

- rescued candidates: `28,760`
- chooser target positives: `1,551`
- positive fraction: `0.0539`
- AUROC: `0.7094`
- AP: `0.1114`

中文解读: AUROC/AP 说明 gate 分数不是完全没信号；但 top tail 的纯度不够。真正应该保留 Step9c 的边只占 5.4%，而 top-k 0.05 的 precision 只有 16.4%，离 oracle 的 100% 很远。

### Rescued-Edge Behavior

| Mode | False-scope preserve | True-changed correct-edit |
|---|---:|---:|
| Step9c completion only | 0.3103 | 0.7895 |
| thresholded gate | 0.9395 | 0.0321 |
| top-k 0.02 | 0.9252 | 0.0513 |
| top-k 0.05 | 0.9056 | 0.0641 |
| top-k 0.10 | 0.8617 | 0.1389 |
| top-k 0.20 | 0.7676 | 0.2543 |
| oracle false-scope fallback | 0.9395 | 0.7895 |
| oracle choose-better | 0.9677 | 0.8024 |

Top-k keep recovers some true-changed edit behavior, but much too slowly relative to the context cost.

## Clean W012 Contrast

| Mode | Keep fraction | Full edge | Context edge | Changed edge | Add | Delete |
|---|---:|---:|---:|---:|---:|---:|
| base | NA | 0.9689 | 0.9828 | 0.0728 | 0.0220 | 0.1208 |
| Step9c completion only | 1.0000 | 0.9575 | 0.9707 | 0.1049 | 0.0455 | 0.1611 |
| Step17 thresholded gate | 0.0000 | 0.9689 | 0.9828 | 0.0728 | 0.0220 | 0.1208 |
| top-k 0.05 | 0.0499 | 0.9689 | 0.9828 | 0.0735 | 0.0235 | 0.1208 |
| top-k 0.10 | 0.1000 | 0.9670 | 0.9807 | 0.0792 | 0.0352 | 0.1208 |
| top-k 0.20 | 0.2000 | 0.9623 | 0.9758 | 0.0906 | 0.0455 | 0.1333 |
| oracle false-scope fallback | NA | 0.9688 | 0.9822 | 0.1049 | 0.0455 | 0.1611 |
| oracle choose-better | 0.0089 | 0.9694 | 0.9828 | 0.1049 | 0.0455 | 0.1611 |

Clean ranking diagnostics:

- rescued candidates: `9,477`
- chooser target positives: `84`
- positive fraction: `0.0089`
- AUROC: `0.8022`
- AP: `0.0206`

The clean gate has ranking signal by AUROC, but the useful-positive tail is extremely sparse. Top-k keeps still mostly admit false-scope edges.

## Event-Type Notes

Noisy top-k 0.10:

| Event type | Actual keep | Target precision among kept | Target recall | False-scope fraction among kept | Changed edge | Add | Delete |
|---|---:|---:|---:|---:|---:|---:|---:|
| edge_add | 0.1088 | 0.1485 | 0.1558 | 0.9168 | 0.1637 | 0.1105 | 0.5128 |
| edge_delete | 0.1019 | 0.1420 | 0.3655 | 0.9428 | 0.4066 | 0.1314 | 0.4463 |

Both slices show the same shape: there is some ranking signal, but the kept set is still dominated by false-scope rescued edges.

## Decision Answers

### 1. Does the Step 17 gate have useful ranking signal beyond its current thresholded all-fallback behavior?

Yes, but it is weak.

The noisy gate has AUROC `0.7094` and AP `0.1114` for the chooser target. Top-k 0.10 improves changed-edge from thresholded/base `0.2710` to `0.2829`, and add from `0.0958` to `0.1105`.

That is real signal, not ranking-dead behavior.

### 2. Is there a small keep_fraction that recovers a meaningful share of the oracle choose-better gain while preserving context?

No.

The best small operating point is arguably top-k 0.10, but it recovers only a small fraction of the oracle edit gain:

- changed-edge: `0.2829` vs oracle `0.3664`
- add: `0.1105` vs oracle `0.2820`
- context-edge: `0.8795`, already below base/oracle context around `0.8835-0.8852`

Top-k 0.20 recovers more changed/add signal, but context-edge drops further to `0.8747`, so the tradeoff is still poor.

中文结论: 有一点可用排序信号，但不是“换个 keep_fraction 就解决”的程度。

### 3. If yes, should the next branch be a fixed-keep / calibrated chooser operating mode with the current gate family?

No.

A fixed-keep operating mode with the current Step17 gate family should not become the next active branch. It is useful as a diagnostic, but the current gate score does not separate the high-value keep-Step9c edges sharply enough.

Stable defaults remain unchanged:

- noisy default: RFT1 + calibrated P2
- Step9c remains a proposal-side branch candidate, not a stable default

### 4. If no, does that make a stronger chooser representation the next justified step?

Yes.

The chooser interface itself is still justified because oracle choose-better is very strong:

- noisy oracle choose-better: full-edge `0.8773`, context-edge `0.8852`, changed-edge `0.3664`, add `0.2820`, delete `0.4463`

But the compact Step17 gate lacks enough top-tail precision. The next justified line should be a stronger rescue-conditioned chooser representation or a training objective explicitly aligned with the rare choose-Step9c target, not another threshold-only operating point.

## Bottom Line

Step18 answers the immediate uncertainty:

- The Step17 gate is not ranking-dead.
- The current thresholded all-fallback point wastes some signal.
- However, the signal is too weak for a fixed-keep frontier to produce a good system-level tradeoff.
- The current failure is best described as insufficient top-tail chooser precision from compact interface features.

下一步如果继续这个方向，应该增强 chooser 本身，而不是继续调 keep fraction。

