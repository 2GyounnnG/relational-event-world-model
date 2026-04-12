# Step 27: Step26 Proposal/Rewrite Factorization

## Scope

Step 27 introduces one evaluation variable:

`component_source = baseline | step26`

The variable is applied independently to the proposal and rewrite components. No model was trained.

The comparison matrix is:

1. baseline proposal + baseline rewrite
2. Step26 proposal + baseline rewrite
3. baseline proposal + Step26 rewrite
4. Step26 proposal + Step26 rewrite
5. oracle scope + baseline rewrite
6. oracle scope + Step26 rewrite

Fixed constraints:

- noisy Step22 multievent interaction substrate
- node threshold: `0.15`
- edge threshold: `0.10`
- no threshold sweep
- no Step26 recipe tweak
- no proposal or rewrite retraining

中文简述：这一步只做 attribution。把 Step26 拆成 proposal 组件和 rewrite 组件，看 gain 和 harm 分别来自哪里。

## Checkpoints

Baseline components:

- baseline proposal: `checkpoints/proposal_noisy_obs_p2/best.pt`
- baseline rewrite: `checkpoints/step6_noisy_rewrite_rft1/best.pt`

Step26 components:

- Step26 proposal: `checkpoints/step26_noisy_interaction_joint_deeper/proposal_best.pt`
- Step26 rewrite: `checkpoints/step26_noisy_interaction_joint_deeper/rewrite_best.pt`

Evaluator:

- `train/eval_step27_step26_factorization.py`

Artifacts:

- `artifacts/step27_step26_factorization/summary.json`
- `artifacts/step27_step26_factorization/summary.csv`

## Overall Matrix

Final-step noisy Step5 test metrics:

| Proposal | Rewrite | Scope | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Recall | Out-of-Scope Miss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline P2 | baseline RFT1 | learned | 0.8754 | 0.8821 | 0.2573 | 0.1111 | 0.4245 | 0.7095 | 0.5481 |
| Step26 | baseline RFT1 | learned | 0.8264 | 0.8312 | 0.3758 | 0.3040 | 0.4580 | 0.8100 | 0.3926 |
| baseline P2 | Step26 | learned | 0.8782 | 0.8851 | 0.2371 | 0.1111 | 0.3813 | 0.7095 | 0.5481 |
| Step26 | Step26 | learned | 0.8264 | 0.8313 | 0.3557 | 0.3061 | 0.4125 | 0.8100 | 0.3926 |
| oracle | baseline RFT1 | oracle | 0.9303 | 0.9370 | 0.3233 | 0.3690 | 0.2710 | 1.0000 | 0.0000 |
| oracle | Step26 | oracle | 0.9290 | 0.9359 | 0.2964 | 0.2683 | 0.3285 | 1.0000 | 0.0000 |

The strongest overall edit-sensitive learned-scope row is `Step26 proposal + baseline RFT1`: changed-edge `0.3758`, add `0.3040`, delete `0.4580`. This is better than the full Step26 row on changed-edge and delete, with nearly identical broad full/context cost.

## Strongly-Interacting Slice

Final-step strongly-interacting metrics:

| Proposal | Rewrite | Scope | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Recall | Out-of-Scope Miss |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline P2 | baseline RFT1 | learned | 0.8771 | 0.8808 | 0.1008 | 0.0833 | 0.3333 | 0.7741 | 0.8527 |
| Step26 | baseline RFT1 | learned | 0.8254 | 0.8281 | 0.2558 | 0.2500 | 0.3333 | 0.8523 | 0.6744 |
| baseline P2 | Step26 | learned | 0.8819 | 0.8857 | 0.0853 | 0.0833 | 0.1111 | 0.7741 | 0.8527 |
| Step26 | Step26 | learned | 0.8287 | 0.8314 | 0.2403 | 0.2583 | 0.0000 | 0.8523 | 0.6744 |
| oracle | baseline RFT1 | oracle | 0.9324 | 0.9356 | 0.2868 | 0.2833 | 0.3333 | 1.0000 | 0.0000 |
| oracle | Step26 | oracle | 0.9302 | 0.9338 | 0.2093 | 0.2000 | 0.3333 | 1.0000 | 0.0000 |

This is the core attribution result. Step26 proposal with baseline RFT1 preserves most of the Step26 changed/add gain while avoiding the Step26 delete collapse.

Strongly-interacting deltas vs baseline P2 + RFT1:

| Row | Changed Edge Delta | Add Delta | Delete Delta | Context Delta | Edge Recall Delta | Out-of-Scope Delta |
|---|---:|---:|---:|---:|---:|---:|
| Step26 proposal + RFT1 | +0.1550 | +0.1667 | +0.0000 | -0.0527 | +0.0782 | -0.1783 |
| P2 proposal + Step26 rewrite | -0.0155 | +0.0000 | -0.2222 | +0.0049 | +0.0000 | +0.0000 |
| Step26 proposal + Step26 rewrite | +0.1395 | +0.1750 | -0.3333 | -0.0494 | +0.0782 | -0.1783 |

中文解释：proposal 组件带来 coverage 和 changed/add；rewrite 组件没有带来强交互 changed/add，反而把 delete 打掉了。Step26 的正贡献主要是 proposal-side，负贡献主要是 rewrite-side。

## Corruption Breakdown

Final-step Step26 proposal + RFT1:

| Corruption | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| N1 | 0.8492 | 0.8541 | 0.3960 | 0.3270 | 0.4748 | 0.8493 | 0.3658 |
| N2 | 0.8242 | 0.8291 | 0.3557 | 0.3082 | 0.4101 | 0.8142 | 0.3658 |
| N3 | 0.8059 | 0.8104 | 0.3758 | 0.2767 | 0.4892 | 0.7665 | 0.4463 |

Final-step Step26 proposal + Step26 rewrite:

| Corruption | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| N1 | 0.8493 | 0.8544 | 0.3792 | 0.3333 | 0.4317 | 0.8493 | 0.3658 |
| N2 | 0.8276 | 0.8326 | 0.3423 | 0.3082 | 0.3813 | 0.8142 | 0.3658 |
| N3 | 0.8023 | 0.8070 | 0.3456 | 0.2767 | 0.4245 | 0.7665 | 0.4463 |

Across N1/N2/N3, reverting the rewrite to RFT1 generally improves changed/delete while keeping the same proposal coverage profile.

## Event Family Notes

All-step event-family metrics:

| Row | Event Family | Changed Edge | Add | Delete | Context Edge | Edge Recall | Out-of-Scope Miss |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline P2 + RFT1 | edge_add | 0.1120 | 0.1120 | NA | 0.8756 | 0.1457 | 0.8543 |
| Step26 proposal + RFT1 | edge_add | 0.3445 | 0.3445 | NA | 0.8249 | 0.3985 | 0.6015 |
| Step26 proposal + Step26 rewrite | edge_add | 0.3445 | 0.3445 | NA | 0.8240 | 0.3985 | 0.6015 |
| baseline P2 + RFT1 | edge_delete | 0.4266 | NA | 0.4266 | 0.8850 | 0.8076 | 0.1924 |
| Step26 proposal + RFT1 | edge_delete | 0.4391 | NA | 0.4391 | 0.8288 | 0.8656 | 0.1344 |
| Step26 proposal + Step26 rewrite | edge_delete | 0.4288 | NA | 0.4288 | 0.8270 | 0.8656 | 0.1344 |

The Step26 proposal is clearly responsible for edge-add recovery: edge-add proposal recall improves from `0.1457` to `0.3985`, and add improves from `0.1120` to `0.3445`. The Step26 rewrite does not improve edge-add beyond what the Step26 proposal already enables.

## Oracle-Scope Rewrite Check

Oracle scope isolates rewrite quality when proposal coverage is perfect.

Oracle Step26 rewrite vs oracle RFT1:

| Slice | Changed Edge Delta | Add Delta | Delete Delta | Context Delta |
|---|---:|---:|---:|---:|
| overall | -0.0268 | -0.1006 | +0.0576 | -0.0011 |
| strongly_interacting | -0.0775 | -0.0833 | +0.0000 | -0.0017 |

The Step26 rewrite is not better than RFT1 for the intended strongly-interacting changed/add target under oracle scope. It improves overall delete under oracle scope, but that improvement does not translate to the strongly-interacting slice and does not compensate for the changed/add loss.

## Decision Answers

1. Is Step26's main positive contribution primarily proposal-side?

Yes. With RFT1 frozen, the Step26 proposal raises overall changed-edge from `0.2573` to `0.3758`, add from `0.1111` to `0.3040`, edge recall from `0.7095` to `0.8100`, and lowers out-of-scope miss from `0.5481` to `0.3926`. On strongly-interacting examples, it raises changed-edge from `0.1008` to `0.2558` and add from `0.0833` to `0.2500`.

2. Is Step26's main negative contribution primarily rewrite-side?

Mostly yes for delete and target-edit quality. With baseline P2 scope, Step26 rewrite lowers overall changed-edge from `0.2573` to `0.2371` and delete from `0.4245` to `0.3813`; on strongly-interacting examples it lowers delete from `0.3333` to `0.1111`. Under Step26 proposal scope, switching from RFT1 to Step26 rewrite lowers strongly-interacting delete from `0.3333` to `0.0000`. The broad full/context drop is mostly caused by the larger Step26 proposal scope, but the damaging delete collapse is rewrite-side.

3. Does the evidence justify the next branch as proposal-coverage-emphasized joint + rewrite regularization?

Yes, if this substrate continues. The useful signal is the Step26 proposal direction, but the rewrite update needs stronger protection. A next branch should preserve the coverage-emphasized proposal gains while regularizing rewrite toward safer RFT1-like behavior, especially for delete and context stability.

4. Or is the cleaner next move to preserve proposal gains while abandoning the Step26 rewrite update?

The cleanest immediate next move is to preserve proposal gains while abandoning or heavily constraining the Step26 rewrite update. The strongest matrix row is `Step26 proposal + RFT1 rewrite`, not full Step26. That row is still not a stable default because full/context-edge drop sharply, but it cleanly shows the component direction to keep.

## Outcome

Stable defaults remain unchanged:

- noisy broad default: `RFT1 + calibrated P2`
- clean broad default: `W012`
- interaction-aware alternative: `I1520`

Step27 does not promote Step26. It reframes Step26 as two separable facts:

- retain: Step26 proposal direction has real noisy interaction coverage/add signal
- reject or constrain: Step26 rewrite update is not useful for the intended strong changed/add target and is responsible for the strong delete collapse

Recommended next line, if continuing this substrate:

- proposal-coverage-emphasized joint training with rewrite regularization, or a branch that explicitly keeps the Step26 proposal direction while reverting to safer RFT1-style rewrite behavior

Artifacts:

- `artifacts/step27_step26_factorization/baseline_proposal_baseline_rewrite.json`
- `artifacts/step27_step26_factorization/step26_proposal_baseline_rewrite.json`
- `artifacts/step27_step26_factorization/baseline_proposal_step26_rewrite.json`
- `artifacts/step27_step26_factorization/step26_proposal_step26_rewrite.json`
- `artifacts/step27_step26_factorization/oracle_scope_baseline_rewrite.json`
- `artifacts/step27_step26_factorization/oracle_scope_step26_rewrite.json`
- `artifacts/step27_step26_factorization/summary.json`
- `artifacts/step27_step26_factorization/summary.csv`
