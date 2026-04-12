# Step 29: Retained Noisy Interaction-Aware Proposal Branch

## Scope

Step 29 is a consolidation pack. No model was trained.

Main question:

Should `Step26 proposal + RFT1 rewrite` be formally retained as the project's noisy interaction-aware branch candidate, even though it is not the broad noisy default?

Inputs are existing frozen evaluations from Steps 22-28:

- noisy P2 + RFT1
- noisy P2 + I1520
- Step23 proposal-only + RFT1
- Step24 light joint
- Step26 full joint
- Step26 proposal + RFT1
- Step28 anchored joint

中文简述：这一步不是新方法，而是“定档”。目标是把 Step26 proposal + RFT1 明确保存为 branch candidate，同时不改变 broad default。

## Artifacts

Evaluator:

- `train/eval_step29_retained_proposal_branch.py`

Machine-readable outputs:

- `artifacts/step29_retained_proposal_branch/summary.json`
- `artifacts/step29_retained_proposal_branch/summary.csv`

All required rows were available. The optional Step23 proposal-only row was also included.

## Overall Comparison

Final-step noisy Step5 metrics:

| Run | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + RFT1 | 0.8754 | 0.8821 | 0.2573 | 0.1111 | 0.4245 | 0.7095 | 0.5481 |
| noisy P2 + I1520 | 0.8361 | 0.8410 | 0.3714 | 0.1090 | 0.6715 | 0.7095 | 0.5481 |
| Step23 proposal-only + RFT1 | 0.8813 | 0.8882 | 0.2349 | 0.0797 | 0.4125 | 0.6653 | 0.5951 |
| Step24 light joint | 0.8762 | 0.8829 | 0.2472 | 0.1195 | 0.3933 | 0.7159 | 0.5347 |
| Step26 full joint | 0.8264 | 0.8313 | 0.3557 | 0.3061 | 0.4125 | 0.8100 | 0.3926 |
| Step26 proposal + RFT1 | 0.8264 | 0.8312 | 0.3758 | 0.3040 | 0.4580 | 0.8100 | 0.3926 |
| Step28 anchored joint | 0.8277 | 0.8327 | 0.3490 | 0.2935 | 0.4125 | 0.8075 | 0.4004 |

Step26 proposal + RFT1 vs noisy P2 + RFT1:

| Metric | Delta |
|---|---:|
| full-edge | -0.0490 |
| context-edge | -0.0509 |
| changed-edge | +0.1186 |
| add | +0.1929 |
| delete | +0.0336 |
| proposal edge recall | +0.1005 |
| out-of-scope miss | -0.1555 |

The retained candidate is clearly not the broad default because it gives back about five points of full/context-edge stability. But it is the cleanest edit/coverage branch on this noisy interaction substrate.

## Strongly-Interacting Slice

Final-step strongly-interacting metrics:

| Run | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| noisy P2 + RFT1 | 0.8771 | 0.8808 | 0.1008 | 0.0833 | 0.3333 | 0.7741 | 0.8527 |
| noisy P2 + I1520 | 0.8378 | 0.8414 | 0.0853 | 0.0750 | 0.2222 | 0.7741 | 0.8527 |
| Step23 proposal-only + RFT1 | 0.8820 | 0.8858 | 0.0930 | 0.0917 | 0.1111 | 0.7306 | 0.8527 |
| Step24 light joint | 0.8793 | 0.8831 | 0.0853 | 0.0833 | 0.1111 | 0.7701 | 0.8372 |
| Step26 full joint | 0.8287 | 0.8314 | 0.2403 | 0.2583 | 0.0000 | 0.8523 | 0.6744 |
| Step26 proposal + RFT1 | 0.8254 | 0.8281 | 0.2558 | 0.2500 | 0.3333 | 0.8523 | 0.6744 |
| Step28 anchored joint | 0.8303 | 0.8331 | 0.2326 | 0.2500 | 0.0000 | 0.8515 | 0.6822 |

Step26 proposal + RFT1 vs noisy P2 + RFT1 on strongly-interacting examples:

| Metric | Delta |
|---|---:|
| full-edge | -0.0517 |
| context-edge | -0.0527 |
| changed-edge | +0.1550 |
| add | +0.1667 |
| delete | +0.0000 |
| proposal edge recall | +0.0782 |
| out-of-scope miss | -0.1783 |

This is the main retention argument. The candidate recovers a large share of the strongly-interacting changed/add headroom while preserving baseline delete.

中文解释：它不是更稳，而是更“敢改”。强交互 add/changed 被真正打开，同时没有像 Step26/28 那样把 delete 打成 0。

## Why It Beats The Local Alternatives

Compared with Step24:

- Step24 preserves broad stability but does not improve the intended strong changed/add slice.
- Step26 proposal + RFT1 improves strong changed-edge by `+0.1705` and strong add by `+0.1667` vs Step24.
- It also improves strong delete by `+0.2222` vs Step24.

Compared with Step26 full joint:

- The proposal coverage is identical.
- Overall changed-edge is higher: `0.3758` vs `0.3557`.
- Overall delete is higher: `0.4580` vs `0.4125`.
- Strong delete is preserved: `0.3333` vs `0.0000`.

Compared with Step28 anchored joint:

- Proposal coverage is effectively the same.
- Overall changed-edge is higher: `0.3758` vs `0.3490`.
- Overall delete is higher: `0.4580` vs `0.4125`.
- Strong delete is preserved: `0.3333` vs `0.0000`.

Compared with I1520 under noisy P2:

- I1520 has stronger overall delete, but it does not solve the noisy strongly-interacting changed/add target.
- Strong changed/add/delete for I1520 are `0.0853` / `0.0750` / `0.2222`.
- Strong changed/add/delete for Step26 proposal + RFT1 are `0.2558` / `0.2500` / `0.3333`.
- Step26 proposal + RFT1 is the better noisy interaction-aware branch for the amplified strong-interaction slice, though I1520 remains an aggressive delete-oriented alternative.

## Corruption Breakdown

Step26 proposal + RFT1 final-step metrics:

| Corruption | Full Edge | Context Edge | Changed Edge | Add | Delete | Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| N1 | 0.8492 | 0.8541 | 0.3960 | 0.3270 | 0.4748 | 0.8493 | 0.3658 |
| N2 | 0.8242 | 0.8291 | 0.3557 | 0.3082 | 0.4101 | 0.8142 | 0.3658 |
| N3 | 0.8059 | 0.8104 | 0.3758 | 0.2767 | 0.4892 | 0.7665 | 0.4463 |

The candidate remains edit-active across N1/N2/N3. The broad context/full-edge cost persists across corruption regimes.

## Event Family Notes

All-step Step26 proposal + RFT1 metrics:

| Event Family | Changed Edge | Add | Delete | Context Edge | Edge Recall | Out-of-Scope Miss |
|---|---:|---:|---:|---:|---:|---:|
| edge_add | 0.3445 | 0.3445 | NA | 0.8249 | 0.3985 | 0.6015 |
| edge_delete | 0.4391 | NA | 0.4391 | 0.8288 | 0.8656 | 0.1344 |
| motif_type_flip | NA | NA | NA | 0.8317 | 0.9043 | NA |
| node_state_update | NA | NA | NA | 0.8224 | 0.8970 | NA |

The branch's main event-family value is edge-add recovery. It also keeps aggregate edge-delete usable when paired with RFT1.

## Decision Answers

1. Is `Step26 proposal + RFT1` worth preserving as the project's noisy interaction-aware branch candidate?

Yes. It is the cleanest retained point on the noisy multievent interaction substrate. It preserves the Step26 proposal coverage gains, recovers strong changed/add, and avoids the Step26/Step28 strong-delete collapse.

2. Why is it not the broad noisy default?

Because it gives back too much broad stability. Overall full-edge falls from `0.8754` to `0.8264`, and context-edge falls from `0.8821` to `0.8312` relative to noisy P2 + RFT1. The stable noisy default must remain `RFT1 + calibrated P2`.

3. In what sense is it better than Step24/26/28?

It is better than Step24 because it actually improves the intended strong changed/add target. It is better than full Step26 and Step28 because it keeps the Step26 proposal coverage direction while avoiding their strong-delete collapse. It is not more stable than those rows; it is cleaner because the useful proposal component is retained without the harmful rewrite drift.

4. Does the evidence justify parking this substrate line after preserving this candidate, rather than continuing local tweaks?

Yes. Step23, Step24, Step26, Step27, and Step28 have isolated the local tradeoff. More local proposal-only, light-joint, or weak-anchor tweaks are unlikely to change the conclusion. The clean handoff is to preserve `Step26 proposal + RFT1` as the branch candidate and park this noisy interaction substrate line unless a genuinely new mechanism is introduced.

## Retained Defaults And Branches

Stable defaults remain unchanged:

- noisy broad default: `RFT1 + calibrated P2`
- clean broad default: `W012`
- interaction-aware alternative: `I1520`

New retained branch candidate:

- noisy interaction-aware branch candidate: `Step26 proposal + RFT1`

This branch should be used when the evaluation priority is strongly-interacting changed/add recovery under noisy multievent interaction, not when broad context/full-edge stability is the priority.

Artifacts:

- `artifacts/step29_retained_proposal_branch/summary.json`
- `artifacts/step29_retained_proposal_branch/summary.csv`
