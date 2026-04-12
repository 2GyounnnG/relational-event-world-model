# Step22-29 Noisy Multievent Interaction Summary

## Scope

This document consolidates the noisy multievent interaction line from Steps 22-29. It does not introduce a new method, checkpoint, or experiment.

Project scope remains unchanged:

- structured synthetic graph-event world model
- proposal local event scope, then rewrite local region
- no raw-image, real-world, hypergraph, or LLM integration

## Stable Defaults

Stable defaults remain unchanged:

- broad clean default: `W012`
- noisy broad default: `RFT1 + calibrated P2`
- interaction-aware alternative: `I1520`
- consistency reference: `C005`
- rollout-aware reference: `R050`
- noisy proposal front-end: calibrated `P2` with `node_threshold = 0.15`, `edge_threshold = 0.10`

## Step22-29 Summary

Step22 established the noisy multievent interaction benchmark by combining Step 5 interaction complexity with Step 6 noisy structured observation. The family did not catastrophically collapse, and `RFT1 + calibrated P2` remained the safest broad noisy stack. The main amplified bottleneck was strongly interacting sequences.

Step23 tested proposal-only noisy interaction-aware adaptation from `P2` with `RFT1` frozen. It did not solve the bottleneck. The candidate mainly became more conservative: broad full/context-edge improved, but proposal edge recall dropped, out-of-scope miss worsened, and changed/add/delete weakened.

Step24 tested a first light joint noisy interaction-aware proposal+rewrite fine-tune. It was informative but not promotable: overall proposal coverage improved slightly, but strongly interacting changed-edge/delete did not improve.

Step25 used oracle event scope to measure headroom. It showed large proposal-coverage headroom for noisy strongly interacting changed/add behavior, while delete remained partly rewrite-side because it did not recover under oracle scope. Step24 rewrite was not better than `RFT1` under oracle scope.

Step26 introduced coverage-emphasized joint training. It recovered real strongly interacting changed/add headroom and improved proposal coverage, but broad full/context-edge dropped substantially and strongly interacting delete collapsed.

Step27 factorized Step26 into proposal-side and rewrite-side components. The positive contribution was mainly proposal-side; the negative contribution was mainly rewrite-side. The clean retained signal became: keep the Step26 proposal direction, but do not keep the Step26 rewrite update as-is.

Step28 added RFT1 anchoring during joint training. It did not improve on the factorized reference and did not fix the rewrite-side delete/context problem.

Step29 formalized the retained branch: `Step26 proposal + RFT1`.

中文总结：Step22-29 的结论不是“换默认模型”，而是“保留一个有用分支”。Step26 的 proposal 方向打开了强交互 changed/add，但 Step26/28 的 rewrite 更新会伤害强交互 delete 和 broad stability，所以最终保留的是 `Step26 proposal + RFT1`。

## Current Disposition

Stable defaults remain unchanged.

Retained noisy interaction-aware branch candidate: `Step26 proposal + RFT1`. It should be used when the evaluation priority is strongly-interacting changed/add recovery under noisy multievent interaction, not when broad context/full-edge stability is the priority.

This branch is not the broad noisy default because broad full-edge / context-edge remain materially weaker than `RFT1 + calibrated P2`.

The Step22-29 local substrate line should now be parked unless a genuinely new mechanism is introduced.

## Why The Branch Is Retained

`Step26 proposal + RFT1` preserves the useful Step26 proposal-side direction:

- stronger proposal coverage on the noisy multievent interaction substrate
- stronger strongly interacting changed/add behavior
- no Step26/Step28 rewrite-side strong-delete collapse

It is cleaner than the full Step26 and Step28 rows because it keeps the proposal gain while reverting to the safer `RFT1` rewrite behavior.

## Why It Is Not The Broad Noisy Default

The retained branch is more edit-active on the noisy interaction substrate, but broad full-edge and context-edge stability are materially weaker than the stable noisy stack.

Use `RFT1 + calibrated P2` as the broad noisy default. Use `Step26 proposal + RFT1` only as a noisy interaction-aware branch candidate when the target is strongly interacting changed/add recovery and the broad stability tradeoff is acceptable.

## Handoff

Do not continue local Step22-29 tweaks such as proposal-only noisy interaction micro-tuning, Step26/28 recipe tweaks, threshold changes, or broad joint sweeps. The clean next phase should start from the consolidated defaults and only reopen this substrate if there is a genuinely new mechanism.

Supporting reports:

- `docs/STEP22_NOISY_MULTIEVENT_INTERACTION.md`
- `docs/STEP23_NOISY_INTERACTION_AWARE_PROPOSAL.md`
- `docs/STEP24_NOISY_INTERACTION_JOINT.md`
- `docs/STEP25_NOISY_MULTIEVENT_ORACLE_HEADROOM.md`
- `docs/STEP26_NOISY_INTERACTION_JOINT_DEEPER.md`
- `docs/STEP27_STEP26_FACTORIZATION.md`
- `docs/STEP28_RFT1_ANCHORED_JOINT.md`
- `docs/STEP29_RETAINED_PROPOSAL_BRANCH.md`
