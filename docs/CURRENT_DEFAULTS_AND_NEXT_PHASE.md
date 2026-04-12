# Current Defaults and Next Phase

## 1. Current defaults to use

- broad system default: `W012`
- noisy structured observation default: `RFT1 + calibrated P2`
- interaction-aware alternative: `I1520`
- consistency reference: `C005`
- rollout-aware reference: `R050`
- noisy interaction-aware branch candidate: `Step26 proposal + RFT1` (not the broad noisy default)

## 2. Models and checkpoints to preserve

- `W012`: [checkpoints/fp_keep_w012/best.pt](/Users/richwang/newprogram/checkpoints/fp_keep_w012/best.pt)
- `WG025`: [checkpoints/fp_keep_weighted025_w012/best.pt](/Users/richwang/newprogram/checkpoints/fp_keep_weighted025_w012/best.pt)
- `DR005`: [checkpoints/w012_delete_rescue005/best.pt](/Users/richwang/newprogram/checkpoints/w012_delete_rescue005/best.pt)
- `C005`: [checkpoints/step3_consistency_w005/best.pt](/Users/richwang/newprogram/checkpoints/step3_consistency_w005/best.pt)
- `R050`: [checkpoints/step4_rollout_w050/best.pt](/Users/richwang/newprogram/checkpoints/step4_rollout_w050/best.pt)
- `I1520`: [checkpoints/step5_interaction_i1520/best.pt](/Users/richwang/newprogram/checkpoints/step5_interaction_i1520/best.pt)
- `P2` proposal: [checkpoints/proposal_noisy_obs_p2/best.pt](/Users/richwang/newprogram/checkpoints/proposal_noisy_obs_p2/best.pt)
- calibrated noisy-observation thresholds for `P2`: `node_threshold = 0.15`, `edge_threshold = 0.10`
- `RFT1` rewrite: [checkpoints/step6_noisy_rewrite_rft1/best.pt](/Users/richwang/newprogram/checkpoints/step6_noisy_rewrite_rft1/best.pt)
- `Step26` proposal branch: [checkpoints/step26_noisy_interaction_joint_deeper/proposal_best.pt](/Users/richwang/newprogram/checkpoints/step26_noisy_interaction_joint_deeper/proposal_best.pt)

## 3. Models that are informative but not current defaults

- `WG025`: best Step 2 edit-preserving alternative, but not the broad default
- `DR005`: useful delete-rescue anchor, but too specialized to carry forward as the main line
- `J05`: informative first joint noisy proposal+rewrite result, but not better overall than `RFT1 + calibrated P2`
- `W012 + calibrated P2`: simpler noisy-observation baseline that remains useful for ablations, but not the current Step 6 main candidate
- `Step26 proposal + RFT1`: retained noisy interaction-aware branch candidate; it improves noisy strongly-interacting changed/add recovery, but broad full/context-edge stability remains weaker than `RFT1 + calibrated P2`

## 4. Reopened lines to avoid for now

- Step 2 keep/rescue retuning around the existing `W012` / `WG025` / `DR005` frontier
- Step 3 exact reverse-order final-state matching as a main consistency substrate
- Step 4 small rollout-loss retuning around the current short-horizon line
- Step 6 threshold-only calibration retuning beyond the current `P2` operating point
- Step 6 regime-aware threshold calibration retesting
- Step 6 temperature-only proposal calibration retesting
- Step 22-29 noisy multievent interaction local substrate tweaks unless a genuinely new mechanism is introduced
- more proposal-only noisy interaction micro-tuning after Step 23
- more Step 26 / Step 28 joint-recipe tweaks on the noisy interaction substrate

## 5. Step22-29 current disposition

Step22 established the noisy multievent interaction benchmark. Step23 proposal-only adaptation failed. Step24 first light joint training was informative but not promotable. Step25 oracle-headroom showed large proposal headroom and remaining rewrite-side delete issues. Step26 recovered strongly-interacting changed/add headroom but harmed broad stability and strong delete. Step27 factorization showed gains are mainly proposal-side and harms mainly rewrite-side. Step28 anchoring did not fix the rewrite problem. Step29 confirmed `Step26 proposal + RFT1` as the retained branch candidate.

Retained noisy interaction-aware branch candidate: `Step26 proposal + RFT1`. It should be used when the evaluation priority is strongly-interacting changed/add recovery under noisy multievent interaction, not when broad context/full-edge stability is the priority.

Stable defaults remain unchanged. The Step22-29 local substrate line should be parked unless a genuinely new mechanism is introduced.

## 6. Next-phase starting point

Future work should start from the consolidated defaults instead of reopening the earlier tuning loops. Treat `W012` as the broad clean default, `RFT1 + calibrated P2` as the noisy structured observation stack, `I1520` as the interaction-aware alternative, `C005` as the sequential consistency reference, and `R050` as the rollout-aware reference. Preserve `Step26 proposal + RFT1` as the noisy interaction-aware branch candidate when that substrate is directly relevant, but do not continue the Step22-29 local line without a genuinely new mechanism. The next phase should ask a new question on top of these stable points rather than spending more effort on saturated local calibration or micro-sweeps.
