# Step30 Rev24 Rescue Candidate Latent Integration Probe

Rev24 tests whether the rev23 rescue-candidate latent signal survives when
attached to the Step30 recovery encoder itself.

## Mechanism

- Base path: rev6-style ordinary edge recovery, initialized from
  `checkpoints/step30_encoder_recovery_rev6/best.pt`.
- New path: a tiny `rescue_candidate_latent` projection plus 3-way classifier.
- Scope: only low-relation, pair-support rescue candidates:
  `relation_hint < 0.50` and `pair_support_hint >= 0.55`.
- Target classes:
  `safe_missed_true_edge`, `low_hint_pair_support_false_admission`,
  `ambiguous_rescue_candidate`.
- Trainable parameters: only the rescue-candidate latent/classifier head.
- Decode use: the safe-class score is used only for bounded rescue-scope
  admissions; it is not used as a global all-pair edge scorer.

## Main Result

The integrated latent head recovers much of the rev23 candidate separability
signal, but it does not become a clean recovery operating point.

| row | noisy precision | noisy recall | noisy F1 |
| --- | ---: | ---: | ---: |
| rev6 | 0.6306 | 0.7049 | 0.6657 |
| rev21 | 0.6310 | 0.7095 | 0.6680 |
| rev24 additive simulation | 0.5893 | 0.7353 | 0.6543 |

Official repo-style selective-rescue eval with `rescue_budget_fraction=0.03`
also stays below the rev6 recovery gate:

| row | noisy precision | noisy recall | noisy F1 |
| --- | ---: | ---: | ---: |
| rev24 aux selective rescue | 0.5753 | 0.7383 | 0.6367 |

## Rescue Candidate Quality

| metric | rev23 probe reference | rev24 integrated |
| --- | ---: | ---: |
| safe-vs-unsafe AP | 0.4293 | 0.4284 |
| safe-vs-unsafe AUROC | 0.7214 | 0.7108 |
| rev21-budget/probe F1 | 0.4518 | not retained |

The score remains meaningfully more separable than scalar residual shaping,
but bounded additive admission still selects too many ambiguous false
admissions.

## Targeted Diagnostics

| row | hint-missed recall | hint-missed avg score | hint-supported FP error | HM vs hard-negative win |
| --- | ---: | ---: | ---: | ---: |
| rev6 | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev21 | 0.3003 | 0.4389 | 0.4688 | 0.3568 |
| rev24 | 0.4039 | 0.4440 | 0.4688 | 0.3596 |

Rev24 preserves the missed-edge rescue direction, but the extra admissions
still cost too much precision.

## Gate Decision

Rev24 does not clear the recovery gate. No Step30c backend rerun was run.

The useful conclusion is narrower: first-class rescue-candidate latent
separation is real, but the current integrated admission rule is not selective
enough. The next smallest justified move is not backend integration; it is a
rescue-scope admission objective or calibration that explicitly separates
safe positives from the large ambiguous candidate pool.
