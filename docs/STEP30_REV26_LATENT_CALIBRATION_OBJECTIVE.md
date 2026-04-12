# Step30 Rev26 Latent Calibration Objective

Rev26 adds one small latent-side objective to the rev24 integrated
`rescue_candidate_latent` setup.

## Mechanism

- Base model: rev24 integrated rescue-candidate latent.
- Ordinary path: frozen rev6-style global recovery path.
- New head: one binary calibration head over the rescue-candidate latent.
- Supervision scope: rescue candidates only:
  `relation_hint < 0.50` and `pair_support_hint >= 0.55`.
- Binary target:
  - positive: `safe_missed_true_edge`
  - negative: `low_hint_pair_support_false_admission`
  - ignored: `ambiguous_rescue_candidate`
- Decode/admission score:
  `P(binary_safe) - P(ambiguous_rescue_candidate)`.

The intent was to improve the latent safe-vs-false boundary rather than add
another decode-only margin rule.

## Candidate Quality

| row | AP | AUROC | precision@budget | budget F1 |
| --- | ---: | ---: | ---: | ---: |
| rev23 probe reference | 0.4293 | 0.7214 | 0.3686 | 0.4518 |
| rev24 integrated safe prob | 0.7511 | 0.7290 | 0.7103 | 0.6991 |
| rev25 class margin | 0.7184 | 0.6858 | 0.6743 | 0.6637 |
| rev26 binary calibrated | 0.9058 | 0.8640 | 0.8104 | 0.7977 |

The binary calibration objective substantially improves the intended
safe-vs-low-hint-false separation.

## Recovery Result

| row | noisy precision | noisy recall | noisy F1 |
| --- | ---: | ---: | ---: |
| rev6 | 0.6306 | 0.7049 | 0.6657 |
| rev21 | 0.6310 | 0.7095 | 0.6680 |
| rev24 safe-only | 0.5893 | 0.7353 | 0.6543 |
| rev25 class-aware | 0.5912 | 0.7376 | 0.6563 |
| rev26 calibrated | 0.5903 | 0.7365 | 0.6553 |

Rev26 improves the intended classifier boundary but does not improve the global
recovery operating point over rev25, and remains below rev6.

## Rescue Scope

| row | precision | recall | F1 | safe admit | low-hint false admit | ambiguous admit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rev6 | 0.3299 | 0.5062 | 0.3995 | 0.5062 | 0.4752 | 0.2121 |
| rev21 | 0.3414 | 0.5407 | 0.4186 | 0.5407 | 0.5272 | 0.2032 |
| rev24 safe-only | 0.2928 | 0.7329 | 0.4184 | 0.7329 | 0.5143 | 0.4461 |
| rev25 class-aware | 0.2998 | 0.7505 | 0.4285 | 0.7505 | 0.6023 | 0.4169 |
| rev26 calibrated | 0.2964 | 0.7420 | 0.4236 | 0.7420 | 0.5216 | 0.4412 |

Selected additions only:

| row | precision | recall | F1 | safe admit | low-hint false admit | ambiguous admit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rev24 safe-only | 0.2339 | 0.2267 | 0.2303 | 0.2267 | 0.0391 | 0.2340 |
| rev25 class-aware | 0.2521 | 0.2443 | 0.2481 | 0.2443 | 0.1271 | 0.2048 |
| rev26 calibrated | 0.2434 | 0.2358 | 0.2395 | 0.2358 | 0.0464 | 0.2291 |

## Gate Decision

Rev26 does not clear the recovery gate.

- It clearly improves safe-vs-low-hint-false separability.
- It suppresses low-hint false additions relative to rev25.
- It gives back too much ambiguous admission.
- Noisy edge F1 remains below rev6.

No Step30c backend rerun was run.

## Diagnosis

The bottleneck has moved: safe-vs-low-hint-false separation is no longer the
main blocker after calibration. The remaining blocker is ambiguous rescue
candidate handling. Another safe-vs-false-only objective is unlikely to fix the
global operating point without first making ambiguity a first-class calibrated
decision.
