# Step30 rev27: Ambiguity-Aware Latent Calibration

## Summary

rev27 added one narrow ambiguity-risk auxiliary head on top of the integrated
`rescue_candidate_latent`. The ordinary rev6-style edge path was kept frozen;
only the rescue-candidate latent/classifier/binary-calibration/ambiguity heads
were trainable from the rev26 checkpoint.

The rev27 admission score was:

```text
safe_vs_false_binary_prob - ambiguity_risk_prob
```

inside the existing rescue scope:

```text
relation_hint < 0.50 and pair_support_hint >= 0.55
```

This was intended to treat `ambiguous_rescue_candidate` as a first-class
admission risk instead of relying only on safe-vs-low-hint-false calibration.

## Result

rev27 did not clear the recovery gate. It slightly reduced ambiguous selected
admissions versus rev26, but the reduction was small and was offset by more
low-hint false admissions plus slightly lower safe rescue.

No Step30c backend rerun was performed.

## Candidate Quality

| row | scope | AP | AUROC | precision@2975 | budget F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| rev24 integrated safe prob | safe vs low-hint false | 0.7511 | 0.7290 | 0.7103 | 0.6991 |
| rev26 safe-vs-false binary | safe vs low-hint false | 0.9058 | 0.8640 | 0.8104 | 0.7977 |
| rev27 safe-vs-false binary | safe vs low-hint false | 0.9113 | 0.8685 | 0.8155 | 0.8026 |
| rev24 softmax ambiguity | ambiguity detection | 0.8861 | 0.8398 | 0.9297 | 0.4498 |
| rev26 softmax ambiguity | ambiguity detection | 0.8907 | 0.8472 | 0.9341 | 0.4519 |
| rev27 ambiguity-risk head | ambiguity detection | 0.8765 | 0.8322 | 0.9207 | 0.4454 |

## Recovery Summary

| row | overall F1 | clean F1 | noisy precision | noisy recall | noisy F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rev6 | 0.7452 | 0.8282 | 0.6306 | 0.7049 | 0.6657 |
| rev21 | 0.7503 | 0.8368 | 0.6310 | 0.7095 | 0.6680 |
| rev25 class-aware | 0.7379 | 0.8282 | 0.5912 | 0.7376 | 0.6563 |
| rev26 calibrated | 0.7374 | 0.8282 | 0.5903 | 0.7365 | 0.6553 |
| rev27 ambiguity-aware | 0.7373 | 0.8282 | 0.5902 | 0.7364 | 0.6552 |

## Rescue-Scope Selected Additions

| row | precision | recall | F1 | safe admit | low-hint false admit | ambiguous admit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rev24 safe-only | 0.2339 | 0.2267 | 0.2303 | 0.2267 | 0.0391 | 0.2340 |
| rev25 class-aware | 0.2521 | 0.2443 | 0.2481 | 0.2443 | 0.1271 | 0.2048 |
| rev26 calibrated | 0.2434 | 0.2358 | 0.2395 | 0.2358 | 0.0464 | 0.2291 |
| rev27 ambiguity-aware | 0.2424 | 0.2349 | 0.2385 | 0.2349 | 0.0605 | 0.2256 |

## Targeted Diagnostics

| row | hint-missed recall | hint-missed avg score | hint-supported FP error | HM vs hard-negative win |
| --- | ---: | ---: | ---: | ---: |
| rev6 | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev21 | 0.3003 | 0.4389 | 0.4688 | 0.3568 |
| rev25 class-aware | 0.4134 | 0.2429 | 0.4688 | 0.1993 |
| rev26 calibrated | 0.4089 | 0.3942 | 0.4688 | 0.2983 |
| rev27 ambiguity-aware | 0.4083 | 0.2785 | 0.4688 | 0.1685 |

## Diagnosis

rev27 confirms that ambiguous rescue candidates are the right failure axis, but
the new standalone ambiguity-risk head did not improve the boundary. The old
rev26 3-way softmax ambiguity probability remained stronger than the new
ambiguity head on AP/AUROC and precision at budget. Admission changed only
slightly: ambiguous selected admissions fell from `0.2291` to `0.2256`, while
low-hint false admissions rose from `0.0464` to `0.0605`.

The recovery gate is not met:

```text
rev27 noisy F1 = 0.6552
rev6 noisy F1  = 0.6657
```

The integrated latent remains useful as a diagnostic mechanism, but rev27 does
not make it a retained recovery operating point.
