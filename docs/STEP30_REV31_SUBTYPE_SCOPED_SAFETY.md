# Step30 Rev31 Subtype-Scoped Safety Objective

## Scope

Rev31 keeps the rev30 benchmark and `positive_ambiguity_safety_hint`. It does not add a new cue, does not run backend integration, and does not alter the ordinary rev6-style edge path.

## Change

Rev31 adds a tiny binary `weak_positive_ambiguity_safety_head` on top of the existing rescue-candidate latent.

The auxiliary loss applies only to candidates matching the existing `weak_positive_ambiguous` subtype rule:

- `relation_hint < 0.50`
- `pair_support_hint >= 0.55`
- `pair_evidence_bundle.positive - pair_evidence_bundle.warning >= 0.12`
- corroboration, endpoint compatibility, or signed witness weakly supports the pair
- the candidate is still ambiguous by the existing relation/support/bundle-margin ambiguity rule

The target is simply GT-positive vs GT-negative inside this subtype. Training starts from rev30, freezes the ordinary/global path, and trains only rescue-candidate components plus the new head.

## Key Results

### Weak-Positive Ambiguity

| row | AP | AUROC | P@1462 | budget F1 |
| --- | ---: | ---: | ---: | ---: |
| rev30 offline probe | 0.5844 | 0.7501 | 0.4473 | 0.5489 |
| rev30 integrated | 0.5510 | 0.7305 | 0.4316 | 0.5296 |
| rev31 subtype head | 0.5799 | 0.7500 | 0.4514 | 0.5539 |

Rev31 narrows the rev30 probe-to-integration gap at the subtype-classification level.

### Rescue Scope

| row | precision | recall | F1 | weak-positive selected | weak-positive P | weak-positive R |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rev26 calibrated | 0.2434 | 0.2358 | 0.2395 | 1462 | 0.2579 | 0.4093 |
| rev30 integrated | 0.2474 | 0.2397 | 0.2435 | 1402 | 0.2760 | 0.4202 |
| rev31 subtype-scoped | 0.2350 | 0.2277 | 0.2313 | 1036 | 0.3108 | 0.3496 |
| rev31 head same rev30 weak count | 0.2497 | 0.2420 | 0.2458 | 1402 | 0.2810 | 0.4278 |

The subtype head is sharper, but the default admission score under-selects weak-positive candidates relative to rev30.

### Recovery

| row | overall F1 | clean F1 | noisy precision | noisy recall | noisy F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rev6 | 0.7452 | 0.8282 | 0.6306 | 0.7049 | 0.6657 |
| rev26 calibrated | 0.7374 | 0.8282 | 0.5903 | 0.7365 | 0.6553 |
| rev30 integrated | 0.7376 | 0.8282 | 0.5907 | 0.7370 | 0.6558 |
| rev31 subtype-scoped | 0.7368 | 0.8282 | 0.5894 | 0.7354 | 0.6544 |
| rev31 head same rev30 weak count | 0.7377 | 0.8282 | 0.5910 | 0.7373 | 0.6561 |
| trivial with rev30 cue | 0.5168 | 0.5741 | 0.4489 | 0.4344 | 0.4416 |

## Diagnosis

Rev31 successfully transfers much of the rev30 offline probe boundary into the integrated latent head. However, the retained integrated admission score is not calibrated well against the rest of the rescue-scope candidates: it admits fewer weak-positive candidates, increasing weak-positive precision but losing too much safe rescue recall.

The diagnostic same-count replacement shows the learned head is useful, but that is not enough to clear the recovery gate. No Step30c backend rerun was run.
