# Step30 rev28: Rescue Ambiguity Subtype Probe

## Summary

rev28 is a diagnostic-only probe. It does not add a new weak observation cue,
train a new recovery model, run backend integration, or continue the rev23-rev27
integrated latent micro-line.

The probe asks whether the broad `ambiguous_rescue_candidate` bucket is too
coarse. It uses existing rescue-scope signals only:

- relation hint
- pair support hint
- signed pair witness
- `pair_evidence_bundle` channels
- rev6 base score
- rev24/rev26 rescue latent probabilities

## Subtype Rules

The probe first defines an `ambiguous_signal` regime independent of GT:

```text
rescue candidate
and (
  relation_hint >= 0.45
  or pair_support_hint < 0.65
  or abs(bundle_positive - bundle_warning) < 0.10
)
```

It then partitions that regime with explicit rules:

| subtype | rule |
| --- | --- |
| weak_positive_ambiguous | `bundle_positive - bundle_warning >= 0.12` plus corroboration, endpoint compatibility, or signed witness support |
| warning_dominated_ambiguous | `bundle_warning - bundle_positive >= 0.12` |
| conflicting_evidence_ambiguous | near-tie positive/warning evidence with high evidence magnitude |
| relation_borderline_ambiguous | relation hint near the rescue boundary after stronger evidence rules |
| low_confidence_ambiguous | remaining ambiguous-signal candidates |

## Subtype Outcome Summary

| subtype | count | GT+ rate | rev26 admit | rev26 selected precision | rev26 positive recall contribution | rev6 score | rev26 binary | rev26 ambiguous |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weak_positive_ambiguous | 3,154 | 0.2920 | 0.4635 | 0.2579 | 0.4093 | 0.4660 | 0.9349 | 0.4326 |
| warning_dominated_ambiguous | 4,877 | 0.1298 | 0.1183 | 0.1924 | 0.1754 | 0.4513 | 0.8661 | 0.6220 |
| conflicting_evidence_ambiguous | 1,472 | 0.2072 | 0.2024 | 0.2617 | 0.2557 | 0.4701 | 0.7765 | 0.4591 |
| relation_borderline_ambiguous | 467 | 0.3383 | 0.1242 | 0.1724 | 0.0633 | 0.6225 | 0.9457 | 0.4001 |
| low_confidence_ambiguous | 1,639 | 0.1629 | 0.2361 | 0.1809 | 0.2622 | 0.4310 | 0.7825 | 0.5049 |

The broad ambiguity signal regime is not homogeneous: the GT-positive rate
ranges from `0.1298` to `0.3383`.

## False Admission Composition

Among current `ambiguous_rescue_candidate` false admissions, the dominant source
is consistently `weak_positive_ambiguous`.

| policy | weak positive | warning dominated | conflicting | relation borderline | low confidence |
| --- | ---: | ---: | ---: | ---: | ---: |
| rev24 safe-only | 0.4954 | 0.2337 | 0.0976 | 0.0293 | 0.1439 |
| rev25 class-aware | 0.4963 | 0.1990 | 0.1272 | 0.0141 | 0.1634 |
| rev26 calibrated | 0.5080 | 0.2182 | 0.1030 | 0.0225 | 0.1484 |

This explains why another generic ambiguity head did not solve rev27: the
largest problematic subtype is not simply "negative-looking ambiguity." It is a
positive-looking ambiguous subtype that still contains many false admissions.

## Counterfactual Sanity

The counterfactuals use rev26 as the baseline and are diagnostic only.

| counterfactual | selected additions | noisy precision | noisy recall | noisy F1 | removed/delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| rev26 baseline | 2,975 | 0.5903 | 0.7365 | 0.6553 | n/a |
| suppress weak_positive_ambiguous | 1,513 | 0.6082 | 0.7200 | 0.6594 | removed 1,462 |
| boost weak_positive_ambiguous | 2,975 | 0.5895 | 0.7355 | 0.6544 | +645 in subtype |
| suppress warning_dominated_ambiguous | 2,398 | 0.5985 | 0.7316 | 0.6584 | removed 577 |
| boost warning_dominated_ambiguous | 2,975 | 0.5872 | 0.7326 | 0.6518 | +1,044 in subtype |
| suppress low_confidence_ambiguous | 2,588 | 0.5959 | 0.7334 | 0.6576 | removed 387 |

Suppressing weak-positive ambiguity improves global noisy F1 relative to rev26,
but mostly by removing many admissions. It still does not beat rev6, and it
throws away many true rescue opportunities. Boosting the same subtype hurts,
confirming that it is not safe to treat it as uniformly rescuable.

## Diagnosis

The current ambiguity bucket is too coarse. It collapses at least two important
things:

- `warning_dominated_ambiguous`: mostly unsafe and relatively easy to suppress.
- `weak_positive_ambiguous`: positive-looking and relatively high-recall, but
  also the dominant source of ambiguous false admissions.

The strongest next boundary is not "ambiguous vs non-ambiguous." rev27 already
showed that is too blunt. The strongest boundary is inside
`weak_positive_ambiguous`: distinguish real safe rescue from positive-looking
false admission.

## Recommendation

Run one narrow rev29 offline `weak_positive_ambiguity_safety_probe`: within
`weak_positive_ambiguous` only, train/evaluate a tiny diagnostic classifier
using existing signals to separate GT-positive rescue candidates from
positive-looking false admissions. Do not integrate it or run backend until that
offline probe shows clear separability.
