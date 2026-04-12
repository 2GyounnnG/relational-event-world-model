# Step30 Positive-Ambiguity-Safety Consolidation

## Scope

This note consolidates the rev30-rev32 positive-ambiguity-safety sub-line. It does not add a new cue, does not train a new model, does not reopen Step22-Step29, and does not run Step30c backend integration.

## Line Summary

### Rev30: Observation-Side Safety Cue

Rev30 added `positive_ambiguity_safety_hint`, a weak/noisy/non-leaky observation-side cue targeted at the hard `weak_positive_ambiguous` rescue subtype.

It established that the benchmark can expose more useful safety information for positive-looking ambiguity than the earlier relation/support/witness/bundle signal set. The offline weak-positive probe improved from rev29 AP `0.5594` / AUROC `0.7243` to rev30 AP `0.5844` / AUROC `0.7501`, while trivial recovery remained far below the encoder.

However, integrated recovery did not clear the rev6 gate. Rev30 noisy F1 reached `0.6558`, below rev6 `0.6657`.

### Rev31: Subtype-Scoped Safety Head

Rev31 added a tiny binary safety head inside `weak_positive_ambiguous` only. It showed that the rev30 cue boundary can survive integration: weak-positive integrated AP rose from `0.5510` to `0.5799`, nearly matching the rev30 offline probe.

The failure shifted from representation to admission. Rev31 default under-selected weak-positive candidates, improving weak-positive precision but losing too many safe rescues. Noisy F1 fell to `0.6544`.

### Rev32: Subtype-Head Budget Calibration

Rev32 did not change representation. It partitioned the rescue budget into weak-positive and non-weak-positive buckets, using a validation-matched weak-positive budget fraction from rev30 retained admissions.

The retained rev32 variant ranked weak-positive candidates with the rev31 subtype head and kept non-weak-positive ranking on the rev30 integrated score. This restored weak-positive selection and slightly improved over rev30/rev31, but noisy F1 only reached `0.6560`, still below rev6.

## Compact Comparison

| row | overall F1 | noisy P | noisy R | noisy F1 | weak-pos selected | rescue P | rescue R | hint-missed recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rev6 | 0.7452 | 0.6306 | 0.7049 | 0.6657 | n/a | n/a | n/a | 0.2817 |
| rev30 integrated | 0.7376 | 0.5907 | 0.7370 | 0.6558 | 1402 | 0.2474 | 0.2397 | 0.4110 |
| rev31 default | 0.7368 | 0.5894 | 0.7354 | 0.6544 | 1036 | 0.2350 | 0.2277 | 0.4045 |
| rev32 retained calibration | 0.7377 | 0.5909 | 0.7372 | 0.6560 | 1430 | 0.2487 | 0.2410 | 0.4117 |

`rev6` has no selective weak-positive rescue admission bucket, so weak-positive and rescue-scope admission metrics are not applicable.

## Park / Retain Decision

Retain `rev32_val_matched_weak_budget_rev30_nonweak` as the current positive-ambiguity-safety recovery-side diagnostic reference.

Also retain rev31 as the representation proof point: it showed the subtype head can learn the weak-positive boundary. Rev32 is the better calibrated recovery-side reference, but it is not a usable retained operating point because it does not beat rev6.

Step30 is not ready for Step30c backend rerun from this line, and it is not ready for adapter/interface work on this basis. The rev30-rev32 sub-line should be parked for now.

## Next-Mechanism Decision

Park now.

Further tiny admission or budget tweaks are low value because the line has already separated the main factors:

- rev30 showed the cue is real and non-leaky,
- rev31 showed the integrated subtype head can learn the boundary,
- rev32 showed better budget use can recover some safe weak-positive admissions,
- but recovery remains pinned around noisy F1 `0.655`-`0.656`, roughly one point below rev6.

The remaining gap is not a missing threshold. It is that the admitted rescue additions still carry too much noisy precision cost. The current positive-ambiguity cue helps select better weak-positive candidates, but it does not make the full rescue admission set clean enough to become a retained recovery operating point.

Before reopening this line, the mechanism would need to change qualitatively. The most plausible reopening condition would be a richer supervision/observation substrate that links ambiguous pair evidence to event-local consistency or downstream edit consequences, rather than another pair-score calibration over the same subtype head.

## Recommendation

Park rev30-rev32 now.
