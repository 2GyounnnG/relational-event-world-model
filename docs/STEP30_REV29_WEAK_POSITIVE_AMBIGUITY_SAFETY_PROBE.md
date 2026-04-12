# Step30 rev29: Weak-Positive Ambiguity Safety Probe

## Summary

rev29 is a narrow offline probe focused only on the
`weak_positive_ambiguous` subtype identified by rev28. It does not add a new
weak observation cue, train an integrated recovery model, run backend
integration, or continue the broad ambiguity-head/admission micro-line.

The core question was whether existing signals can separate safe rescue
candidates from positive-looking false admissions inside this hard subtype.

## Probe Setup

Scope:

```text
rescue candidate
and ambiguous-signal
and subtype == weak_positive_ambiguous
```

Probe:

- standardized logistic regression
- trained on train split, selected by validation AP
- features are existing signals only:
  relation/support hints, signed witness, pair-evidence bundle channels, rev6
  score/logit, rev24 class probabilities, and rev26 binary/ambiguity scores

Data:

| split | count | GT-positive rate |
| --- | ---: | ---: |
| train | 15,906 | 0.2892 |
| val | 3,104 | 0.2835 |
| test | 3,154 | 0.2920 |

Best validation:

```text
epoch = 239
val AP = 0.5662
val AUROC = 0.7529
```

## Separability

| scorer | AP | AUROC | precision@1462 | budget recall | budget F1 | pos mean | neg mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rev6 score | 0.4805 | 0.6717 | 0.4015 | 0.6374 | 0.4927 | 0.5281 | 0.4404 |
| rev24 safe logit | 0.5129 | 0.6946 | 0.4193 | 0.6656 | 0.5145 | 1.2534 | 0.9050 |
| rev25 class margin | 0.5397 | 0.7145 | 0.4282 | 0.6797 | 0.5254 | 0.3687 | 0.1004 |
| rev26 binary-minus-ambiguous | 0.5467 | 0.7170 | 0.4302 | 0.6830 | 0.5279 | 0.6217 | 0.4530 |
| rev29 logistic probe | 0.5637 | 0.7285 | 0.4337 | 0.6884 | 0.5321 | 0.5836 | 0.4222 |

The probe is sharper than the existing single scores, but only modestly. The
same-budget precision gain over rev26 is `0.0034`.

## Score Distributions

| scorer | GT+ q25/q50/q75 | GT- q25/q50/q75 |
| --- | --- | --- |
| rev26 binary-minus-ambiguous | 0.4631 / 0.6456 / 0.8027 | 0.3083 / 0.4442 / 0.5905 |
| rev29 logistic probe | 0.4325 / 0.6043 / 0.7488 | 0.2819 / 0.4031 / 0.5485 |

There is visible separation, but the distributions still overlap heavily.

## Feature Ranking

Top one-feature rankings:

| feature | direction | AP | AUROC | pos mean | neg mean |
| --- | --- | ---: | ---: | ---: | ---: |
| rev26 binary-minus-ambiguous | positive | 0.5467 | 0.7170 | 0.6217 | 0.4530 |
| rev24 safe prob | positive | 0.5412 | 0.7149 | 0.6686 | 0.5306 |
| rev24 safe-minus-reject | positive | 0.5397 | 0.7145 | 0.3687 | 0.1004 |
| rev26 ambiguous prob | negative | 0.5371 | 0.7099 | 0.3273 | 0.4761 |
| rev24 ambiguous prob | negative | 0.5319 | 0.7102 | 0.2971 | 0.4277 |
| rev6 score | positive | 0.4805 | 0.6717 | 0.5281 | 0.4404 |
| relation hint | positive | 0.4649 | 0.6699 | 0.4129 | 0.3681 |
| pair support hint | positive | 0.4373 | 0.6203 | 0.6265 | 0.6002 |
| signed pair witness | positive | 0.4064 | 0.6268 | 0.1561 | -0.0306 |

The most useful signals are still model-derived latent/class scores. Raw bundle
channels alone are weak inside this subtype.

## Counterfactual Admission

These are diagnostic-only offline substitutions over rev26. Only
`weak_positive_ambiguous` admissions are changed.

| policy | total selected | weak selected | weak P | weak R | noisy P | noisy R | noisy F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rev26 baseline | 2,975 | 1,462 | 0.2579 | 0.4093 | 0.5903 | 0.7365 | 0.6553 |
| probe replace same count | 2,975 | 1,462 | 0.2647 | 0.4202 | 0.5907 | 0.7369 | 0.6557 |
| probe replace half count | 2,244 | 731 | 0.3283 | 0.2606 | 0.6009 | 0.7305 | 0.6594 |
| suppress all weak-positive | 1,513 | 0 | 0.0000 | 0.0000 | 0.6082 | 0.7200 | 0.6594 |

Same-count replacement barely improves global recovery. Conservative admission
helps noisy F1, but mostly by reducing admissions, and still does not reach rev6
noisy F1 (`0.6657`).

## Diagnosis

`weak_positive_ambiguous` is not hopelessly entangled, but the existing signals
do not expose a strong enough safety boundary. The probe improves AP/AUROC and
same-budget precision slightly, yet the practical recovery effect is tiny.

The most promising existing signals are:

- rev26 binary-minus-ambiguous score
- rev24 safe probability / class margin
- negative ambiguity probability
- rev6 base score
- relation hint
- signed pair witness

Raw bundle channels are not enough inside this subtype, despite being useful for
constructing the subtype.

## Recommendation

Do not integrate rev29. The next move should be to design one stronger
observation-side safety signal for positive-looking ambiguous rescue candidates,
because existing signals provide only weak separability inside
`weak_positive_ambiguous`.
