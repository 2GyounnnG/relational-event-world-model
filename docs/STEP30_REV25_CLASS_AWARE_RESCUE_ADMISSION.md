# Step30 Rev25 Class-Aware Rescue Admission Probe

Rev25 tests whether the integrated rev24 3-way rescue-candidate latent can be
used more explicitly during admission.

## Rule

No model was retrained. Rev25 reuses the rev24 integrated checkpoint and ranks
rescue additions by:

```text
P(safe_missed_true_edge)
  - max(P(low_hint_pair_support_false_admission), P(ambiguous_rescue_candidate))
```

The rule is applied only inside the same rescue scope:

```text
relation_hint < 0.50
pair_support_hint >= 0.55
```

Validation selected the inherited `top_20pct` budget.

## Recovery Result

| row | noisy precision | noisy recall | noisy F1 |
| --- | ---: | ---: | ---: |
| rev6 | 0.6306 | 0.7049 | 0.6657 |
| rev21 | 0.6310 | 0.7095 | 0.6680 |
| rev24 safe-only | 0.5893 | 0.7353 | 0.6543 |
| rev25 class-aware | 0.5912 | 0.7376 | 0.6563 |

Rev25 improves over rev24 safe-only admission, but does not beat rev6.

## Rescue-Scope Admission

| row | precision | recall | F1 | safe admit | low-hint false admit | ambiguous admit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rev6 | 0.3299 | 0.5062 | 0.3995 | 0.5062 | 0.4752 | 0.2121 |
| rev21 | 0.3414 | 0.5407 | 0.4186 | 0.5407 | 0.5272 | 0.2032 |
| rev24 safe-only | 0.2928 | 0.7329 | 0.4184 | 0.7329 | 0.5143 | 0.4461 |
| rev25 class-aware | 0.2998 | 0.7505 | 0.4285 | 0.7505 | 0.6023 | 0.4169 |

For selected additions only, rev25 improves precision from `0.2339` to `0.2521`
and reduces ambiguous admissions from `0.2340` to `0.2048`, but it increases
low-hint false-admission picks from `0.0391` to `0.1271`.

## Gate Decision

Rev25 does not clear the recovery gate:

- It beats rev24 safe-only admission slightly.
- It does not beat rev6 noisy edge F1.
- It does not produce a clean enough precision profile to justify Step30c.

No backend integration was run.

## Diagnosis

The 3-way latent is useful, but the class-aware margin is still not enough.
It suppresses some ambiguous candidates, but the admission pressure shifts into
low-hint false admissions. The next smallest move should either park this
decode-only admission policy line or add a very narrow false-admission-aware
calibration objective for the latent classifier, rather than another top-k rule.
