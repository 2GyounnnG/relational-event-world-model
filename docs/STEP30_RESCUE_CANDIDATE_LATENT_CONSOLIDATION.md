# Step30 Rescue-Candidate Latent Consolidation

## Executive Summary

The rev23-rev27 rescue-candidate latent line is real and informative, but it
should now be parked for the current phase.

It established that first-class rescue-candidate representations are more
separable than scalar residual shaping. The rev23 probe showed meaningful
rescue-scope signal, rev24 proved much of that signal survives integration, and
rev26 sharply improved the safe-vs-low-hint-false boundary.

The line did not produce a retained recovery operating point. Every integrated
variant still paid for missed-edge rescue with too many false admissions,
especially ambiguous rescue candidates. rev27 tested the obvious next small
move, an ambiguity-aware head, and it did not improve the operating point.

## Revision Consolidation

| rev | change | useful result | unresolved failure |
| --- | --- | --- | --- |
| rev23 | Offline first-class rescue-candidate latent probe over existing rescue-scope features | Stronger separation than scalar residuals; recovery-side simulation looked promising | Probe only; not an integrated recovery model |
| rev24 | Integrated tiny 3-way rescue-candidate latent head, ordinary path frozen | Latent separability mostly survived integration; hint-missed recall jumped | Safe-score-only admission admitted too many unsafe/ambiguous candidates |
| rev25 | Class-aware admission using safe minus strongest reject class | Better than safe-only admission; reduced ambiguous selected admissions | Low-hint false admissions rose too much; noisy F1 still below rev6 |
| rev26 | Binary safe-vs-low-hint-false calibration head | Strong safe-vs-low-hint-false AP/AUROC; low-hint false selected admissions fell | Ambiguous rescue candidates became dominant remaining risk |
| rev27 | Ambiguity-risk head over integrated latent | Slightly reduced ambiguous selected admissions | New head was weaker than rev26 softmax ambiguity; low-hint false admissions rose; noisy F1 did not improve |

## Compact Comparison

| row | candidate AP/AUROC | overall F1 | noisy P | noisy R | noisy F1 | rescue P | rescue R | hint-missed R | low-hint false admit | ambiguous admit |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rev6 reference | n/a | 0.7452 | 0.6306 | 0.7049 | 0.6657 | 0.3299 | 0.5062 | 0.2817 | 0.4752 | 0.2121 |
| rev23 probe | 0.4293 / 0.7214 | sim only | 0.6361 | 0.7153 | 0.6734 | 0.3686 | 0.5837 | n/a | n/a | n/a |
| rev24 safe-only | 0.4284 / 0.7108 | 0.7368 | 0.5893 | 0.7353 | 0.6543 | 0.2928 | 0.7329 | 0.4039 | 0.0391 | 0.2340 |
| rev25 class-aware | 0.7184 / 0.6858 | 0.7379 | 0.5912 | 0.7376 | 0.6563 | 0.2998 | 0.7505 | 0.4134 | 0.1271 | 0.2048 |
| rev26 calibrated | 0.9058 / 0.8640 | 0.7374 | 0.5903 | 0.7365 | 0.6553 | 0.2964 | 0.7420 | 0.4089 | 0.0464 | 0.2291 |
| rev27 ambiguity-aware | 0.8765 / 0.8322 | 0.7373 | 0.5902 | 0.7364 | 0.6552 | 0.2960 | 0.7410 | 0.4083 | 0.0605 | 0.2256 |

Notes:

- rev23 is a probe/simulation row, not a retained integrated recovery model.
- rev24 candidate AP/AUROC is safe-vs-all-unsafe over the integrated safe score.
- rev25 and rev26 candidate AP/AUROC are safe-vs-low-hint-false comparisons.
- rev27 candidate AP/AUROC is the new ambiguity-risk head on ambiguity detection.
- rev24-rev27 low-hint false and ambiguous admission columns are selected rescue additions only, because that is where admission policy is being tested.
- rev6 rescue precision/recall and admission rates are full rescue-scope decode behavior, since it has no separate selected-addition rescue policy.

## Park / Retain Decision

Retain rev23 as the diagnostic proof that first-class rescue-candidate latent
representations expose more structure than scalar residuals.

Retain rev26 as the strongest targeted latent-calibration reference for the
safe-vs-low-hint-false boundary.

Do not retain rev24-rev27 as usable recovery operating points. None clearly
beats the rev6 recovery reference, and all integrated variants keep the same
core problem: rescue recall improves, but global precision and noisy F1 do not
recover enough.

Step30 is not ready for backend rerun, adapter/interface work, or backend joint
training from this line. The rev23-rev27 micro-line should be parked unless the
next step changes the observation/supervision substrate itself.

## Next-Mechanism Design Memo

### Missing Information

The current rescue-candidate latent sees relation hints, pair support,
pair-evidence bundle signals, base scores, and pair representations, but it
does not expose why an ambiguous candidate is ambiguous in a way that is useful
for admission.

The current `ambiguous_rescue_candidate` label is also too broad. It groups
together several different cases:

- genuinely under-supported true-like candidates that should maybe be admitted
- weak/contradictory candidates that should be rejected
- mid-hint candidates where relation and support disagree
- candidates where bundle channels are internally conflicting

That class is useful diagnostically, but too coarse as a training/admission
target. A small head can learn that the class exists, but not which ambiguous
cases are safe enough to rescue.

### Why rev23-rev27 Did Not Close the Gap

rev23 proved separability exists in a richer representation. rev24 proved that
the signal survives integration. rev25-rev27 then exhausted the obvious
decode/head fixes:

- safe-only admission over-admitted ambiguous candidates.
- class-aware admission reduced ambiguity but increased low-hint false
  admissions.
- safe-vs-false calibration fixed one boundary while leaving ambiguity as the
  dominant risk.
- ambiguity-risk supervision did not outperform the existing 3-way ambiguity
  softmax and did not improve recovery.

The bottleneck is therefore not another classifier head. It is the ambiguity
definition and observation/supervision signal available for ambiguous rescue
candidate safety.

### Stronger Mechanism Family

The next mechanism should be an observation/representation-side
`rescue_ambiguity_subtype` family.

Instead of treating ambiguity as one reject bucket, split ambiguous rescue
candidates into a small number of structured, synthetic, weakly supervised
subtypes that reflect why the candidate is ambiguous:

- `conflicting_evidence_ambiguous`: pair support and bundle channels disagree.
- `weak_positive_ambiguous`: weak positive support exists, but relation hint is
  low and evidence is incomplete.
- `warning_dominated_ambiguous`: false-admission warning cues dominate positive
  support.

This is meaningfully different from another latent head because it changes the
supervision substrate. The model would no longer be asked to learn one broad
ambiguous class and then guess which part of it is safe. It would learn a
structured ambiguity representation that can separate "ambiguous but rescuable"
from "ambiguous and unsafe."

### Smallest First Experiment

Run one diagnostic-first rev28 experiment:

1. Do not add a new external cue.
2. Re-label existing rescue-scope ambiguous candidates into 3 synthetic
   subtypes using existing pair-evidence bundle relationships:
   `positive > warning`, `warning > positive`, and `near-tie/conflict`.
3. Train a tiny offline probe, not an integrated model, to predict:
   `safe_missed_true_edge`, `low_hint_pair_support_false_admission`,
   `weak_positive_ambiguous`, `warning_dominated_ambiguous`,
   `conflicting_ambiguous`.
4. Evaluate whether subtype-aware ranking improves selected rescue precision at
   the same budget without losing safe rescue recall.
5. Only if the offline probe shows a clear gain should an integrated recovery
   model be attempted.

This keeps the next step observation/representation-side, avoids backend work,
and tests whether the real blocker is the coarse ambiguity class.

## Recommendation

Park the rev23-rev27 integrated rescue-candidate latent micro-line and run
exactly one small rev28 offline `rescue_ambiguity_subtype_probe` that refines
the ambiguous candidate supervision using existing bundle relationships before
any further integration or backend rerun.
