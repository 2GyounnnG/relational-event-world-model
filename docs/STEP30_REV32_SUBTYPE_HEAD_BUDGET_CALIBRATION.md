# Step30 Rev32 Subtype-Head Budget Calibration

## Scope

Rev32 is a calibration-only probe over the existing rev31 integrated subtype head. It does not add a new cue, retrain the recovery model, alter the ordinary rev6-style path, or run Step30c backend integration.

## Change

Rev31 learned a sharper `weak_positive_ambiguous` safety boundary, but its default shared rescue budget under-selected weak-positive candidates. Rev32 tests whether the learned head becomes useful when admission count is controlled.

The retained rev32 rule is:

- keep the total rescue budget equal to the existing top-20-percent rescue budget,
- split that budget into `weak_positive_ambiguous` and non-weak-positive buckets,
- set the weak-positive budget fraction from validation by matching rev30 retained weak-positive admissions,
- rank weak-positive candidates with the rev31 subtype safety head,
- rank non-weak-positive candidates with the rev30 integrated score.

This keeps the probe narrow: the representation and cues are unchanged, and only budget allocation changes.

## Calibration

| item | value |
| --- | ---: |
| val rescue budget | 2956 |
| val rev30 weak-positive selected | 1421 |
| val weak-positive budget fraction | 0.4807 |
| test rescue budget | 2975 |
| rev32 test weak-positive budget | 1430 |
| rev30 test weak-positive selected | 1402 |

## Admission Results

| row | admitted | weak-pos selected | rescue P | rescue R | rescue F1 | weak-pos P | weak-pos R | low-hint false rate | ambiguous rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rev30 integrated | 2975 | 1402 | 0.2474 | 0.2397 | 0.2435 | 0.2760 | 0.4202 | 0.0436 | 0.2285 |
| rev31 default | 2975 | 1036 | 0.2350 | 0.2277 | 0.2313 | 0.3108 | 0.3496 | 0.0355 | 0.2346 |
| rev31 same test weak count | 2975 | 1402 | 0.2413 | 0.2339 | 0.2376 | 0.2810 | 0.4278 | 0.0299 | 0.2341 |
| rev32 val-matched weak budget, rev31 nonweak | 2975 | 1430 | 0.2420 | 0.2345 | 0.2382 | 0.2790 | 0.4332 | 0.0299 | 0.2339 |
| rev32 val-matched weak budget, rev30 nonweak | 2975 | 1430 | 0.2487 | 0.2410 | 0.2448 | 0.2790 | 0.4332 | 0.0432 | 0.2282 |

The retained rev32 variant is `rev32_val_matched_weak_budget_rev30_nonweak`: it best preserves rev30's nonweak operating point while using the sharper rev31 subtype head inside weak-positive ambiguity.

## Recovery Results

| row | overall F1 | clean F1 | noisy P | noisy R | noisy F1 | hint-missed recall | hint-supported FP error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| rev6 | 0.7452 | 0.8282 | 0.6306 | 0.7049 | 0.6657 | 0.2817 | 0.4688 |
| rev26 calibrated | 0.7374 | 0.8282 | 0.5903 | 0.7365 | 0.6553 | 0.4089 | 0.4688 |
| rev30 integrated | 0.7376 | 0.8282 | 0.5907 | 0.7370 | 0.6558 | 0.4110 | 0.4688 |
| rev31 default | 0.7368 | 0.8282 | 0.5894 | 0.7354 | 0.6544 | 0.4045 | 0.4688 |
| rev32 val-matched weak budget, rev31 nonweak | 0.7373 | 0.8282 | 0.5902 | 0.7363 | 0.6552 | 0.4081 | 0.4688 |
| rev32 val-matched weak budget, rev30 nonweak | 0.7377 | 0.8282 | 0.5909 | 0.7372 | 0.6560 | 0.4117 | 0.4688 |
| trivial with rev30 cue | 0.5168 | 0.5741 | 0.4489 | 0.4344 | 0.4416 | n/a | n/a |

## Diagnosis

Rev32 uses the rev31 subtype head better than rev31 default: weak-positive selected count is restored, weak-positive recall improves, and noisy F1 edges above rev30/rev31.

However, the gain is small. Rev32 reaches noisy F1 `0.6560`, still below rev6 `0.6657`. The remaining issue is not weak-positive under-admission alone; even with better subtype-head budget control, the recovery point still pays too much noisy precision cost.

No Step30c backend rerun was run because the recovery gate was not met.
