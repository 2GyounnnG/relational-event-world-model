# Step30 Rev30 Positive Ambiguity Safety Cue

## Scope

Rev30 adds exactly one observation-side cue, `positive_ambiguity_safety_hint`, targeted at the hard `weak_positive_ambiguous` rescue subtype. It does not reopen Step22-Step29, does not add backend training, and does not run Step30c.

## Cue Design

`positive_ambiguity_safety_hint` is a scalar pair-level weak observation in `[0, 1]`.

- High values weakly indicate that positive-looking ambiguous rescue evidence is internally coherent.
- Low values weakly warn that the same positive-looking evidence is suspicious.
- Outside low-relation, support-backed, positive-looking ambiguity, the cue is centered near `0.5`.
- It is corrupted with dropout, flips, Gaussian jitter, and quantization.
- It uses local coherence context plus noisy class-conditioned means, not a clean adjacency copy.

The cue is separate from `relation_hint`, `pair_support_hint`, `signed_pair_witness`, and the four-channel `pair_evidence_bundle`.

## Model Consumption

The ordinary rev6-style edge path does not consume this cue.

The cue is consumed only through a zero-initialized projection into the existing rescue-candidate latent path:

- start from rev26 checkpoint,
- freeze ordinary/global recovery parameters,
- train rescue-candidate latent/classifier/binary calibration plus the new projection,
- use the existing rescue-scope admission simulation.

## Key Results

### Weak-Positive Ambiguity Probe

| row | AP | AUROC | P@1462 | budget F1 |
| --- | ---: | ---: | ---: | ---: |
| rev29 current-signal probe | 0.5594 | 0.7243 | 0.4289 | 0.5262 |
| rev30 with safety cue probe | 0.5844 | 0.7501 | 0.4473 | 0.5489 |
| rev30 integrated score | 0.5510 | 0.7305 | 0.4316 | 0.5296 |
| safety cue only | 0.3776 | 0.6140 | 0.3707 | 0.4549 |
| rev26 score | 0.5467 | 0.7170 | 0.4302 | 0.5279 |

### Recovery Summary

| row | overall F1 | clean F1 | noisy P | noisy R | noisy F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rev6 | 0.7452 | 0.8282 | 0.6306 | 0.7049 | 0.6657 |
| rev17 global bundle | 0.7407 | 0.8264 | 0.5911 | 0.7450 | 0.6592 |
| rev26 calibrated | 0.7374 | 0.8282 | 0.5903 | 0.7365 | 0.6553 |
| rev30 integrated | 0.7376 | 0.8282 | 0.5907 | 0.7370 | 0.6558 |
| trivial with rev30 cue | 0.5168 | 0.5741 | 0.4489 | 0.4344 | 0.4416 |

### Rescue Scope

| row | precision | recall | F1 | weak-positive P | weak-positive R |
| --- | ---: | ---: | ---: | ---: | ---: |
| rev26 calibrated | 0.2434 | 0.2358 | 0.2395 | 0.2579 | 0.4093 |
| rev30 integrated | 0.2474 | 0.2397 | 0.2435 | 0.2760 | 0.4202 |
| rev30 safety probe same weak count | 0.2497 | 0.2420 | 0.2458 | 0.2709 | 0.4300 |

## Diagnosis

The new cue is a real observation-side safety signal: it improves weak-positive ambiguity separability, and the cue alone is weak enough that trivial recovery remains far below the encoder.

However, rev30 does not clear the recovery gate. Integrated noisy F1 reaches `0.6558`, below rev6 `0.6657` on this test set. The offline safety-cue probe is stronger than the integrated model, suggesting that the new signal is useful but not yet transferred cleanly into the retained recovery operating point.

No backend rerun was run.
