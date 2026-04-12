# Step30 Signed-Pair-Witness Consolidation

## Purpose

This note consolidates the Step30 `signed_pair_witness` sub-line:

- rev13 signed-pair-witness weak observation cue
- rev14 bounded witness correction path
- rev15 signed witness correction supervision
- rev16 false-admission suppression objective

The goal is to make the stop/park decision explicit and design the next genuinely stronger observation/representation-side mechanism. This is not a backend adapter plan, not backend joint training, and not another tiny correction-loss tweak.

## Consolidation Summary

rev13 established that `signed_pair_witness` is a real new observation/representation signal. It is weak, noisy, non-leaky, and complementary to `relation_hint` and `pair_support_hint`. It materially improved the named bottleneck:

- hint-missed true-edge recall improved over rev6
- hint-missed true-edge average score improved
- hint-missed true edges ranked above hard negatives more often
- hint-supported false-positive error was initially cleaner than rev6

However, rev13 did not produce a better global recovery operating point:

- noisy edge F1 fell below rev6
- noisy precision dropped
- targeted rescue gains did not convert into a better full adjacency decode

rev14 moved `signed_pair_witness` out of the main edge-head input and into a bounded additive correction path. This was a better consumption pattern than direct fusion because it recovered some global F1 relative to rev13, but it became recall-heavy and raised unsafe false admissions.

rev15 added signed correction supervision. It nearly tied rev6 on noisy edge F1, but it did not beat rev6 and still had a worse safety profile:

- recall stayed above rev6
- precision stayed below rev6
- hint-supported false-positive error stayed worse than rev6

rev16 directly penalized positive correction on unsafe rescue-like GT-negative candidates. This slightly improved the false-admission diagnostic relative to rev15, but the gain was not large enough:

- noisy edge F1 dropped below rev15
- noisy precision remained far below rev6
- the line still failed to become a cleaner operating point

Chinese summary: `signed_pair_witness` 打开了真实的新信号，但 rev14-rev16 这些“如何消费这个信号”的小修小补没有把它变成更好的全局恢复点。它能救一部分 low-hint true edges，但同时仍然太容易放进 unsafe false admissions。

## Compact Comparison

Noisy split unless otherwise noted.

| Row | Overall Edge F1 | Noisy Precision | Noisy Recall | Noisy Edge F1 | Hint-Missed Recall | Hint-Supported FP Error |
|---|---:|---:|---:|---:|---:|---:|
| rev6 reference | 0.7383 | 0.6300 | 0.7072 | 0.6550 | 0.2817 | 0.4688 |
| rev13 direct witness | 0.7349 | 0.6026 | 0.7257 | 0.6481 | 0.3925 | 0.4529 |
| rev14 bounded correction | 0.7364 | 0.5959 | 0.7474 | 0.6528 | 0.3644 | 0.5018 |
| rev15 signed supervision | 0.7377 | 0.5972 | 0.7499 | 0.6546 | 0.3553 | 0.5122 |
| rev16 FP suppression | 0.7365 | 0.5970 | 0.7471 | 0.6533 | 0.3602 | 0.5028 |

## Park / Retain Decision

Retain `rev13 signed_pair_witness` as the current diagnostic reference for the mechanism family.

Reason:

- It most cleanly demonstrates that the new cue exposes real rescue-safety information.
- It gives the strongest targeted signal among the signed-witness runs: higher hint-missed recall and lower hint-supported false-positive error than rev6.
- It remains non-leaky because trivial recovery stays far below encoder recovery globally.

Do not retain rev13 as a usable operating point.

Reason:

- It does not beat rev6 on noisy edge F1.
- It is not strong enough to justify backend integration.

Do not retain rev14, rev15, or rev16 as usable operating points.

Reason:

- rev15 is the closest global-F1 endpoint, but its false-admission safety profile is wrong.
- rev16 is slightly cleaner than rev15 on the targeted false-admission axis, but it loses global F1 and still remains below rev6.

Step30 is not ready for backend rerun or adapter/interface work from this sub-line.

The rev13-rev16 signed-witness micro-line should now be parked for the current phase unless a genuinely new observation/representation mechanism is introduced.

## Why Small Tweaks Did Not Close The Gap

The recurring failure pattern is stable:

- The cue helps raise scores for hint-missed true edges.
- The cue does not expose enough information to reliably separate safe rescues from unsafe false admissions.
- Bounded correction, signed supervision, and false-admission suppression all operate on the same limited cue and pair context.
- Making the correction path more recall-friendly hurts precision.
- Making it safer does not recover enough global F1.

This suggests the bottleneck is not correction-head optimization. The missing piece is richer evidence about why a low-hint pair should be admitted safely.

## Next-Mechanism Design Memo

### Missing Information

`signed_pair_witness` gives a scalar leaning signal. It does not explain the evidence behind that leaning.

The current encoder still cannot reliably tell:

- low-hint true edges with independent corroborating evidence
- low-hint false admissions that merely look plausible under pair support
- negative witness that reflects true absence versus noisy dropout
- positive witness that reflects real connectivity versus spurious local co-activation

In short: the current cue says "lean positive or negative," but not "why this rescue is safe."

### Proposed Next Mechanism Family

Move from scalar witness to a small multi-source weak pair evidence packet.

Candidate name:

`pair_evidence_bundle`

This would remain synthetic, structured, weak, noisy, and pair-level. It would not expose clean adjacency. It should provide several complementary weak reasons for or against admitting a pair, for example:

- weak co-activation evidence
- weak anti-coactivation or exclusion evidence
- endpoint-compatibility evidence
- relation-process witness evidence
- noisy disagreement flags between evidence sources

The important change is not adding "one more scalar." The change is exposing a small structured evidence pattern so the encoder can learn rescue safety as a first-class representation problem.

### Why This Is Different From rev13-rev16

rev13-rev16 all ask the model to squeeze safe/unsafe rescue information out of one scalar witness plus existing relation/support hints.

The proposed family gives the model multiple weak evidence channels with different failure modes. That creates a learnable pattern:

- safe low-hint true edges should have partially consistent positive evidence across independent weak sources
- unsafe false admissions should show conflict, weak support, or negative evidence in at least one source

This is meaningfully different from:

- bounded additive correction
- signed correction supervision
- false-admission suppression over the same cue
- another decode/scorer tweak

### Smallest First Experiment

Run exactly one recovery-first experiment:

`Step30 rev17 pair_evidence_bundle_probe`

Scope:

- Add one structured weak observation field: `pair_evidence_bundle`
- Keep existing `relation_hint`, `pair_support_hint`, and `signed_pair_witness` unchanged for comparison
- Use 3 to 4 weak packet dimensions, not a large feature family
- Train a minimal encoder consumption path that projects the bundle into the pair edge representation
- Evaluate recovery only before any backend rerun

Required first-pass gates:

- noisy edge F1 must beat rev6
- hint-missed recall must stay above rev6
- hint-supported false-positive error must not exceed rev6 materially
- trivial decode must remain clearly below encoder

If this first experiment fails those gates, the Step30 observation-side rescue-safety direction should be paused rather than followed by another correction micro-line.

## Recommended Next Action

Run exactly one small first experiment in the new stronger mechanism family:

`Step30 rev17 pair_evidence_bundle_probe`

Do not continue rev13-rev16 correction/supervision tweaks. Do not move to backend adapter/interface work until the new observation/representation mechanism beats rev6 at recovery level.
