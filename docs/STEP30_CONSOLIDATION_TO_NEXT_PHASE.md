# Step30 Consolidation to Next Phase

## Executive Summary

Step30 should stop at the current consolidated point.

The phase produced real observation/representation-side progress, but it did not produce a retained recovery operating point that clearly beats the rev6 recovery reference. The repeated pattern is now stable: stronger rescue recall and better targeted rescue diagnostics are possible, but the added admissions still cost too much noisy precision.

The next phase should not be another Step30 micro-tweak. It should move one level up to a stronger synthetic observation/representation substrate while staying fully structured, synthetic, and evaluation-first.

## Step30 Project-Level Consolidation

### Retained Recovery Reference

`rev6 missed-edge evidence recovery with retained rev6 decode settings` remains the current best Step30 recovery reference.

It is not the most aggressive missed-edge rescue point, but it remains the cleanest retained recovery operating point after the full Step30 line because later variants do not clearly improve noisy edge F1 without paying too much precision cost.

### Diagnostic References to Preserve

| reference | role |
| --- | --- |
| rev6 | current best retained Step30 recovery reference |
| rev13 `signed_pair_witness` | diagnostic proof that new weak pair evidence can improve targeted rescue bottlenecks |
| `pair_evidence_bundle` / rev17 | stronger targeted observation family than scalar witness, but not retained due to precision damage |
| rev23 | proof that first-class rescue-candidate latents are more separable than scalar residual shaping |
| rev26 | best targeted latent-calibration reference for safe-vs-low-hint-false separation |
| rev32 | current positive-ambiguity-safety recovery-side diagnostic reference |

### What Worked

Step30 established that the rescue problem is real and measurable. Selective rescue, missed-edge evidence, signed witness, pair evidence bundles, rescue-candidate latents, and positive-ambiguity safety all produced targeted signal improvements.

Most importantly, Step30 showed that observation/representation changes are more promising than pure decode tweaks. The strongest gains came when new structured weak evidence or first-class rescue-candidate representations were introduced, not when thresholds or scalar residuals were retuned.

### What Failed to Become Retained

None of the post-rev6 micro-lines produced a recovery operating point that clearly beat rev6.

The repeated failure mode is precision. Later variants often improve hint-missed recall, weak-positive subtype ordering, or rescue-candidate separability, but the admitted rescues still include too many false positives. The gain transfers into recall more reliably than into a clean global noisy edge F1 improvement.

### Parked Step30 Lines

The following lines should be parked for the current phase:

- rev7-rev12 rescue-decode micro-line
- rev13-rev16 signed-witness correction/supervision micro-line
- rev19-rev22 rescue-residual shaping micro-line
- rev23-rev27 integrated rescue-candidate latent micro-line
- rev30-rev32 positive-ambiguity-safety micro-line

They should only be reopened if a genuinely new observation or representation mechanism changes the available signal, not for another local admission, threshold, residual, or head tweak.

## What Is Not Justified Next

Adapter/interface work is not justified next. Step30 did not produce a clearly stronger recovery operating point to hand off to an interface phase.

Backend joint training is not justified next. The current evidence says the bottleneck is observation/representation safety signal, not downstream backend coupling.

Step30c backend reruns are not justified from the parked lines. The recovery gates were not clearly met.

Reopening parked Step30 micro-lines is not justified. They have already mapped the current local frontier and repeatedly return the same recall-vs-precision tradeoff.

Real-world data, raw real-world images, hypergraph expansion, and LLM integration are not justified next. They would add uncontrolled complexity before the synthetic observation bridge is strong enough.

Another single cue, tiny head, admission rule, residual loss, or threshold tweak is not justified as the main next phase. The last several sub-lines show those moves are now below the useful mechanism scale.

## Next-Phase Ordering Memo

### 1. Step31: Synthetic Multi-View Observation Bridge

This should be the immediate next phase.

The next mechanism should introduce a controlled synthetic multi-view observation setting: multiple weak, noisy, structured observations of the same graph-event state or transition, with disagreement/corroboration available to the encoder as an observation-side representation problem.

This is stronger than Step30 because it changes the evidence substrate, not just how one pair-level cue is consumed. It gives the model a plausible way to distinguish true weak positive ambiguity from false positive-looking ambiguity through cross-view consistency, without exposing clean adjacency.

### 2. Later: Synthetic Rendered / Image-Like Observation Bridge

If Step31 succeeds, the next natural direction is a synthetic rendered or image-like observation bridge.

This should still be fully synthetic and controlled. The goal would be to move from structured weak observations toward raw-like perception while preserving known graph-event ground truth and diagnostic decompositions.

### 3. Later: Adapter / Interface Phase

Adapter or interface work should wait until the observation/recovery side has a stronger retained operating point than rev6, or until a new observation bridge produces clear downstream-aligned recovery gains.

### 4. Much Later: Real-World / Hypergraph / LLM Expansions

These remain out of scope for the current repository phase. They should only be considered after synthetic observation and representation bridges are stable enough to make failures interpretable.

## Recommended Next Entry Point

Run exactly one new phase entry:

`Step31 synthetic_multi_view_observation_bridge_probe`

Narrow definition:

- generate two or three independent noisy structured observation views for the same synthetic graph-event example,
- keep all data synthetic and graph-structured,
- expose only weak view-level relation/support/evidence observations, not clean adjacency,
- add small cross-view consistency/corroboration features or an encoder path that can compare views,
- evaluate recovery-first against rev6, rev17, rev26, and rev32,
- include trivial multi-view baselines to verify the bridge is non-leaky,
- do not run backend integration unless the recovery gate clearly beats rev6.

This is the smallest next move that is meaningfully larger than the parked Step30 micro-tweaks while staying inside the repository's current scope.
