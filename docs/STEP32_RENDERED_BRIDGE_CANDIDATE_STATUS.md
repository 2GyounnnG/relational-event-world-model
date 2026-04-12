# Step32 Rendered Bridge Candidate Status

## Scope

This note records the current public status of the bounded **Step32 synthetic rendered / image-like bridge** line.

It is not a formal promotion note.
It does not reopen parked Step22–31 micro-lines.
It does not introduce a new mechanism.
Step32 source is already in the public repo.
It records the current best isolated Step32 candidate references and the fixed evaluation protocol now used for this line.

---

## Why Step32 exists

Step32 is the next-phase bridge question after the retained multi-view Step31 result.

The phase question is:

> Can the observation substrate be pushed from structured weak multi-view signals toward a more rendered / image-like synthetic observation family, while keeping evaluation recovery-first, interpretable, and comparable to the existing backend?

This line remains:

- synthetic
- fully controlled
- fully labeled
- evaluation-first
- recovery-first
- isolated from backend joint training

---

## Fixed comparison protocol

For the current Step32 candidate line, threshold fiddling is considered closed.
Future comparisons for this line should use exactly two evaluation settings only:

1. **Default threshold**
   - `clean=0.50`
   - `noisy=0.50`

2. **Validated calibrated threshold**
   - `clean=0.30`
   - `noisy=0.30`

The `0.30 / 0.30` point was selected as the simplest stable calibrated operating point after:

- symmetric threshold sweeps
- validation-only threshold selection
- an asymmetric clean/noisy check

The asymmetric check did **not** find a better validation-overall operating point than the symmetric `0.30 / 0.30` pair.

---

## Candidate progression

### smoke256

Main result:

- infrastructure worked end-to-end
- default threshold `0.50` collapsed to zero predicted positive edges
- calibrated evaluation near `0.30` exposed nonzero recovery signal
- this established that the immediate blocker was calibration / under-confident edge logits, not broken training infrastructure

### smoke1024

Main result:

- Step32 first beat the trivial rendered baselines under calibrated evaluation
- default-threshold behavior remained weak
- this established Step32 as more than pure smoke, but still clearly calibration-dependent

Key calibrated result at `0.30 / 0.30`:

- overall F1: `0.4514`
- noisy F1: `0.4295`

### candidate01

Main result:

- a larger bounded run improved both calibrated behavior and default-threshold non-collapse
- it passed the gate needed to justify a larger isolated candidate run

Key calibrated result at `0.30 / 0.30`:

- overall F1: `0.4803`
- noisy F1: `0.4501`

### candidate_main

Main result:

- candidate_main became the first strong isolated Step32 candidate
- it clearly beat both trivial rendered baselines at the validated calibrated threshold
- it remained calibration-dependent and did not justify backend transfer

Key metrics:

Default `0.50 / 0.50`:

- overall F1: `0.3748`
- clean F1: `0.4047`
- noisy F1: `0.3457`

Validated calibrated `0.30 / 0.30`:

- overall F1: `0.5265`
- clean F1: `0.5727`
- noisy F1: `0.4856`

### candidate_next

Main result:

- simple scale-up helped clearly under the same fixed protocol
- candidate_next became the best calibrated isolated Step32 candidate
- default-threshold behavior improved materially
- calibrated behavior improved materially
- backend transfer still remained closed because Step30 rev6 is still ahead

Candidate_next metrics:

Default `0.50 / 0.50`:

- overall F1: `0.4688`
- clean F1: `0.5315`
- noisy F1: `0.4036`

Validated calibrated `0.30 / 0.30`:

- overall F1: `0.5859`
- clean F1: `0.6475`
- noisy F1: `0.5268`

Direct delta vs candidate_main:

Default threshold:

- overall F1: `+0.0940`
- clean F1: `+0.1268`
- noisy F1: `+0.0579`

Validated calibrated threshold:

- overall F1: `+0.0594`
- clean F1: `+0.0749`
- noisy F1: `+0.0412`

### candidate_scale

Main result:

- pure scale-up clearly helped the default-threshold operating point
- candidate_scale became the best default-threshold isolated Step32 reference
- calibrated recovery regressed slightly versus candidate_next, so pure scale-up no longer clearly helps the calibrated operating point
- Step32 is no longer only calibration-dependent in the earlier sense, but remains threshold-sensitive
- backend transfer still remained closed because Step30 rev6 is still ahead

Candidate_scale metrics:

Default `0.50 / 0.50`:

- overall F1: `0.5860`
- clean F1: `0.6515`
- noisy F1: `0.5193`

Validated calibrated `0.30 / 0.30`:

- overall F1: `0.5685`
- clean F1: `0.6216`
- noisy F1: `0.5191`

Direct delta vs candidate_next:

Default threshold:

- overall F1: `+0.1172`
- clean F1: `+0.1200`
- noisy F1: `+0.1157`

Validated calibrated threshold:

- overall F1: `-0.0174`
- clean F1: `-0.0259`
- noisy F1: `-0.0077`

---

## Current retained isolated references

The current retained isolated Step32 references are:

- best calibrated isolated reference: `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30`
- best default-threshold isolated reference: `step32_rendered_bridge_candidate_scale @ clean=0.50, noisy=0.50`

These are **candidate references**, not formal promoted Step32 checkpoints.

The formal Step32 checkpoint path remains untouched:

- `checkpoints/step32_rendered_bridge/`
- `checkpoints/step32_rendered_bridge/best.pt`

---

## Why it is not formally promoted

### 1. It is still threshold-sensitive

Candidate_scale changed the interpretation: Step32 has progressed beyond the earlier phase where only the calibrated threshold exposed meaningful recovery.

Candidate_scale:

- default overall F1: `0.5860`
- calibrated overall F1: `0.5685`

However, the operating point still matters materially. At `0.30 / 0.30`, candidate_scale trades precision for recall and does not improve the best calibrated reference. The best calibrated reference remains candidate_next, while the best default-threshold reference is now candidate_scale.

### 2. Step30 rev6 is still ahead

Retained Step30 rev6 reference:

- overall F1: `0.7452`
- noisy F1: `0.6657`

Best Step32 calibrated isolated reference, candidate_next at validated calibrated threshold:

- overall F1: `0.5859`
- noisy F1: `0.5268`

Best Step32 default-threshold isolated reference, candidate_scale at default threshold:

- overall F1: `0.5860`
- noisy F1: `0.5193`

So Step32 has not yet reached the existing weak-observation recovery reference.

### 3. Backend transfer remains closed

No backend transfer was run for candidate_scale.

For candidate_next and candidate_scale:

- `beats_step30_rev6_overall_f1: false`
- `beats_step30_rev6_noisy_f1: false`
- `backend_transfer_rerun: false`

So the correct public decision remains:

- do not reopen backend transfer yet
- do not promote the formal Step32 path yet

---

## Current public interpretation

The right public interpretation is:

> Step32 is no longer just an infrastructure smoke line.
> It now has meaningful default-threshold behavior and a separate best calibrated isolated reference.
> Pure scale-up clearly improved the default-threshold operating point, but calibrated recovery appears to be starting to saturate under pure scale-up.
> Step32 remains threshold-sensitive, remains below Step30 rev6, and is therefore not formally promotable yet.

---

## Current recommended status

Retain publicly:

- `step31_simple_late_fusion` as the retained Step31 multi-view bridge reference
- `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30` as the best calibrated isolated Step32 reference
- `step32_rendered_bridge_candidate_scale @ clean=0.50, noisy=0.50` as the best default-threshold isolated Step32 reference

Do **not** currently do any of the following:

- formal Step32 checkpoint promotion
- backend transfer reopening
- threshold fiddling for the current candidate checkpoint
- reopening parked Step22–31 micro-lines

---

## Next smallest justified action

The next smallest justified action is **not** more threshold tuning for the same checkpoint.

For the current candidate checkpoints, threshold protocol is considered fixed.

Future work should either:

1. stop scale-only progression here and retain the split calibrated/default references, or
2. wait for a genuinely stronger Step32 mechanism before running another bounded Step32 progression under the same fixed protocol.
