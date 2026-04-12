# Step32 Rendered Bridge Candidate Status

## Scope

This note records the current public status of the bounded **Step32 synthetic rendered / image-like bridge** line.

It is not a formal promotion note.
It does not reopen parked Step22–31 micro-lines.
It does not introduce a new mechanism.
It records the current best isolated Step32 candidate and the fixed evaluation protocol now used for this line.

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
- candidate_next is the current best isolated Step32 candidate
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

---

## Current retained isolated candidate

The current retained isolated Step32 candidate reference is:

- checkpoint: `checkpoints/step32_rendered_bridge_candidate_next/best.pt`
- validated calibrated operating point: `clean=0.30`, `noisy=0.30`

This is a **candidate reference**, not a formal promoted Step32 checkpoint.

---

## Why it is not formally promoted

### 1. It is still calibration-dependent

Even though candidate_next is much healthier at the default threshold than earlier runs, calibrated evaluation remains clearly stronger than default evaluation.

Candidate_next:

- default overall F1: `0.4688`
- calibrated overall F1: `0.5859`

So the line still does not have a default-stable operating point.

### 2. Step30 rev6 is still ahead

Retained Step30 rev6 reference:

- overall F1: `0.7452`
- noisy F1: `0.6657`

Candidate_next at validated calibrated threshold:

- overall F1: `0.5859`
- noisy F1: `0.5268`

So Step32 has not yet reached the existing weak-observation recovery reference.

### 3. Backend transfer remains closed

For candidate_next:

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
> It now has a real isolated learned candidate that beats the trivial rendered baselines under a validation-selected calibrated operating point.
> But it remains calibration-dependent, remains below Step30 rev6, and is therefore not formally promotable yet.

---

## Current recommended status

Retain publicly:

- `step31_simple_late_fusion` as the retained Step31 multi-view bridge reference
- `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30` as the current retained isolated Step32 candidate reference

Do **not** currently do any of the following:

- formal Step32 checkpoint promotion
- backend transfer reopening
- threshold fiddling for the current candidate checkpoint
- reopening parked Step22–31 micro-lines

---

## Next smallest justified action

The next smallest justified action is **not** more threshold tuning for the same checkpoint.

For the current candidate checkpoint, threshold protocol is considered fixed.

Future work should either:

1. run another bounded Step32 progression under the same fixed protocol to test scale saturation, or
2. stop here and use `candidate_next` as the public isolated Step32 candidate reference until a genuinely stronger Step32 mechanism is ready.
