# Relational / Event-Centric World Model

A public research repository for **local event-centric world modeling** in a fully structured synthetic graph environment.

The central question is simple:

> If world changes are sparse, local, and event-driven, can a model do better by first proposing the relevant event region and then rewriting only that local region, instead of predicting the whole next graph monolithically?

This project is intentionally scoped to a clean synthetic regime first:

- structured graph state input only
- synthetic data only
- no raw real-world images
- no real-world datasets
- no hypergraph formulation
- no LLM integration

The goal is to isolate the local-event modeling hypothesis before moving to richer observation families.

---

## Public project scope

The working stack remains a two-stage local world model:

1. **Proposal**
   - predict the local event scope in the graph
2. **Rewrite**
   - update only that local region to produce the next graph state

A long-running theme in the repository is the distinction between:

- **changed region**: nodes / edges that actually change
- **event scope**: the local region associated with the event, which may include extra context

Many later bottlenecks come from the gap between predicting the correct scope and rewriting the correct changed subset inside that scope.

---

## Stable defaults currently in use

These are the retained public references to compare new experiments against:

- **Broad clean default:** `W012`
- **Broad noisy default:** `RFT1 + calibrated P2`
- **Interaction-aware clean alternative:** `I1520`
- **Consistency reference:** `C005`
- **Rollout-aware reference:** `R050`
- **Retained noisy interaction-aware branch candidate:** `Step26 proposal + RFT1`
  - this is a branch candidate only, not the broad noisy default

Current calibrated noisy-observation proposal thresholds for `P2`:

- `node_threshold = 0.15`
- `edge_threshold = 0.10`

---

## What is already established

The public repository documents a staged progression through:

- oracle-local rewrite feasibility
- learned-scope bridging
- robust rewrite under learned-scope noise
- sequential composition consistency
- short-horizon rollout stability
- transfer to harder multi-event structural regimes
- noisy structured observation
- noisy multi-event interaction benchmarking
- weak-observation bridge probes
- multi-view synthetic observation bridging

The main retained conclusions so far are:

- local rewrite is learnable
- learned-scope bridging is viable
- proposal/rewrite interface robustness is the central learned-scope bottleneck
- noisy structured observation is survivable
- the noisy multievent interaction substrate is established, but its local tweak line is parked for now
- synthetic multi-view observation is a real stronger bridge family

---

## Current phase status

### Parked lines

The following lines are considered sufficiently explored or parked unless a genuinely new mechanism is introduced:

- old Step 2 keep/rescue micro-retuning
- old Step 6 threshold-only / temperature-only recalibration loops
- the Step 22–29 noisy multievent interaction local tweak line
- the Step 30 rescue/decode micro-lines
- the Step 31 learned-vs-late-fusion micro-line

### Retained Step31 reference

For the synthetic multi-view bridge family, the current retained backend-transfer reference is:

- **`step31_simple_late_fusion`**

### Active next-phase question

The active next-phase entry is:

- **Step32 = synthetic rendered / image-like bridge probe**

This phase asks whether the observation substrate can be pushed from structured weak multi-view signals toward a more rendered / image-like synthetic observation family while keeping evaluation mechanistic and comparable to the existing backend.

---

## Current public Step32 status

Step32 is **not** yet a formally promoted system checkpoint.

However, the bounded rendered-bridge line now has a meaningful isolated candidate result:

- current best isolated Step32 candidate: `step32_rendered_bridge_candidate_next`
- validated calibrated operating point: `clean=0.30`, `noisy=0.30`
- at that operating point, the learned bridge beats the trivial rendered baselines
- it remains **calibration-dependent**
- it still does **not** beat the retained Step30 rev6 recovery reference
- backend transfer remains **closed** for now

So the right public interpretation is:

> Step32 has moved beyond pure smoke infrastructure and now has a real calibrated candidate, but it is not yet strong enough for formal promotion or backend-transfer reopening.

A compact public note for this line lives in:

- `docs/STEP32_RENDERED_BRIDGE_CANDIDATE_STATUS.md`

---

## Repository structure

```text
relational-event-world-model/
├── data/
├── models/
├── train/
├── checkpoints/
├── artifacts/
├── docs/
└── README.md
```

In practice:

- `data/` contains synthetic data generation and dataset loaders
- `models/` contains proposal / rewrite model definitions and baselines
- `train/` contains stage-specific training and evaluation scripts
- `docs/` contains phase summaries, retained-branch notes, and current status docs

Generated datasets, checkpoints, and artifacts are intentionally treated differently from public source and summary docs; not every local experimental byproduct is meant to be committed as a public canonical result.

---

## Recommended public usage notes

If you are reading or extending this repository:

- use `W012` as the broad clean default
- use `RFT1 + calibrated P2` as the broad noisy default
- use `I1520` when clean structured interaction sensitivity is the main target
- use `Step26 proposal + RFT1` only when the target is noisy strongly-interacting changed/add recovery and a broad full/context tradeoff is acceptable
- treat `step31_simple_late_fusion` as the retained structured multi-view bridge reference
- treat the current Step32 line as an active probe with a validated calibrated candidate, not a fully promoted replacement for earlier retained references

---

## Current public summary in one paragraph

This repository supports a staged, mechanistically interpretable local event-centric world model pipeline in a structured synthetic graph world. The current public defaults remain `W012` for broad clean use and `RFT1 + calibrated P2` for broad noisy use, with `I1520`, `C005`, `R050`, and `Step26 proposal + RFT1` preserved as targeted references. The Step22–31 local tweak lines are parked, `step31_simple_late_fusion` is the retained multi-view bridge reference, and Step32 currently stands as a synthetic rendered / image-like bridge probe with a validated calibrated candidate but no formal promotion yet.
