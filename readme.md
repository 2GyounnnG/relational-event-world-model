# Relational / Event-Centric World Model

A staged research project on **local event-centric world modeling** in a fully structured synthetic graph environment.

The core question is:

> If world changes are fundamentally sparse, local, and event-driven, can a model do better by first identifying the relevant event region and then rewriting only that local region, instead of predicting the whole next graph monolithically?

This repository is intentionally scoped to a clean synthetic regime first:

- structured graph state input only
- no raw images
- no real-world data
- no hypergraph formulation yet
- no LLM integration

The purpose is to isolate the local-event modeling hypothesis before moving to noisier observations, more complex interaction structure, and later-stage realism.

---

## 1. Project Scope

The current repository studies a two-part local world model:

1. **Proposal**
   - predicts which local region of the graph is relevant to the current event(s)
2. **Rewrite**
   - predicts how that local region should change

The project has already moved well beyond the earliest oracle-only feasibility stage. It now covers:

- oracle-local rewrite feasibility
- learned-scope bridging
- robust rewrite under learned scope noise
- sequential composition consistency
- short-horizon rollout stability
- transfer to more complex multi-event structural regimes
- transfer to noisy structured observation regimes

---

## 2. Synthetic Graph-Event Environment

The environment is a synthetic graph world with:

- node type
- node state features
- adjacency structure

### Event family

Current event types:

1. `node_state_update`
2. `edge_add`
3. `edge_delete`
4. `motif_type_flip`

### Dataset properties

The data pipeline supports:

- variable-size graphs
- train / val / test splits
- one-event transitions
- two-event transitions
- independent two-event pairs
- changed-region annotations
- event-scope annotations
- matched sequential-composition data
- rollout data
- multi-event interaction data
- noisy structured observation data

A central distinction is preserved throughout the project:

- **changed region**: nodes / edges that actually changed
- **event scope**: the local region associated with the event, which may include extra context

That distinction became one of the defining project themes: many later bottlenecks come from the gap between predicting the correct scope and rewriting the correct changed subset inside that scope.

---

## 3. Repository Structure

```text
relational-event-world-model/
├── data/
│   ├── generate_graph_event_data.py
│   ├── dataset.py
│   ├── collate.py
│   ├── graph_event_train.pkl
│   ├── graph_event_val.pkl
│   ├── graph_event_test.pkl
│   ├── graph_event_step3_matched_test.pkl
│   ├── graph_event_step3_sequential_train.pkl
│   ├── graph_event_step3_sequential_val.pkl
│   ├── graph_event_step3_sequential_test.pkl
│   ├── graph_event_rollout_train.pkl
│   ├── graph_event_rollout_val.pkl
│   ├── graph_event_rollout_test.pkl
│   ├── graph_event_step5_train.pkl
│   ├── graph_event_step5_val.pkl
│   ├── graph_event_step5_test.pkl
│   ├── graph_event_step6a_train.pkl
│   ├── graph_event_step6a_val.pkl
│   └── graph_event_step6a_test.pkl
│
├── models/
│   ├── baselines.py
│   ├── oracle_local.py
│   ├── oracle_local_delta.py
│   └── proposal.py
│
├── train/
│   ├── train_baseline_global.py
│   ├── oracle_sanity_check.py
│   ├── train_oracle_local.py
│   ├── train_oracle_local_delta.py
│   ├── train_oracle_local_delta_earlystop.py
│   ├── train_scope_proposal.py
│   ├── train_scope_proposal_noisy_obs.py
│   ├── eval_scope_proposal.py
│   ├── eval_proposal_conditioned_delta.py
│   ├── eval_independent_pair_consistency.py
│   ├── eval_sequential_composition_consistency.py
│   ├── eval_rollout_stability.py
│   ├── eval_step5_multievent_regime.py
│   ├── eval_noisy_structured_observation.py
│   ├── train_step3_sequential_consistency.py
│   ├── train_step4_rollout_finetune.py
│   ├── train_step5_interaction_finetune.py
│   ├── train_step6_noisy_rewrite_finetune.py
│   ├── train_step6_joint_noisy_finetune.py
│   └── ...
│
├── checkpoints/
├── docs/
├── README.md
└── ...
```

---

## 4. Stage Summary

The cleanest way to understand the project is by stage.

### Step 1 — Oracle-local rewrite feasibility

**Question:** If the correct event scope is given, can local rewrite itself be learned?

**Answer:** Yes.

This established the core viability of local event-centric modeling. Oracle-local rewrite is learnable, and merge-back sanity checks pass.

A key early lesson was that **edge behavior** is the main bottleneck, especially:

- delete accuracy
- changed-vs-context balance
- avoiding trivial keep/copy collapse

This led to the current oracle-scope edge reference:

- `delta_keep_weight = 1.10`
- `delta_add_weight = 1.0`
- `delta_delete_weight = 3.0`

This later became the `keep110` reference.

---

### Step 1b — Typed failure analysis

A major type-side result was that `motif_type_flip` is not just a generic hard classification case. The model has a strong **copy-current-type bias**, so true flips are the real difficulty.

This established that:

- locality helps
- but flip-specific supervision still matters

---

### Step 1c — Learned-scope bridge

**Question:** Can a learned proposal replace oracle scope well enough to support local rewrite?

**Key findings:**

- node-only proposal is not enough
- explicit node+edge proposal is necessary
- learned-scope bridge is viable
- threshold choice matters substantially

The strongest learned-scope bridge reference became:

- explicit node+edge proposal
- `node_flip_weight = 2.0`
- `node_threshold = 0.20`
- `edge_threshold = 0.15`
- rewrite initialized from `keep110`

This gave a meaningful learned-scope bridge, but did not yet solve robust rewrite under learned-scope noise.

---

### Step 2 — Robust rewrite under learned-scope noise

**Question:** Once scope is learned rather than oracle-perfect, can rewrite remain reliable?

This became the central bottleneck of the project.

Important negative results:

- predicted-only training collapses toward keep/copy
- union-scope training remains too copy-heavy

The main lesson was:

> The proposal/rewrite interface is not a detail; it is the core learned-scope bottleneck.

#### Current broad Step 2 default

**`W012`**

Interpretation:

- strongest broad learned-scope rewrite compromise so far
- best overall balance between edit sensitivity and context preservation
- not perfect, but the most stable broad reference

#### Strong alternatives worth keeping

- `WG025`
  - more edit-preserving
  - stronger on delete / changed / flip
  - weaker on context stability

- `DR005`
  - delete-rescue style alternative
  - important proof that targeted delete recovery is real
  - not the broad default

**Step 2 status:** sufficiently closed for now.

---

### Step 3 — Sequential composition consistency

**Question:** If two events are independent, does the model behave consistently when event order changes?

An exact reverse-order final-state matched dataset was first constructed, but this turned out to be effectively vacuous: final targets commute by construction, so exact reverse-order gaps were zero.

The real Step 3 substrate became:

**sequential composition consistency**

That is, comparing:

- first-step quality
- second-step quality
- path-level sensitivity under event reordering

This produced real, nontrivial consistency signals.

#### Current Step 3 candidate

**`C005`**

Interpretation:

- light consistency objective helps on the true sequential-composition substrate
- path-gap reductions are real
- still trades against step quality
- useful Step 3 mode, not a universal replacement

---

### Step 4 — Rollout stability

**Question:** Can the learned-scope local world model roll forward autoregressively without rapidly collapsing?

Main result:

- rollout degradation is real
- degradation is cumulative
- but it is not catastrophic over short horizons

The characteristic failure mode is conservative drift:

- changed-edge behavior remains weak
- add behavior tends to collapse
- context edge accuracy stays relatively strong

#### Current Step 4 candidate

**`R050`**

Interpretation:

- best rollout-aware candidate so far
- improves rollout-specific edit behavior
- still trades against overall stability

---

### Step 5 — More complex synthetic structural regime

**Question:** Does the method family transfer to a more complex multi-event structural world?

Step 5 introduced a harder synthetic regime with:

- 3-event sequences
- fully independent chains
- partially dependent chains
- strongly interacting chains

Main result:

- the method family does transfer
- the main new bottleneck is **event interaction complexity**

#### Current Step 5 defaults

- broad default: **`W012`**
- interaction-aware alternative: **`I1520`**

Interpretation:

- `W012` is the safest broad-transfer model
- `I1520` is the stronger interaction-sensitive alternative
- interaction-aware training helps, but gains saturate quickly

---

### Step 6 — Noisy structured observation

**Question:** What happens when the model no longer sees a perfect graph state, but only a noisy structured observation?

This became the bridge from ideal structured-world modeling toward more realistic imperfect observation.

#### Step 6a — Noisy structured observation benchmark

Main result:

- the method family remains usable under noisy structured observations
- the main bottleneck shifts to proposal robustness, especially on the edge side

#### Step 6b — Noisy proposal training

Main result:

- noisy-observation proposal training helps
- proposal-only best checkpoint becomes **`P2`**

#### Step 6c — Global threshold calibration

Main result:

- calibration matters
- after threshold calibration, `P2` becomes the system-level best proposal front-end

Current best noisy-observation front-end:

- **calibrated `P2`**
- `node_threshold = 0.15`
- `edge_threshold = 0.10`

#### Step 6d — Regime-aware threshold calibration

Main result:

- no meaningful gain over global calibration

Useful negative result:

- remaining calibration issues are not simply per-regime threshold mismatch

#### Step 6e — Temperature scaling / confidence calibration

Main result:

- no meaningful downstream gain over threshold calibration
- calibration beyond thresholds did not help in the first clean pass

Useful negative result:

- proposal-side simple confidence calibration appears close to saturation

#### Step 6f — Rewrite-only noisy adaptation

Proposal frozen as calibrated `P2`, rewrite adapted under noisy structured observation.

Main result:

- rewrite-only noisy adaptation helps
- gains are incremental, not transformative
- best current candidate from this line is:

### **`RFT1 + calibrated P2`**

This is the current best broad noisy-observation system candidate.

#### Step 6g — Light joint noisy proposal+rewrite fine-tuning

A light joint noisy-observation fine-tuning pass was tested.

Main result:

- coupling signal is real
- but the joint line does **not** beat `RFT1 + calibrated P2`
- joint results are informative, but not the new default

#### Current Step 6 main candidate

### **`RFT1 + calibrated P2`**

This is the current best noisy structured observation system stack.

**Step 6 status:** sufficiently closed for now.

---

## 5. Current Stable Defaults

At the current project checkpoint, the most useful defaults are:

### Broad structured-world default
**`W012`**

### Step 3 consistency reference
**`C005`**

### Step 4 rollout-aware reference
**`R050`**

### Step 5 interaction-aware alternative
**`I1520`**

### Step 6 proposal front-end
**calibrated `P2`**
- `node_threshold = 0.15`
- `edge_threshold = 0.10`

### Step 6 broad noisy-observation system default
**`RFT1 + calibrated P2`**

---

## 6. Strong Alternatives Worth Keeping

These are not the current defaults, but remain important reference points:

- `WG025`
  - strongest edit-preserving Step 2 alternative

- `DR005`
  - strongest delete-rescue Step 2 anchor

- `J05`
  - informative joint noisy-observation result
  - useful evidence that coupling has signal
  - not better than `RFT1 + calibrated P2`

---

## 7. What Is Explicitly Not Being Reopened Right Now

The following lines are considered sufficiently explored for the current phase and should **not** be reopened unless a later phase specifically demands it:

- more Step 2 keep/rescue micro-tuning
- more Step 6 threshold-only retuning
- more Step 6 temperature-only calibration variants
- reopening proposal-only architecture changes without a genuinely new bottleneck
- broader local tuning loops inside Steps 2–6 without a new mechanism-level question

---

## 8. Current Interpretation

The repository now supports a coherent view of local event-centric world modeling in a structured synthetic graph world:

1. local rewrite is learnable
2. learned-scope bridging is viable
3. proposal/rewrite interface robustness is the central learned-scope bottleneck
4. sequential composition consistency is a real issue
5. rollout degradation is real but manageable
6. the family transfers to a more complex multi-event structural regime
7. noisy structured observation is survivable
8. proposal-side noisy robustness can be improved substantially
9. rewrite-side noisy adaptation provides the strongest current Step 6 system result
10. fully joint noisy coupling has signal, but not yet a decisive net gain

---

## 9. Recommended Usage Notes

If you are running new experiments in this repository:

- use `W012` as the default broad model unless you specifically care about stronger edit preservation
- use `I1520` when event interaction sensitivity is the main concern
- use `RFT1 + calibrated P2` when working in noisy structured observation settings
- compare against `WG025`, `DR005`, or `J05` only when the mechanism under study directly touches their corresponding tradeoff axes

---

## 10. Next-Phase Entry Point

The next phase should **not** start by reopening old Step 2–6 tuning loops.

Instead, it should begin from the consolidated defaults:

- broad structural default: `W012`
- consistency reference: `C005`
- rollout reference: `R050`
- interaction-aware alternative: `I1520`
- noisy-observation front-end: calibrated `P2`
- noisy-observation broad system: `RFT1 + calibrated P2`

The next phase should ask a genuinely new question rather than revisiting already-saturated local tradeoff lines.

---

## 11. Summary in One Sentence

This repository now supports a staged, mechanistically interpretable local event-centric world model pipeline in a structured synthetic graph world, with stable defaults for learned-scope rewrite, sequential consistency, rollout stability, interaction complexity, and noisy structured observation.
