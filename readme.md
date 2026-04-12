# Relational / Event-Centric World Model

A staged research project on **local event-centric world modeling** in a fully structured synthetic graph environment.

The central question is:

> If world changes are sparse, local, and event-driven, can a model do better by first proposing the relevant event region and then rewriting only that local region, instead of predicting the whole next graph monolithically?

This repository is intentionally scoped to a clean synthetic regime first:

- structured graph state input only
- no raw images
- no real-world data
- no hypergraph formulation
- no LLM integration

The goal is to isolate the local-event modeling hypothesis before moving to richer realism.

---

## 1. Project Scope

The current working stack is an **event-centric local world model** with two explicit stages:

1. **Proposal**
   - predict the local event scope in the graph
2. **Rewrite**
   - update only that local region to produce the next graph state

The repository has already moved beyond early oracle-local feasibility. It now covers:

- oracle-local rewrite feasibility
- learned-scope bridging
- robust rewrite under learned-scope noise
- sequential composition consistency
- short-horizon rollout stability
- transfer to more complex multi-event structural regimes
- transfer to noisy structured observation regimes
- scope/edit failure decomposition under learned scope
- proposal-side internal completion and interface-side fallback studies
- noisy multi-event interaction benchmarking

A key long-running project theme is the distinction between:

- **changed region**: nodes / edges that actually change
- **event scope**: the local region associated with the event, which may include extra context

Many later bottlenecks come from the gap between predicting the correct scope and rewriting the correct changed subset inside that scope.

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

### Dataset capabilities

The data pipeline now supports:

- variable-size graphs
- train / val / test splits
- one-event transitions
- two-event composition data
- sequential composition data
- rollout data
- multi-event interaction data
- noisy structured observation data
- noisy multi-event interaction data
- changed-region annotations
- event-scope annotations
- interaction labels such as:
  - `fully_independent`
  - `partially_dependent`
  - `strongly_interacting`

---

## 3. Repository Structure

The codebase is organized around data generation, proposal/rewrite models, and stage-specific training/evaluation scripts.

```text
relational-event-world-model/
├── data/
│   ├── generate_graph_event_data.py
│   ├── generate_step22_noisy_step5_data.py
│   ├── generate_step23_noisy_step5_train.py
│   ├── dataset.py
│   ├── collate.py
│   └── *.pkl datasets
│
├── models/
│   ├── baselines.py
│   ├── oracle_local.py
│   ├── oracle_local_delta.py
│   └── proposal.py
│
├── train/
│   ├── train_scope_proposal.py
│   ├── train_scope_proposal_noisy_obs.py
│   ├── train_step3_sequential_consistency.py
│   ├── train_step4_rollout_finetune.py
│   ├── train_step5_interaction_finetune.py
│   ├── train_step6_noisy_rewrite_finetune.py
│   ├── train_step6_joint_noisy_finetune.py
│   ├── train_step23_noisy_interaction_proposal.py
│   ├── train_step24_noisy_interaction_joint.py
│   ├── eval_scope_proposal.py
│   ├── eval_proposal_conditioned_delta.py
│   ├── eval_sequential_composition_consistency.py
│   ├── eval_rollout_stability.py
│   ├── eval_step5_multievent_regime.py
│   ├── eval_noisy_structured_observation.py
│   ├── eval_step22_noisy_multievent_interaction.py
│   └── ...
│
├── checkpoints/
├── artifacts/
├── docs/
└── README.md
```

---

## 4. Stage Summary

### Step 1 — Oracle-local rewrite feasibility

**Question:** If the correct event scope is given, can local rewrite itself be learned?

**Answer:** Yes.

This established the basic viability of local event-centric modeling. A key early lesson was that **edge behavior** is the main bottleneck, especially:

- delete accuracy
- changed-vs-context balance
- avoiding trivial keep/copy collapse

---

### Step 2 — Robust rewrite under learned-scope noise

**Question:** Once scope is learned rather than oracle-perfect, can rewrite remain reliable?

This became the central learned-scope bottleneck.

#### Current broad Step 2 default

**`W012`**

Interpretation:

- strongest broad learned-scope rewrite compromise so far
- best overall balance between edit sensitivity and context preservation
- safest broad default to carry forward

#### Strong alternatives worth keeping

- `WG025`
  - strongest edit-preserving alternative
- `DR005`
  - delete-rescue anchor

**Step 2 status:** sufficiently closed for now.

---

### Step 3 — Sequential composition consistency

The original exact reverse-order final-state matching setup turned out to be vacuous.

The real Step 3 substrate became **sequential composition consistency**.

#### Current Step 3 candidate

**`C005`**

Interpretation:

- light consistency objective helps on the real sequential-composition substrate
- path-gap reductions are real
- not a new broad default

---

### Step 4 — Rollout stability

Main result:

- rollout degradation is real
- degradation is cumulative
- but not catastrophic over short horizons

The characteristic failure mode is conservative drift:

- changed-edge stays weak
- add behavior tends to collapse
- context-edge remains relatively strong

#### Current Step 4 candidate

**`R050`**

---

### Step 5 — More complex multi-event structural regime

Step 5 introduced a harder synthetic regime with:

- three-event sequences
- fully independent chains
- partially dependent chains
- strongly interacting chains

Main result:

- the method family transfers
- the main new bottleneck is **event interaction complexity**

#### Current Step 5 defaults

- broad default: **`W012`**
- interaction-aware alternative: **`I1520`**

Interpretation:

- `W012` is the safest broad-transfer model
- `I1520` is the stronger interaction-sensitive alternative
- gains from interaction-aware tuning exist, but broad stability remains important

---

### Step 6 — Noisy structured observation

Step 6 introduced corrupted structured observations while keeping the latent target next-state clean.

Main result:

- the family remains usable under noisy structured observations
- the main bottleneck shifts toward proposal robustness and proposal/rewrite coupling

#### Current noisy proposal front-end

**calibrated `P2`**

- `node_threshold = 0.15`
- `edge_threshold = 0.10`

#### Current Step 6 main candidate

**`RFT1 + calibrated P2`**

Interpretation:

- best balanced noisy-observation stack found so far
- better overall than `W012 + calibrated P2`
- light joint noisy fine-tuning showed signal, but did not beat this stack

**Step 6 status:** sufficiently closed for now.

---

### Steps 7–10 — Scope/edit gap, edge miss anatomy, closure, and rescued-scope decomposition

These stages reframed the learned-scope bottleneck more precisely.

Main results:

- the dominant overall bottleneck is **edge-side proposal miss / out-of-scope miss**
- the dominant missed-edge subtype is:
  - **both endpoints already inside predicted node scope**
- proposal-side internal edge completion is a real mechanism family
- the context loss from budgeted internal completion does **not** mainly come from broad rewrite spillover
- it comes overwhelmingly from **rescued false-scope edges** being admitted and then actively rewritten

This established a useful proposal-side branch candidate:

- **Step 9c fixed-budget internal completion @ 10%**

But that branch did not become a new default.

---

### Steps 11–15 — Proposal-side minimal guard line

A series of minimal proposal-side guard / reranking / objective / representation experiments were tested to reduce false-scope rescue admissions.

Main result:

- oracle guard ceilings were real
- but the minimal learned guard family did not materially improve the system-level tradeoff
- continuing tiny guard tweaks became low value

**Status:** parked for now.

---

### Steps 16–21 — Rewrite-side fallback interface line

An oracle rewrite-side fallback probe showed a very strong ceiling:

- if rescued false-scope edges are prevented from being actively rewritten,
- most of the Step 9c context loss can be recovered,
- while preserving the main changed-edge / add / delete gains

This established that the interface direction is real.

However, the learned chooser line was extensively tested and did not become usable:

- compact chooser collapsed to all-fallback at default operating point
- chooser was not ranking-dead, but top-tail quality was too weak
- objective-only tweaks were insufficient
- shallow/local representation probes helped only slightly
- a contained two-path local-subgraph chooser still did not materially beat the best shallow probe

**Status:** parked for now.

---

### Step 22 — Noisy multievent interaction benchmark

This stage combined:

- Step 5 multi-event interaction complexity
- Step 6 noisy structured observation

Main result:

- the new substrate is meaningful and nontrivial
- `RFT1 + calibrated P2` remains the safest broad noisy stack on this substrate
- noise does **not** catastrophically collapse the family
- the main amplified bottleneck is on **strongly interacting** sequences
- the failure is mainly **edit / proposal coverage limited**, not broad context collapse

---

### Step 23 — Proposal-only noisy interaction-aware adaptation

Proposal-only adaptation from noisy `P2`, with rewrite frozen as `RFT1`, was tested on the Step 22 substrate.

Main result:

- it improved broad full/context-edge mainly by becoming more conservative
- proposal edge recall dropped
- out-of-scope miss worsened
- changed-edge / add / delete weakened
- strongly interacting proposal miss did not improve materially

**Conclusion:** proposal-only adaptation does not solve the noisy interaction bottleneck.

---

### Step 24 — Light joint noisy interaction-aware coupling

A first light joint proposal+rewrite fine-tuning pass was tested on the Step 22 substrate.

Main result:

- overall proposal coverage / miss improved slightly relative to noisy `P2 + RFT1`
- it avoided the conservative collapse seen in Step 23
- but it did **not** improve the intended strongly-interacting downstream changed-edge/delete behavior

**Conclusion:** informative, but not promotable.

---

### Steps 25–29 — Oracle headroom, factorization, and retained noisy interaction branch

These stages completed the current noisy multievent interaction line.

Main results:

- Step 25 oracle-scope headroom showed large proposal-coverage headroom for strongly interacting changed/add behavior, while delete remained partly rewrite-limited.
- Step 26 coverage-emphasized joint training recovered real strongly interacting changed/add headroom, but broad full/context-edge stability dropped and strongly interacting delete collapsed.
- Step 27 factorization showed the positive Step 26 signal is primarily proposal-side, while the negative signal is primarily rewrite-side.
- Step 28 RFT1-anchored joint training did not improve on the cleaner factorized reference.
- Step 29 confirmed **`Step26 proposal + RFT1`** as the retained noisy interaction-aware branch candidate.

Retained noisy interaction-aware branch candidate: `Step26 proposal + RFT1`. It should be used when the evaluation priority is strongly-interacting changed/add recovery under noisy multievent interaction, not when broad context/full-edge stability is the priority.

This branch is not the broad noisy default because broad full-edge / context-edge remain materially weaker than `RFT1 + calibrated P2`.

**Status:** sufficiently explored for the current phase; park this local substrate line unless a genuinely new mechanism is introduced.

---

## 5. Current Stable Defaults

### Broad structured-world default
**`W012`**

### Broad noisy default
**`RFT1 + calibrated P2`**

### Interaction-aware alternative
**`I1520`**

### Consistency reference
**`C005`**

### Rollout-aware reference
**`R050`**

### Noisy proposal front-end
**calibrated `P2`**
- `node_threshold = 0.15`
- `edge_threshold = 0.10`

### Retained noisy interaction-aware branch candidate
**`Step26 proposal + RFT1`**

This is a branch candidate only, not the broad noisy default.

---

## 6. Important Non-Default References

These are informative and should be preserved, but they are not the current defaults:

- `WG025`
  - strongest edit-preserving Step 2 alternative
- `DR005`
  - delete-rescue anchor
- `J05`
  - first light noisy joint line with real coupling signal
- `Step 9c` fixed-budget internal completion @ 10%
  - proposal-side branch candidate with meaningful but not default-level value
- `Step26 proposal + RFT1`
  - retained noisy interaction-aware branch candidate for strongly interacting changed/add recovery
  - not the broad noisy default because broad full/context-edge stability is weaker than `RFT1 + calibrated P2`

---

## 7. What Is Explicitly Not Being Reopened Right Now

The following lines are considered sufficiently explored or temporarily parked:

- more Step 2 keep/rescue micro-tuning
- more Step 6 threshold-only retuning
- more Step 6 temperature-only calibration variants
- the parked chooser-interface line from Steps 16–21
- more proposal-only noisy interaction micro-tuning after Step 23
- more Step 26 / Step 28 joint-recipe tweaks on the noisy interaction substrate
- the Step 22–29 local substrate line, unless a genuinely new mechanism is introduced
- more tiny guard / reranking / minimal-representation proposal-side tweaks after Steps 11–15

---

## 8. Current Interpretation

The repository now supports a coherent view of local event-centric world modeling in a structured synthetic graph world:

1. local rewrite is learnable
2. learned-scope bridging is viable
3. proposal/rewrite interface robustness is the central learned-scope bottleneck
4. sequential composition consistency is real
5. rollout degradation is real but manageable
6. the family transfers to harder multi-event structural regimes
7. noisy structured observation is survivable
8. the proposal-side internal completion line has real but non-default value
9. the rewrite-side fallback interface has a strong oracle ceiling, but no usable learned chooser yet
10. the new noisy multievent substrate is established
11. on that substrate, `Step26 proposal + RFT1` is the retained noisy interaction-aware branch candidate
12. the Step 22–29 line is now parked for this phase unless a genuinely new mechanism is introduced

---

## 9. Recommended Usage Notes

If you are running new experiments in this repository:

- use `W012` as the default broad clean model
- use `RFT1 + calibrated P2` as the default noisy system stack
- use `I1520` when interaction sensitivity is the main clean-structured concern
- use `Step26 proposal + RFT1` only when the target is noisy strongly interacting changed/add recovery and a broad full/context-edge tradeoff is acceptable
- compare against `WG025`, `DR005`, `J05`, or `Step 9c` only when the mechanism under study directly touches their tradeoff axes
- do not continue the Step 22–29 noisy multievent interaction substrate line without a genuinely new mechanism

---

## 10. Next-Phase Entry Point

The next phase should **not** reopen old Step 2–6 tuning loops.

It should begin from the consolidated defaults:

- broad structural default: `W012`
- noisy broad default: `RFT1 + calibrated P2`
- interaction-aware alternative: `I1520`
- consistency reference: `C005`
- rollout reference: `R050`
- retained noisy interaction-aware branch candidate: `Step26 proposal + RFT1`

The Step 22–29 noisy multievent interaction substrate is sufficiently explored for the current phase. Future work should preserve `Step26 proposal + RFT1` as a branch candidate, but should not continue local tweaks on this line unless a genuinely new mechanism is introduced.

---

## 11. Summary in One Sentence

This repository now supports a staged, mechanistically interpretable local event-centric world model pipeline in a structured synthetic graph world, with stable defaults for clean and noisy settings, established interaction and rollout references, and a parked noisy multievent interaction line whose retained branch candidate is `Step26 proposal + RFT1`.
