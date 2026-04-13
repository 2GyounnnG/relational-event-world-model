# Step33 Synthetic Physics-Like Benchmark Proposal

## Status

This is a planning proposal, not an implementation note.

Step33 should be treated as a new synthetic benchmark design phase after Step32, not as a training phase, backend transfer phase, or real-data phase.

Do not use this proposal to reopen parked Step22-31 micro-lines.
Do not use it to switch to real-world data, raw videos, CLEVRER directly, hypergraphs, or LLM integration.

The project remains a structured synthetic graph-event world model with a proposal/rewrite spine.

---

## 1. Why Step33 Is The Right Next Phase After Step32

Step32 pushed the project from structured weak observations toward rendered synthetic observations. It is now source-public and documented, with:

- a best calibrated isolated reference
- a best observed default-threshold isolated reference
- a same-scale variance caveat
- backend transfer still closed because Step30 rev6 remains ahead

The important Step32 lesson is that perception-like rendering alone is not the next obvious source of progress. Scale-up helped default-threshold behavior, but calibrated recovery appears saturated around the current candidate level.

Step33 should therefore add a new controlled scientific pressure: local physical dynamics.

The goal is not to make the benchmark harder by adding raw perception. The goal is to keep exact labels and mechanistic evaluation while making event consequences more physically grounded:

- local interventions can propagate through springs or contacts
- event scope and changed region can differ for meaningful physical reasons
- state recovery includes continuous quantities such as position and velocity
- edge semantics become physical constraints, not just abstract relations

This is a better next phase than more Step32 scale-only work.

---

## 2. Why Not Jump Directly To CLEVRER Or Real Data

Jumping directly to CLEVRER, real videos, or real images would introduce too many uncontrolled difficulties at once:

- video perception
- object discovery and tracking
- occlusion and visual ambiguity
- QA-style task formatting
- annotation and causal-label mismatch
- uncontrolled physics and domain variation

Those are not the current project's core question.

The current project is about local event proposal, local rewrite, changed-region recovery, and mechanistic diagnostics inside a controlled graph-event world. Step33 should preserve that structure.

Real data may become useful later, but only after the project has a physics-like synthetic benchmark where the event/rewrite machinery is known to work under controlled physical dynamics.

---

## 3. Why CLEVRER Is A Reference, Not The Next Benchmark

CLEVRER is useful inspiration because it is physics-like, causal, and object-centric.

But CLEVRER should not be adopted directly as Step33 because:

- it is video-first, while Step33 should be structured-first
- it has QA-style task pressure, while this repo is event-transition and rewrite oriented
- it would make raw perception and question interpretation major difficulties too early
- it does not naturally preserve the repo's local proposal/rewrite decomposition
- it would blur event scope vs changed-region diagnostics

Step33 should be CLEVRER-like only in spirit:

- small physical objects
- local interactions
- causal consequences
- controlled counterfactual-style labels

It should not be CLEVRER in task form.

---

## 4. Core Design Goals

Step33 must preserve:

- graph-state transition framing
- local event scope proposal
- local rewrite of next state
- fully controlled labels
- event scope vs changed region diagnostics
- clean/noisy observation variants
- oracle and trivial baselines before learned candidates
- recovery-first evaluation
- mechanistic interpretability

Step33 should add:

- continuous physical state
- local forces and constraints
- physically meaningful edges
- propagation beyond the immediate event scope
- physical residual diagnostics

Step33 should not become:

- generic trajectory prediction
- video understanding
- raw image perception
- CLEVRER QA
- backend transfer
- a new mechanism search before the benchmark is validated

---

## 5. Smallest Viable Synthetic Physics-Like World

The preferred first Step33 world is a tiny 2D particle-spring-contact system.

Initial world:

- 6-10 particles
- 2D bounded box
- sparse spring graph
- optional local contact candidates
- typed particles
- deterministic simulator
- one local event per transition
- short rollout horizon after the event, for example 3-8 microsteps

Each particle has:

- type
- position
- velocity
- radius
- mass
- pinned/fixed flag

Each spring has:

- endpoints
- rest length
- stiffness
- damping
- active/inactive state

Contacts should remain simple at first:

- soft repulsion or contact flag based on distance
- no complex friction
- no rigid-body rotation
- no piles or chaotic multi-contact scenes

This world is small enough to label exactly and inspect manually, but physical enough to create meaningful local propagation.

---

## 6. Graph Schema

The graph represents a physical scene at time `t`.

### Nodes

Nodes represent particles or simple objects.

Candidate node features:

- object type
- position `x, y`
- velocity `vx, vy`
- radius
- mass
- pinned flag
- local kinetic energy proxy
- observation mask or confidence in noisy variants

### Edges

Edges represent physical relations.

There should be at least two edge channels:

1. Spring edges
   - spring active flag
   - stiffness
   - rest length
   - damping
   - current stretch/compression

2. Contact/proximity edges
   - distance
   - normalized gap
   - near-contact flag
   - penetration amount, if any
   - relative velocity along pair axis

The first Step33 benchmark can keep these edge types in one structured graph with typed edge features. It does not require hypergraphs.

### Targets

The transition target should include:

- next node state
- next spring/contact edge state
- event type
- event scope
- changed node mask
- changed edge mask
- optional hop-distance labels from event scope
- physical residual labels

---

## 7. Event Family

Events should be local, discrete, and physically interpretable.

Minimal first event family:

1. `node_impulse`
   - apply a velocity impulse to one particle
   - event scope: one node

2. `spring_break`
   - deactivate one existing spring
   - event scope: one edge and its endpoints

3. `spring_retension`
   - change one spring rest length or stiffness
   - event scope: one edge and its endpoints

Optional later events:

- `spring_attach`
- `pin_toggle`
- `mass_shift`
- `radius_shift`
- `local_damping_change`

Do not start with:

- global gravity changes
- hidden global mode switches
- many simultaneous events
- random teleports
- complex collision piles

The event should be local, but its consequences may propagate through physically connected neighbors. That is the point of Step33.

---

## 8. Observation Regime

Step33 should start with structured observations.

Initial structured observation can include:

- noisy or clean positions
- noisy or clean velocities
- observed particle type
- spring hints
- contact/proximity hints
- pair distances
- pair relative velocities
- masks/confidence values

Observation variants should follow the project's existing style:

- clean structured observation
- noisy structured observation
- possibly partial edge observation
- possibly corrupted contact hints

Rendered observation can be added later as a secondary bridge, not as the first Step33 difficulty.

A later rendered variant could draw:

- disks for particles
- lines for springs
- contact halos
- velocity arrows

But raw perception should not be the first Step33 challenge. The first challenge should be whether the proposal/rewrite machinery can handle controlled physical propagation.

---

## 9. Labels And Diagnostics

Step33 should be fully labeled.

Required labels:

- event type
- event scope nodes
- event scope edges
- changed nodes
- changed edges
- next node state
- next edge state
- hop distance from event scope
- clean simulator state before and after event

Changed region should be defined with explicit thresholds, for example:

- position delta above threshold
- velocity delta above threshold
- spring/contact state changed
- physical residual changed above threshold

Physical diagnostics should include:

- spring length residual
- contact penetration violation
- total kinetic energy change proxy
- pinned-object violation
- local momentum sanity check, where applicable
- constraint residual inside and outside proposed scope

The diagnostics should separate:

- missed changed region
- over-edited unchanged region
- correct event scope but poor rewrite
- wrong event scope but good broad rewrite
- physically implausible rewrite

---

## 10. Oracle And Trivial Baselines

Before any learned candidate, Step33 should include oracle and trivial baselines.

Required baselines:

- no-change baseline
- copy-state baseline
- one-hop event-neighborhood rewrite
- spring-neighborhood rewrite
- proximity/contact-neighborhood rewrite
- oracle event-scope baseline
- oracle changed-region baseline
- global rewrite upper-bound baseline

Proposal baselines:

- predict event node/edge from maximum local delta
- predict changed region from one-hop spring neighborhood
- predict changed region from distance/contact threshold
- random same-budget scope

Rewrite baselines:

- copy all unchanged state
- apply local event edit only
- simple deterministic local physics rollout using observed clean state
- oracle-scope rewrite

These baselines are needed to prove the benchmark has useful headroom before training any model.

---

## 11. Evaluation Protocol

Proposal quality should measure:

- event type accuracy
- event-scope precision/recall/F1
- changed-node precision/recall/F1
- changed-edge precision/recall/F1
- changed-region recall by hop distance
- scope budget curves
- false positive scope outside physically reachable region

Rewrite quality should measure:

- position MAE/MSE
- velocity MAE/MSE
- edge-state F1
- changed-region error
- unchanged-region preservation error
- physical residual error
- constraint violation rate

Local-vs-global diagnostics should measure:

- event scope vs changed region overlap
- under-editing physically affected neighbors
- over-editing distant unchanged nodes
- clean vs noisy split
- event-family split
- oracle gap
- trivial baseline gap

The benchmark should reward correct local causality and controlled rewrite, not broad global smoothing.

---

## 12. Minimal First Step33 Probe

The first Step33 experiment should be evaluation-first and small.

Recommended first probe:

- 6-8 particles
- sparse springs
- no rendered observation
- clean structured observation only
- event families:
  - `node_impulse`
  - `spring_break`
  - `spring_retension`
- short deterministic rollout
- small generated dataset
- no large training run
- no backend transfer

First probe goals:

- verify simulator determinism
- verify labels are exact and inspectable
- verify event scope and changed region are not identical
- verify oracle baselines have headroom
- verify trivial baselines are not already saturated
- verify metrics expose over-editing and under-editing

Only after this should Step33 move to noisy observation or learned proposal/rewrite smoke.

---

## 13. Why This Is Not Just Video Physics Or Trajectory Prediction

Step33 is not a video physics benchmark because raw video is not the first observation mode.

Step33 is not generic trajectory prediction because the central task is not just predicting future positions. The central task is:

1. propose the local event scope
2. identify the physically changed region
3. rewrite the next graph state locally
4. preserve unchanged structure
5. diagnose the gap between event scope and changed region

The benchmark remains event-centric, graph-based, fully labeled, and mechanistically evaluated.

The physics-like world is used to make local consequences more meaningful, not to replace the project's core structure.

---

## 14. Risks And Non-Goals

### Risks

- the simulator becomes too complex and distracts from the modeling question
- contact dynamics become chaotic too early
- observations expose the event too directly
- continuous losses hide poor proposal quality
- changed-region thresholds become arbitrary
- trivial local baselines saturate the benchmark
- global rollout error dominates local rewrite evaluation
- rendered variants pull the project back into perception-first difficulty

### Non-Goals

Step33 is not:

- CLEVRER adoption
- real-world data learning
- raw video understanding
- raw image understanding
- hypergraph modeling
- LLM integration
- backend transfer
- parked Step22-31 reopening
- large training before benchmark validation

---

## 15. Final Decision Recommendation

Step33 should be pursued next as a synthetic benchmark design phase.

The first concrete action should be to write and review a compact Step33 data/eval specification before implementation. That specification should pin down:

- simulator equations
- graph schema
- event families
- observation variants
- label definitions
- changed-region thresholds
- oracle baselines
- trivial baselines
- first smoke metrics

The first actual experiment should be a tiny structured data/eval probe with oracle and trivial baselines. It should not be a candidate training run.
