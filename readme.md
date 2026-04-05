# NewProgram  
**Working project title:** Relational / Event-Centric World Model  
**Current status:** Stage 1 design and setup  
**Main goal:** Build a small, controlled experimental framework to test whether **local event proposal + local rewrite dynamics + delayed causal-consistency regularization** can outperform plain global graph dynamics in structured synthetic environments.

---

# 1. Project Overview

This project explores a new direction for world models inspired by three major ideas:

1. **Relational state representation**  
   Instead of representing the world as a single dense latent vector, represent it as a structured graph of objects and relations.

2. **Event-centric local dynamics**  
   Instead of learning only a global transition from one state to the next, represent dynamics as a small set of local events that rewrite local parts of the graph.

3. **Causal consistency / order invariance**  
   If two local events are independent, then applying them in different orders should ideally produce equivalent or near-equivalent final world states.

The long-term motivation comes from combining:
- modern world model research,
- graph-based / object-centric dynamics,
- and ideas inspired by Stephen Wolfram’s local rewrite / causal invariance perspective.

However, this project does **not** begin from a large-scale AI system.  
It begins from a **small, synthetic, fully controlled graph event environment**, so that the core modeling hypothesis can be tested clearly.

---

# 2. Core Research Question

The central question of this project is:

> Can a relational, event-centric world model learn local mechanisms more robustly than a plain global transition model in structured environments?

More specifically:

- Can the model identify which local regions of a graph are worth updating?
- Can a small number of local events explain overall state transitions?
- Can local rewrite dynamics improve long-horizon rollout stability?
- Can a weak causal-consistency loss improve robustness when independent events are reordered?

---

# 3. Current Project Scope

This project is currently only focused on **Stage 1**:

- synthetic graph environment
- structured graph state input
- no raw images
- no LLM integration yet
- no real-world data yet
- no hypergraph complexity yet
- no large model training yet

The purpose of Stage 1 is to answer a very narrow but foundational question:

> In a small graph world with local events, is learned local event structure more useful than plain global graph dynamics?

---

# 4. Current Folder Structure

The project currently lives under:

`D:\newprogram`

Planned internal structure:

```text
newprogram/
│
├── data/
│   └── generate_graph_event_data.py
│
├── models/
│   ├── encoder.py
│   ├── proposal.py
│   ├── local_rewrite.py
│   └── baselines.py
│
├── train/
│   ├── train_baseline_global.py
│   ├── train_oracle_local.py
│   └── train_learned_proposal.py
│
├── eval/
│   ├── eval_prediction.py
│   ├── eval_proposal.py
│   └── eval_consistency.py
│
└── README.md

# 5. Meaning of Each Folder

## `data/`
This folder will contain code for generating synthetic datasets.

Main responsibilities:
- generate graph states
- generate local events
- apply event rules to produce next states
- save state transitions
- optionally save ground-truth event metadata for evaluation

Important note:
- Training may not always use event labels,
- but synthetic generation should preserve them for analysis, debugging, and optional weak supervision.

---

## `models/`
This folder contains model definitions.

Main modules planned:

### `encoder.py`
Encodes graph state into node / edge embeddings.

Expected functionality:
- accept graph state \(G_t\)
- produce latent node and/or edge representations
- serve as the shared front-end for all baselines and main models

Likely implementation:
- simple GNN / message passing network
- lightweight first version only

---

### `proposal.py`
Implements the Event Proposal Module.

Expected functionality:
- score candidate local regions
- select top-k candidate event regions
- support restricted search space:
  - single node
  - single edge
  - 1-hop ego subgraph

Important:
- this module is one of the most difficult and fragile parts of the project
- in early experiments, proposal must be strongly constrained
- free-form arbitrary subgraph discovery is **not** the initial goal

---

### `local_rewrite.py`
Implements local event updates.

Expected functionality:
- take a selected local region
- predict how that region changes
- update node features and/or edge structure locally
- merge local rewrites back into the full graph

Important:
- local rewrite should remain truly local
- this module should not silently become a disguised global transition network

---

### `baselines.py`
Contains baseline models.

Planned baselines:

1. **Global transition baseline**
   - takes whole graph input
   - predicts next whole graph directly

2. **Oracle local rewrite baseline**
   - uses ground-truth event regions
   - tests upper bound of local rewrite idea when proposal is perfect

3. optionally later:
   - graph dynamics without explicit event proposal
   - sparse graph update baseline

---

## `train/`
Training entry points.

Expected scripts:

### `train_baseline_global.py`
Train the plain graph transition baseline.

Purpose:
- verify that the synthetic task is learnable at all
- establish a minimum performance reference

---

### `train_oracle_local.py`
Train the local rewrite model with oracle event regions.

Purpose:
- test whether local rewrite is effective when region selection is correct
- measure upper bound of event-centric approach before proposal learning

---

### `train_learned_proposal.py`
Train the learned proposal + local rewrite model.

Purpose:
- test whether event proposal can approximate oracle region selection
- this is the main Stage 1 model

Later:
- this script may eventually include phased training:
  - first without consistency
  - then with weak consistency regularization

---

## `eval/`
Evaluation scripts.

Expected scripts:

### `eval_prediction.py`
Evaluate:
- one-step prediction accuracy
- multi-step rollout error

---

### `eval_proposal.py`
Evaluate proposal quality using synthetic ground truth.

Metrics may include:
- top-k hit rate
- IoU-like region overlap
- precision / recall over changed nodes / edges
- sparsity of selected regions

---

### `eval_consistency.py`
Evaluate independent event reordering consistency.

Purpose:
- compare final graph states under different application orders of independent events
- this is central for the future causal-consistency part

---

# 6. Current Stage: Stage 1

## Stage 1 Objective
Build a minimal synthetic graph world where:
- only a few local events happen per step,
- some events are independent,
- event regions are well-defined,
- model performance can be analyzed clearly.

The main deliverable of Stage 1 is **not** a publishable large system.

The main deliverable is:
- a clean experimental sandbox,
- a working baseline,
- an oracle local model,
- and an initial learned proposal model.

---

# 7. Stage 1 Dataset Design

## 7.1 State Format
Each graph state \(G_t\) should contain:

- a set of nodes
- a sparse set of edges
- node features
- possibly edge features later

Initial default setting:
- 8 to 12 nodes per graph
- sparse random graph connectivity
- node type or categorical label
- low-dimensional continuous node state

Possible representation:
- adjacency matrix
- edge list
- node feature matrix

---

## 7.2 Event Types
The environment should contain only a **small number of local event types**.

Initial target: 3 to 5 event types only.

Examples:
- node state update
- edge add
- edge delete
- small neighborhood rewrite
- motif-triggered type change

Important design rule:
- each event must affect only a small local subgraph

---

## 7.3 Transition Generation
Each step should apply:
- 1 event, or
- 2 events

At least some of the 2-event cases should be **independent**:
- non-overlapping event scopes
- reorderable in principle

This creates the raw material for future consistency experiments.

---

## 7.4 Why Synthetic Data First
Synthetic data is necessary because it allows us to know:
- what changed
- where it changed
- what the true event scopes were
- whether two events were independent

This is essential for debugging the proposal module.

---

# 8. Main Model Plan for Stage 1

## 8.1 Encoder
A lightweight graph encoder.

Likely first version:
- 2 to 3 message-passing layers
- hidden dimension 64 or 128
- simple node and edge embedding updates

Goal:
- produce representations good enough for proposal and rewrite

---

## 8.2 Event Proposal Module
This module chooses which local regions are “worth rewriting.”

### Important current design decision:
Do **not** allow arbitrary subgraph search in Stage 1.

Candidate regions should be restricted to:
- single nodes
- single edges
- 1-hop ego neighborhoods

Then:
- score each candidate
- select top-k candidates

Initial recommended setting:
- k = 1 or 2

### Why restricted proposal?
Because unrestricted subgraph discovery is combinatorially hard and unstable.
Stage 1 must remain controlled.

---

## 8.3 Local Rewrite Module
Given a selected region:
- predict local node updates
- predict local edge changes
- apply changes back to graph

This module is supposed to learn:
- local mechanisms
- not full-graph memorization

---

## 8.4 Merge Strategy
After local rewrites:
- merge predicted local updates into global graph state

Early-stage simplification:
- avoid overlapping events in main training data, or
- use deterministic merge order

Later stages may support more complex merge logic.

---

# 9. Baseline Plan

Stage 1 requires at least three systems:

## Baseline A: Global Transition Model
Whole graph in, next graph out.

Purpose:
- simplest structured baseline
- shows what plain graph dynamics can do

---

## Baseline B: Oracle Local Rewrite
Uses true event region labels.

Purpose:
- measures upper bound of local rewrite if proposal is correct
- helps isolate whether proposal is the bottleneck

---

## Baseline C: Learned Proposal + Local Rewrite
Main experimental model.

Purpose:
- test whether event proposal can approximate useful local mechanisms

---

# 10. Training Strategy

## Phase 1
Train the baseline global transition model first.

Reason:
- verify task is learnable
- establish reference performance

---

## Phase 2
Train oracle local rewrite.

Reason:
- test whether local rewrite is useful in principle
- if oracle local model is not competitive, proposal is not the main issue

---

## Phase 3
Train learned proposal + local rewrite.

Initial loss:
- state prediction loss
- sparsity regularization
- locality regularization

### Important:
Do **not** add consistency loss immediately.

Reason:
- proposal learning is fragile early on
- consistency loss too early may destabilize training

---

## Phase 4
After proposal becomes reasonably stable:
- add weak consistency loss
- only for high-confidence independent event pairs
- start with small weight

---

# 11. Main Loss Components

## 11.1 Prediction Loss
Measures mismatch between predicted next state and ground-truth next state.

Includes:
- node feature loss
- edge structure loss

---

## 11.2 Sparsity Loss
Encourages the model to select only a small number of regions / events.

Purpose:
- prevent degeneration into “rewrite everything”

---

## 11.3 Locality Loss
Encourages small update scope.

Purpose:
- make event proposal correspond to local changes
- prevent proposal from becoming a hidden global updater

---

## 11.4 Consistency Loss (Later Only)
For two independent events \(e_a, e_b\):

- apply \(e_a\) then \(e_b\)
- apply \(e_b\) then \(e_a\)

Encourage similar final graph states.

This is **not** part of the earliest training stage.

---

# 12. Evaluation Plan

## 12.1 Prediction Metrics
- one-step prediction accuracy
- multi-step rollout error

---

## 12.2 Proposal Metrics
Since synthetic data has true event regions, we can evaluate:
- top-k hit rate
- region overlap
- precision / recall on changed nodes / edges

---

## 12.3 Structural Metrics
- average number of proposed events
- average region size
- whether proposal collapses into global rewrite behavior

---

## 12.4 Consistency Metrics
For known independent events:
- compare final states under reordering
- measure latent or graph-state discrepancy

This becomes more important in later Stage 1 or early Stage 2.

---

# 13. What Has Been Conceptually Decided So Far

The following high-level decisions have already been made in discussion:

1. The project should begin with **synthetic graph data**, not raw images.
2. The project should test **local event structure**, not giant end-to-end world models.
3. Event proposal should be **restricted and controlled**, not unconstrained.
4. Proposal learning should **not** depend on immediate strong consistency regularization.
5. Consistency should be added **later**, after the model can already explain transitions reasonably.
6. Oracle local rewrite is necessary as a diagnostic baseline.
7. The project should be built in small stages so that failure modes are interpretable.

---

# 14. Main Open Problems Right Now

These are the most important unresolved design problems at the current stage:

## 14.1 Dataset Rule Design
Need to finalize:
- exact graph size
- exact node feature format
- exact event rules
- exact independence definition

---

## 14.2 Graph Representation Format
Need to choose:
- adjacency matrix vs edge list
- whether edge features are needed from the beginning
- save format for training data

---

## 14.3 Proposal Parameterization
Need to decide:
- hard top-k vs soft mask
- candidate region scoring function
- whether training uses weak supervision at the beginning

---

## 14.4 Local Rewrite Parameterization
Need to decide:
- whether rewrite predicts node updates only first
- whether edge changes are predicted jointly or separately
- how to merge multiple local updates back into graph state

---

## 14.5 Training Stabilization
Need to decide:
- whether to warm-start proposal with state-difference signals
- whether to use synthetic event labels for pretraining
- when exactly to introduce consistency regularization

---

# 15. Recommended Immediate Next Steps

These are the next concrete tasks to complete.

## Step 1: Write dataset generator
Priority: highest

Need:
- graph initialization
- event rule engine
- transition generation
- save samples with optional event metadata

This is the most important next task.

---

## Step 2: Build global baseline
Need:
- graph encoder
- next-state predictor
- basic training loop

This confirms the task is learnable.

---

## Step 3: Build oracle local rewrite baseline
Need:
- local region input
- local rewrite network
- merge logic

This tests the upper bound of local rewriting.

---

## Step 4: Build proposal module
Need:
- candidate region construction
- candidate scoring
- top-k selection

This is the first main “research” component.

---

## Step 5: Add evaluation scripts
Especially:
- proposal quality
- rollout stability
- reorder consistency

---

# 16. Longer-Term Future Directions

These are **not** current priorities, but are likely future stages if Stage 1 works.

## Stage 2
More difficult structured environments:
- richer event types
- overlapping events
- more complex graph dynamics
- softer independence assumptions

---

## Stage 3
Object-centric perception:
- derive graph state from images
- use slot-based or object-centric encoders
- bridge from pixels to structured state

---

## Stage 4
Causal dependency learning:
- explicit event dependency graph
- independent vs dependent event prediction
- stronger order-consistency objectives

---

## Stage 5
Toward larger world models / LLM integration:
- structured memory
- event-centric planning
- graph-augmented language reasoning
- explicit world-state updates for LLM agents

---

# 17. Intended Communication Use of This README

This README is meant to serve as a persistent project memory document.

Its purpose is:
- to summarize the current project logic,
- to preserve the design decisions already made,
- and to allow rapid re-entry into the project in a new LLM conversation if context is lost.

When opening a new LLM chat, the user should be able to paste this README and quickly re-establish:

- what the project is,
- what stage it is in,
- what files are planned,
- what has already been decided,
- what remains open,
- and what the next steps are.

---

# 18. Short Summary for Quick Reuse

If a new chat window needs only a short context summary, use this:

> This project is building a small synthetic graph-based world model to test whether local event proposal + local rewrite dynamics + later causal-consistency regularization can outperform plain global graph dynamics.  
> Current stage is Stage 1 only: structured graph input, synthetic local event dataset, no images, no LLM integration yet.  
> Planned folders are `data`, `models`, `train`, `eval`.  
> Main baselines are global transition, oracle local rewrite, and learned proposal + local rewrite.  
> Immediate next step is writing the dataset generator and getting the global baseline running.

---

# 19. Naming Note

The folder is currently called `newprogram`, but the actual project name is still undecided.

Possible future project names:
- EventGraph World Model
- Relational Rewrite World Model
- Local Event Dynamics
- Graph Event World Model
- Causal Rewrite WM

For now, keep the generic name until the first version is working.

---

# 20. Final Principle

The project must remain scientifically interpretable.

That means:
- start small,
- avoid unnecessary complexity,
- isolate one hypothesis at a time,
- and always compare against strong simple baselines.

The first real success criterion is **not** building a huge model.

It is demonstrating, in a fully controlled graph world, that:
- learned local event structure is useful,
- proposal can identify meaningful local regions,
- and local dynamics can be more mechanism-like than plain global transitions.



# Current Progress Update

## Stage 1 Modeling Status

The Stage 1 synthetic graph-event pipeline has now progressed from **dataset construction** to **working baseline modeling**. The dataset is no longer only a designed environment; it is now being used to train and evaluate actual predictive models. This marks the first concrete modeling phase of the project. :contentReference[oaicite:0]{index=0}

---

## What Has Been Completed

### 1. Dataset loading and batching pipeline
A full dataset pipeline has been implemented for the Stage 1 graph-event data.

This includes:
- loading serialized train / validation splits,
- converting each sample into model-ready tensors,
- handling variable-size graphs through padding,
- constructing a `node_mask` so that padded nodes do not corrupt the loss,
- and preserving event-related metadata for later proposal and consistency evaluation.

This step was necessary because the Stage 1 dataset contains graphs with variable numbers of nodes, so naive fixed-size batching was not sufficient.

### 2. Initial global transition baseline
A full-graph transition baseline was implemented and trained.

This model takes the entire graph state at time \(t\) and predicts the next full graph state at time \(t+1\), including:
- next-step node features,
- and next-step adjacency structure.

The purpose of this baseline was not to solve the final research problem, but to answer the first necessary question:

> Is the Stage 1 synthetic transition task learnable at all?

The answer is now clearly yes. Training and validation losses decreased substantially, and edge prediction accuracy reached a strong level on the validation set. The original global baseline achieved its best validation loss at approximately **Epoch 13**, confirming that the task is learnable with a relatively lightweight graph model. :contentReference[oaicite:1]{index=1}

### 3. Typed node prediction refinement
After the first global baseline worked, the node prediction mechanism was refined.

Originally, the model treated the full node feature vector as one continuous regression target. However, inspection of the synthetic dataset showed that node features are actually mixed-structure:
- the first channel is a **discrete node type**,
- while the remaining channels are **continuous state variables**.

Because of this, the node head was redesigned into two parts:
- a **type head** for node-type classification,
- and a **state head** for continuous-state regression.

The loss was updated accordingly:
- cross-entropy for node type,
- regression loss for continuous state,
- and masked binary cross-entropy for edges.

This change was important not mainly because it guaranteed a lower scalar total loss, but because it made the model align with the real semantic structure of the data.

### 4. Typed global baseline results
The refined typed global baseline was successfully trained.

Its validation performance stabilized around:
- **validation type accuracy** of about `0.97`,
- **validation state MAE** of about `0.057–0.061`,
- **validation edge accuracy** of about `0.972–0.974`.

The best validation loss for this typed baseline was **0.208255**, achieved at approximately **Epoch 11**. This indicates that the refined baseline is stable, interpretable, and strong enough to serve as the main Stage 1 global reference model. :contentReference[oaicite:2]{index=2}

---

## Why This Was Done

These steps were done for a very specific scientific reason.

The project’s core hypothesis is not simply that a graph neural network can predict the next state of a graph. The real hypothesis is more specific:

> In a structured environment with local events, explicitly modeling local event regions and local rewrite mechanisms may be more useful than using a plain whole-graph transition model.

Before testing that hypothesis, the project first needed a strong and interpretable **global baseline**.

This was necessary for three reasons:

1. **To verify task learnability**  
   If even the global baseline could not learn the synthetic transition task, then later failures of local rewrite models would be uninterpretable.

2. **To establish a reference point**  
   Any future oracle local model or learned proposal model must be compared against a real baseline, not just against intuition.

3. **To identify the true difficulty of the data**  
   The refined typed baseline helped separate three different prediction subproblems:
   - node type prediction,
   - continuous state prediction,
   - edge prediction.

This decomposition is useful because it reveals which parts of the task are easy and which parts remain genuinely challenging.

---

## What Has Been Learned So Far

The current experiments support the following conclusions:

1. **The Stage 1 synthetic graph-event task is learnable.**  
   The baseline models train stably and achieve strong validation performance. :contentReference[oaicite:3]{index=3}

2. **Whole-graph prediction is already quite strong.**  
   In particular, edge prediction is learned very effectively by the global baseline. :contentReference[oaicite:4]{index=4}

3. **Node prediction is more interpretable when separated into type and state.**  
   Treating discrete node type and continuous node state as different targets produces cleaner metrics and a more faithful model design.

4. **The project now has a proper baseline for comparison.**  
   This is a major milestone, because later local event models can now be evaluated against a concrete standard rather than a vague expectation.

---

## Current Project Status Assessment

At this point, the Stage 1 project status can be summarized as:

> **Dataset complete, global baseline complete, typed global baseline complete.**

This means the project has now successfully completed:
- synthetic graph-event dataset generation,
- dataset loading and padded batching,
- mask-aware training for variable-size graphs,
- an initial global transition baseline,
- and a refined typed global baseline aligned with dataset semantics.

The project is no longer blocked on infrastructure.  
It is now ready to move to the next real research comparison.

---

## Recommended Next Step

The next immediate modeling step should be:

### Build the Oracle Local Rewrite baseline

Purpose:
- use ground-truth event regions,
- test whether local rewriting is effective when proposal is perfect,
- determine whether local event structure provides an advantage over the global baseline when the region-selection problem is removed.

This is the most important next step because it directly tests the central Stage 1 research question:

> Is local event-centric modeling useful in principle, before proposal learning is introduced?

If the oracle local rewrite model is not competitive, then the problem is not proposal learning; the local rewrite idea itself would need reconsideration.  
If the oracle model is competitive or better, then proposal learning becomes the next meaningful bottleneck.

---

## Short Summary for Future Re-entry

If a future chat only needs the most important current context, use this:

> The Stage 1 synthetic graph-event dataset is complete and already supports variable-size batching with padding and node masks.  
> A global graph transition baseline has been successfully implemented and trained, confirming that the synthetic transition task is learnable.  
> The node prediction head was then refined into a typed formulation: node type is predicted with classification, while continuous node state is predicted with regression.  
> The typed global baseline is now the main Stage 1 reference model, with validation type accuracy around 97%, validation state MAE around 0.06, and validation edge accuracy around 97.3–97.4%.  
> The next step is to build the Oracle Local Rewrite baseline and compare it against the global baseline.