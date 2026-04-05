# Relational / Event-Centric World Model

Stage 1 synthetic graph-event world model project.

## 1. Project Goal

This repository studies a narrow but important question:

> In a structured synthetic graph environment with local events, can local event-centric modeling outperform a plain global graph transition model?

The project is intentionally staged.  
Current work is limited to **Stage 1**:

- synthetic graph environment
- structured graph state input
- no raw images
- no LLM integration
- no real-world data
- no hypergraph rewrite complexity yet

The goal of Stage 1 is to test the local-event hypothesis in the cleanest possible setting before adding learned proposal or causal-consistency machinery.

---

## 2. Current Stage-1 Dataset

The synthetic graph-event dataset is already implemented and working.

### Event types
1. `node_state_update`
2. `edge_add`
3. `edge_delete`
4. `motif_type_flip`

### Dataset properties
- train / val / test splits
- variable-size graphs
- padded batching
- `node_mask`
- one-event and two-event transitions
- independent two-event pairs
- `changed_nodes`
- `changed_edges`
- `event_scope_union_nodes`
- `event_scope_union_edges`

### Important distinction
The dataset stores both:

- **changed region**: the nodes / edges that actually changed
- **event scope**: the local region associated with the event, which may include context beyond the strictly changed elements

This distinction is important for oracle-local experiments.

---

## 3. What Has Been Completed

### 3.1 Global transition baselines
Implemented and trained:

- global whole-graph baseline
- typed global baseline with:
  - node type classification head
  - node continuous state regression head
  - edge prediction head

Purpose:
- verify learnability of the synthetic task
- establish a clean Stage-1 reference
- separate node type / node state / edge difficulty

### 3.2 Oracle local rewrite baseline
Implemented and trained:

- oracle local rewrite model using ground-truth event scope
- scope-only losses
- merge-back into full graph output
- full-graph and scope-only evaluation

### 3.3 Oracle merge-back sanity check
Implemented:
- exact oracle reconstruction sanity check using `event_scope_union_*`

Result:
- oracle scope + merge-back perfectly reconstruct the next graph
- all changed nodes / edges are covered by oracle scope

This confirms that the oracle-local experiment is well-defined.

### 3.4 Delta edge formulation
Implemented and tested:
- edge prediction as 3-class delta:
  - `keep`
  - `add`
  - `delete`

Also implemented:
- breakdown evaluation by event type
- breakdown evaluation by one-event / two-event setting
- changed-vs-context edge analysis
- early stopping and balanced model selection for delta training

---

## 4. Main Current Findings

### 4.1 Local rewrite is learnable
With oracle event scope provided, local rewrite dynamics can be learned.

### 4.2 Edge prediction is the main difficulty
Node type and node state are comparatively stable.  
Edge prediction is the main bottleneck.

### 4.3 The critical failure mode was edge deletion
Initial oracle-local edge results showed:
- `edge_add` was easy
- `edge_delete` almost collapsed

### 4.4 Delta edge modeling + delete-aware weighting helps
The most useful current edge formulation is:

- 3-class edge delta prediction: `keep / add / delete`
- delete-aware loss weighting
- balanced checkpoint selection based on scope add/delete accuracy
- early stopping

Current best delta setup:
- `delta_delete_weight = 3.0`
- early stopping enabled
- best checkpoint selected by balanced scope add/delete score

### 4.5 Current interpretation
This does **not** yet prove that local event-centric modeling is superior to the typed global baseline in a fully fair end-to-end sense, because oracle-local models are given privileged event scope and copy outside-scope structure.

What it **does** show is:

- local rewrite is viable in principle
- delete-aware edge modeling matters
- event-centric dynamics have meaningful signal worth pursuing further

---

## 5. Current Repository Structure

```text
relational-event-world-model/
├── data/
│   ├── generate_graph_event_data.py
│   ├── dataset.py
│   ├── collate.py
│   ├── graph_event_train.pkl
│   ├── graph_event_val.pkl
│   └── graph_event_test.pkl
│
├── models/
│   ├── baselines.py
│   ├── oracle_local.py
│   └── oracle_local_delta.py
│
├── train/
│   ├── train_baseline_global.py
│   ├── oracle_sanity_check.py
│   ├── train_oracle_local.py
│   ├── eval_oracle_local_breakdown.py
│   ├── train_oracle_local_delta.py
│   ├── train_oracle_local_delta_earlystop.py
│   └── eval_oracle_local_breakdown_delta.py
│
├── checkpoints/
│   ├── global_baseline/
│   ├── global_baseline_typed/
│   ├── oracle_local_rewrite_typed/
│   └── oracle_local_rewrite_delta/
│
├── .gitignore
├── readme.md
└── ...
```

---

## 6. File Guide

### `data/generate_graph_event_data.py`
Synthetic graph-event data generator.

### `data/dataset.py`
Dataset loader for pickled graph-event samples.

### `data/collate.py`
Padding-aware collate function for variable-size graphs.

### `models/oracle_local.py`
Oracle local rewrite baseline with local typed prediction and merge-back.

### `models/oracle_local_delta.py`
Oracle local rewrite baseline with edge delta prediction (`keep/add/delete`).

### `train/oracle_sanity_check.py`
Checks that oracle event scope plus merge-back exactly reconstructs the next graph.

### `train/train_baseline_global.py`
Trains the global typed baseline.

### `train/train_oracle_local.py`
Trains the oracle local typed baseline.

### `train/eval_oracle_local_breakdown.py`
Breakdown evaluation for oracle local typed model.

### `train/train_oracle_local_delta.py`
Delta-edge oracle local training script.

### `train/train_oracle_local_delta_earlystop.py`
Delta-edge oracle local training with:
- early stopping
- balanced add/delete checkpoint selection

### `train/eval_oracle_local_breakdown_delta.py`
Breakdown evaluation for delta-edge oracle local model.

---

## 7. Recommended Entry Points

### Generate / inspect data
```bash
python data/generate_graph_event_data.py
```

### Train typed global baseline
```bash
python train/train_baseline_global.py \
  --train_path data/graph_event_train.pkl \
  --val_path data/graph_event_val.pkl
```

### Oracle merge-back sanity check
```bash
python train/oracle_sanity_check.py \
  --data_path data/graph_event_val.pkl
```

### Train oracle local typed baseline
```bash
python train/train_oracle_local.py \
  --train_path data/graph_event_train.pkl \
  --val_path data/graph_event_val.pkl
```

### Train delta oracle local baseline
```bash
python train/train_oracle_local_delta_earlystop.py \
  --train_path data/graph_event_train.pkl \
  --val_path data/graph_event_val.pkl \
  --delta_delete_weight 3.0
```

### Evaluate delta oracle local breakdown
```bash
python train/eval_oracle_local_breakdown_delta.py \
  --checkpoint_path checkpoints/oracle_local_rewrite_delta/best.pt \
  --data_path data/graph_event_test.pkl \
  --split_name test
```

---

## 8. Current Best Experimental Direction

At the current stopping point, the most useful oracle-local edge configuration is:

- **edge mode**: delta 3-class (`keep/add/delete`)
- **delete weighting**: `3.0`
- **selection metric**: balanced scope add/delete accuracy
- **training control**: early stopping

This is the current working delta baseline, not the final project endpoint.

---

## 9. What Is Not Done Yet

Not yet implemented:

- learned event proposal
- proposal-vs-rewrite disentangling beyond oracle scope
- causal consistency / order-invariance losses
- long-horizon rollout studies
- real-world data
- image-based input
- hypergraph rewrite formulation

---

## 10. Suggested Next Research Steps

Likely next steps after the current checkpoint:

1. stabilize keep/context edge performance without losing delete
2. study `motif_type_flip` node-type difficulty more directly
3. compare final oracle-local vs typed-global results in a clean summary table
4. move from oracle scope toward constrained learned proposal
5. later add causal-consistency regularization for independent events

---

## 11. Notes on Git Hygiene

Recommended:
- do **not** commit large checkpoint files unless necessary
- keep dataset pickles out of version control if they are reproducible
- keep `__pycache__/` out of version control
- use `.gitignore` for checkpoints, caches, and local artifacts

Typical ignores:
```gitignore
__pycache__/
*.pyc
checkpoints/
data/*.pkl
.DS_Store
```

If dataset pickles are intentionally part of the repo, remove `data/*.pkl` from the ignore rule.
