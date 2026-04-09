# Relational / Event-Centric World Model

Stage 1 synthetic graph-event world model project.

## 1. Project Goal

This repository studies a narrow but important question:

> In a structured synthetic graph environment with local events, can local event-centric modeling outperform a plain global graph transition model?

The project is intentionally staged.
Current work is still limited to **Stage 1**:

- synthetic graph environment
- structured graph state input
- no raw images
- no LLM integration
- no real-world data
- no hypergraph rewrite complexity

The purpose of Stage 1 is to test the local-event hypothesis in the cleanest possible setting before adding learned proposal, causal-consistency regularization, or richer inputs.

---

## 2. Current Stage-1 Scope and Dataset

The synthetic graph-event dataset is implemented and working.

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

- **changed region**: nodes / edges that actually changed
- **event scope**: the local region associated with the event, which may include context beyond the strictly changed elements

This distinction remains central to the oracle-local experiments.

---

## 3. What Has Been Implemented

### 3.1 Global baselines
Implemented and trained:
- global whole-graph baseline
- typed global baseline with separate heads for:
  - node type classification
  - node continuous state regression
  - edge prediction

### 3.2 Oracle local typed rewrite baseline
Implemented and trained:
- oracle local rewrite model using ground-truth event scope
- scope-only losses
- merge-back into full-graph prediction
- full-graph and scope-only evaluation

### 3.3 Oracle merge-back sanity check
Implemented and passed:
- exact oracle reconstruction sanity check using `event_scope_union_*`
- confirms that oracle scope plus merge-back can exactly reconstruct the next graph
- confirms that all changed nodes / edges are covered by oracle scope

### 3.4 Oracle local delta-edge baseline
Implemented and trained:
- edge prediction as 3-class delta:
  - `keep`
  - `add`
  - `delete`

Also implemented:
- breakdown evaluation by event type
- breakdown evaluation by one-event / two-event setting
- changed-vs-context edge analysis
- early stopping
- checkpoint selection by validation metrics

---

## 4. Current Scientific Interpretation

### 4.1 Local rewrite is learnable under oracle scope
With oracle event scope provided, local rewrite dynamics can be learned.

### 4.2 Edge prediction is the main bottleneck
Node type and node state are comparatively stable.
Edge prediction remains the main source of difficulty.

### 4.3 The original failure mode was delete collapse
In earlier oracle-local edge experiments:
- `edge_add` was easy
- `edge_delete` nearly collapsed

### 4.4 Delta edges plus delete-aware weighting fixed the worst failure mode
The key improvement was:
- delta edge modeling (`keep / add / delete`)
- delete-aware loss weighting
- early stopping
- balanced validation selection over add/delete

This established a usable oracle-local delta baseline.

### 4.5 The project is now in a different regime
The main question is no longer:

> Can delete be learned at all?

The main question is now:

> Can keep/context be improved without sacrificing delete?

Recent ablations show a real tension between:
- **keep/context stability**
- **delete/changed-edge sensitivity**

That tradeoff is now the central Stage-1 problem.

---

## 5. Recent Oracle-Local Delta Ablations

All rows below are **test-set overall breakdown metrics** from `eval_oracle_local_breakdown_delta.py`.

| Variant | scope_edge | delta_all | keep | add | delete | changed | context | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Baseline: `delete_weight=3.0`, no keep upweight, add/delete checkpoint selection | 0.7179 | 0.7039 | 0.7125 | 0.9179 | 0.4708 | 0.6826 | 0.7374 | Strong reference baseline; still delete-aware but keep/context not yet optimized |
| Global `dropout=0.1` | 0.7306 | 0.7164 | 0.7501 | 0.9091 | 0.4153 | 0.6441 | 0.7786 | Better keep/context, but delete falls |
| Edge-head-only `edge_dropout=0.1` | 0.7217 | 0.7161 | 0.7426 | 0.8900 | 0.4583 | 0.6583 | 0.7568 | More targeted than global dropout, but still not a clear win |
| Edge-head-only `edge_dropout=0.05` | 0.6902 | 0.6800 | 0.6556 | 0.9150 | 0.5431 | 0.7161 | 0.6758 | Pushes toward delete/changed, hurts keep/context too much |
| 3-way checkpoint selection (`keep+add+delete`) | 0.6668 | 0.6515 | 0.5904 | 0.9296 | 0.6028 | 0.7525 | 0.6192 | Strongly delete/changed-oriented; clearly too aggressive for current goal |
| `delta_keep_weight=1.25` | 0.7410 | 0.7253 | 0.7639 | 0.9399 | 0.3861 | 0.6455 | 0.7940 | Strongest keep/context-oriented setting, but delete drops too much |
| **`delta_keep_weight=1.10`** | **0.7362** | **0.7207** | **0.7426** | **0.9370** | **0.4389** | **0.6719** | **0.7718** | **Current best balance candidate** |

### 5.1 Practical reading of the table

- **Dropout-only ablations** did not solve the target problem cleanly.
  They changed the tradeoff, but did not produce a robust keep/context improvement at acceptable delete cost.
- **3-way checkpoint selection** over-amplified the delete/changed side of the tradeoff.
- **Keep-weighting is the first ablation family that directly moved the system in the intended direction.**
- `delta_keep_weight=1.25` improved keep/context substantially but overpaid in delete.
- **`delta_keep_weight=1.10` is the current lead candidate because it improves keep/context and overall edge quality relative to the original baseline, while only modestly reducing delete.**

### 5.2 Current recommendation

Use the following as the **current main oracle-local delta candidate**:

- edge mode: delta 3-class (`keep/add/delete`)
- `delta_keep_weight = 1.10`
- `delta_add_weight = 1.0`
- `delta_delete_weight = 3.0`
- no model dropout
- no edge-head dropout
- early stopping enabled
- checkpoint selection kept on the original add/delete validation criterion

Keep the original `delete_weight=3.0` baseline as the primary reference point.

---

## 6. Current Repository Structure

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
│   ├── oracle_local_rewrite_delta/
│   └── oracle_local_rewrite_delta_*/
│
├── .gitignore
├── README.md
└── ...
```

---

## 7. File Guide

### `data/generate_graph_event_data.py`
Synthetic graph-event data generator.

### `data/dataset.py`
Dataset loader for pickled graph-event samples.

### `data/collate.py`
Padding-aware collate function for variable-size graphs.

### `models/oracle_local.py`
Oracle local typed rewrite baseline with merge-back.

### `models/oracle_local_delta.py`
Oracle local delta-edge rewrite baseline.
Current work uses the `keep / add / delete` edge formulation.

### `train/oracle_sanity_check.py`
Verifies that oracle event scope plus merge-back can exactly reconstruct the next graph.

### `train/train_baseline_global.py`
Training script for the typed global baseline.

### `train/train_oracle_local.py`
Training script for the oracle local typed baseline.

### `train/eval_oracle_local_breakdown.py`
Breakdown evaluation for the oracle local typed model.

### `train/train_oracle_local_delta.py`
Basic training script for the oracle local delta-edge model.

### `train/train_oracle_local_delta_earlystop.py`
Main Stage-1 delta-edge training script with:
- early stopping
- class-weighted delta loss (`keep/add/delete`)
- validation-based checkpoint selection

### `train/eval_oracle_local_breakdown_delta.py`
Main breakdown evaluation script for oracle local delta experiments.

---

## 8. Recommended Entry Points

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

### Train the current lead delta candidate
```bash
python train/train_oracle_local_delta_earlystop.py \
  --train_path data/graph_event_train.pkl \
  --val_path data/graph_event_val.pkl \
  --dropout 0.0 \
  --edge_dropout 0.0 \
  --delta_keep_weight 1.10 \
  --delta_add_weight 1.0 \
  --delta_delete_weight 3.0 \
  --selection_keep_weight 0.0 \
  --selection_add_weight 0.5 \
  --selection_delete_weight 0.5 \
  --save_dir checkpoints/oracle_local_rewrite_delta_keep110
```

### Evaluate the current lead delta candidate
```bash
python train/eval_oracle_local_breakdown_delta.py \
  --checkpoint_path checkpoints/oracle_local_rewrite_delta_keep110/best.pt \
  --data_path data/graph_event_test.pkl \
  --split_name test
```

### Reproduce the original delete-aware reference baseline
```bash
python train/train_oracle_local_delta_earlystop.py \
  --train_path data/graph_event_train.pkl \
  --val_path data/graph_event_val.pkl \
  --delta_keep_weight 1.0 \
  --delta_add_weight 1.0 \
  --delta_delete_weight 3.0 \
  --selection_keep_weight 0.0 \
  --selection_add_weight 0.5 \
  --selection_delete_weight 0.5 \
  --save_dir checkpoints/oracle_local_rewrite_delta
```

---

## 9. What Not To Do Yet

Still out of scope for the current checkpoint:

- learned event proposal
- proposal-vs-rewrite disentangling beyond oracle scope
- causal consistency / order-invariance losses
- long-horizon rollouts
- real-world data
- image input
- hypergraph rewrite formulation

Do not jump to these until the Stage-1 oracle-local edge tradeoff is better understood.

---

## 10. Suggested Immediate Next Step

If one more **minimal** Stage-1 experiment is allowed, the most informative next run is:

- `delta_keep_weight = 1.05`

Why:
- `1.25` was too far toward keep/context and hurt delete too much
- `1.10` is currently the best balance candidate
- `1.05` is the cleanest single follow-up to test whether an even smaller keep upweight preserves more delete while retaining most of the keep/context gain

Do **not** reopen broad dropout sweeps or broad checkpoint-selection sweeps first.

---

## 11. Notes on Comparison Tables

The typed global baseline and oracle local typed baseline are completed and remain part of the Stage-1 story.
This README update focuses on the oracle-local delta line because that is where the recent ablation work has happened.
When preparing a final paper-style summary, include a clean three-way comparison table for:

- typed global baseline
- oracle local typed baseline
- oracle local delta baseline / current lead candidate

and keep the more detailed delta ablation table separate.

---

## 12. Git Hygiene

Recommended:
- do not commit large checkpoint files unless necessary
- keep reproducible dataset pickles out of version control if possible
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
