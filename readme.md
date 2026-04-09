# Relational / Event-Centric World Model

Stage 1 synthetic graph-event world model project.

## 1. Project Goal

This repository studies a narrow but important question:

> In a structured synthetic graph environment with local events, can local event-centric modeling outperform a plain global graph transition model?

Current work is intentionally limited to **Stage 1**:

- synthetic graph environment
- structured graph state input
- no raw images
- no LLM integration
- no real-world data
- no hypergraph rewrite complexity
- no end-to-end proposal+rewrite coupling yet

The purpose of Stage 1 is to isolate the local-event hypothesis in the cleanest setting before adding learned proposal, causal-consistency objectives, or more realistic domains.

---

## 2. Dataset

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

A key distinction is preserved between:

- **changed region**: nodes / edges that actually changed
- **event scope**: the local region associated with the event, which may include extra context

---

## 3. What Has Been Implemented

### Global baselines
- global whole-graph baseline
- typed global baseline

### Oracle-local baselines
- oracle local typed rewrite baseline
- oracle local delta-edge rewrite baseline
- oracle merge-back sanity check

### Delta-edge tooling
- 3-class edge delta prediction (`keep / add / delete`)
- breakdown evaluation by event type
- changed-vs-context edge analysis
- early stopping and checkpoint selection

### Type-difficulty tooling
- type breakdown evaluation
- flip-target diagnostics for `motif_type_flip`

### Proposal tooling
- node-only scope proposal
- explicit node+edge scope proposal
- proposal-conditioned rewrite bridge evaluation

---

## 4. Stage-1 Main Findings

### 4.1 Oracle-local rewrite is learnable
With oracle event scope provided, local rewrite dynamics are clearly learnable.

### 4.2 Edge prediction is the main bottleneck
Node type and node state are relatively stable. Edge prediction is harder, especially deletion behavior and the balance between changed-edge sensitivity and keep/context preservation.

### 4.3 Delta edge modeling with delete-aware weighting is effective
A 3-class delta edge head with delete-aware weighting substantially improved over earlier local edge behavior.

### 4.4 A mild keep-aware edge loss tilt gives the best current oracle-scope edge balance
Among the tested oracle-scope delta variants, a mild keep-aware weighting (`delta_keep_weight=1.10`, with `delta_delete_weight=3.0`) gave the best overall edge tradeoff.

### 4.5 `motif_type_flip` is a real type-side failure mode
Typed evaluation revealed that the main type difficulty is not generic node classification, but true type-flip cases. The models tend to copy the current type unless flip-target supervision is strengthened.

### 4.6 Flip-aware type supervision helps much more under oracle-local modeling than under global typed modeling
Adding flip-aware type weighting only marginally helped the global typed baseline, but gave a much larger improvement for the oracle-local typed baseline. This suggests locality matters not only for edge rewrite, but also for discrete type-transition ambiguity.

### 4.7 Learned-scope proposal is viable, but node-only proposal is not enough
A node-only proposal can recover useful node-scope signal, but edge scope derived by pairwise node logic is too crude.

### 4.8 Explicit node+edge proposal closes much of the bridge gap
An explicit node+edge proposal model, combined with rewrite-facing threshold selection, produces a learned-scope bridge that is substantially better than the earlier node-only bridge and noticeably closer to oracle-scope rewrite.

### 4.9 Proposal-side flip-aware node weighting improves learned-scope type behavior
Adding a flip-aware node proposal weight improved the learned-scope bridge without materially breaking the edge-side bridge, and is the current best learned-scope bridge candidate.

---

## 5. Current Recommended Reference Points

### 5.1 Oracle-scope edge reference
Current oracle-scope edge lead candidate:

- model: oracle-local delta rewrite
- edge mode: delta 3-class (`keep / add / delete`)
- `delta_keep_weight = 1.10`
- `delta_add_weight = 1.0`
- `delta_delete_weight = 3.0`
- early stopping enabled

### 5.2 Typed reference points

| Model | Key result | Interpretation |
|---|---:|---|
| Global typed baseline | flip ≈ 0.000 | strong current-type copy bias |
| Global typed + flip-aware weighting | flip ≈ 0.018 | only marginal improvement |
| Oracle-local typed baseline | flip ≈ 0.029 | locality alone is not enough |
| Oracle-local typed + `type_flip_weight=1.5` | flip ≈ 0.063 | best balanced typed candidate |
| Oracle-local typed + `type_flip_weight=2.0` | flip ≈ 0.270 | aggressive proof-of-correctability variant |

### 5.3 Oracle-scope delta edge comparison

| Variant | scope_edge | delta_all | keep | add | delete | changed | context | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Original delta baseline | 0.7179 | 0.7039 | 0.7125 | 0.9179 | 0.4708 | 0.6826 | 0.7374 | delete-oriented reference |
| `edge_dropout=0.10` | 0.7217 | 0.7161 | 0.7426 | 0.8900 | 0.4583 | 0.6583 | 0.7568 | mild edge regularization; not a clear new best |
| `edge_dropout=0.05` | 0.6902 | 0.6800 | 0.6556 | 0.9150 | 0.5431 | 0.7161 | 0.6758 | too delete/changed-oriented |
| keep/add/delete selection | 0.6668 | 0.6515 | 0.5904 | 0.9296 | 0.6028 | 0.7525 | 0.6192 | over-pushes delete/changed |
| `delta_keep_weight=1.25` | 0.7410 | 0.7253 | 0.7639 | 0.9399 | 0.3861 | 0.6455 | 0.7940 | strongest keep/context, but delete drops too far |
| **`delta_keep_weight=1.10`** | **0.7362** | **0.7207** | **0.7426** | **0.9370** | **0.4389** | **0.6719** | **0.7718** | **current best overall edge tradeoff** |

### 5.4 Learned-scope bridge comparison

| Bridge variant | scope_edge | delta_all | keep | add | delete | changed | context | Type flip | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Node-only proposal bridge | 0.4514 | 0.3533 | 0.3476 | 0.9157 | 0.3822 | 0.6719 | 0.4479 | low | not viable |
| Explicit node+edge proposal, unweighted, default bridge tuning | 0.6972 | 0.6655 | 0.6727 | 0.8767 | 0.4188 | 0.6719 | 0.7059 | low | first viable bridge |
| Frozen-proposal rewrite training | 0.9596 | 0.9596 | 0.9999 | 0.0000 | 0.0000 | 0.0000 | 0.9999 | 0.0000 | collapses to copy-heavy behavior |
| Union-scope rewrite training | 0.9236 | 0.9208 | 0.9571 | 0.2821 | 0.0306 | 0.4258 | 0.9601 | 0.0000 | more robust than predicted-only training, but still too copy-heavy |
| **Explicit node+edge proposal + `node_flip_weight=2.0`, bridge at node=0.20 / edge=0.15** | **0.7289** | **0.7118** | **0.7238** | **0.6818** | **0.4000** | **0.6719** | **0.7416** | **0.0344** | **current best learned-scope bridge candidate** |

---

## 6. Current Interpretation

The current results do **not** yet prove that the local event-centric approach is superior to the global baseline in a fully fair end-to-end sense. Oracle-local rewrite still enjoys privileged scope information, and the learned-scope bridge is not yet an end-to-end learned local world model.

What the current Stage-1 results **do** establish is:

- local rewrite is learnable
- delete-aware edge modeling matters
- mild keep-aware weighting improves edge balance
- true type-flip is a distinct failure mode
- flip-aware supervision is much more effective under local modeling than under the global typed baseline
- learned proposal is viable
- explicit edge proposal is necessary
- rewrite-facing threshold choice matters
- proposal-side flip-aware node weighting improves learned-scope type behavior without materially breaking the current edge-side bridge

---

## 7. What Is Still Not Done

Not yet implemented:

- end-to-end proposal+rewrite joint training that actually works
- proposal-vs-rewrite disentangling beyond the current bridge experiments
- causal consistency / order-invariance losses
- long-horizon rollout studies
- real-world data
- raw-image input
- hypergraph rewrite formulation

---

## 8. Recommended Near-Term Direction

The current best next direction is **not** to add more architectural complexity immediately. The more useful path is:

1. keep the current oracle-scope edge lead candidate as the main rewrite reference
2. keep the current learned-scope bridge candidate as the main learned-proposal reference
3. treat frozen-proposal and union-scope rewrite training as negative results / cautionary evidence
4. only move to more coupled proposal+rewrite experiments after these Stage-1 reference points are fully documented

---

## 9. Current Repository Structure

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
│   ├── oracle_local_delta.py
│   └── proposal.py
│
├── train/
│   ├── train_baseline_global.py
│   ├── oracle_sanity_check.py
│   ├── train_oracle_local.py
│   ├── eval_oracle_local_breakdown.py
│   ├── train_oracle_local_delta.py
│   ├── train_oracle_local_delta_earlystop.py
│   ├── eval_oracle_local_breakdown_delta.py
│   ├── eval_type_breakdown.py
│   ├── train_scope_proposal.py
│   ├── eval_scope_proposal.py
│   ├── train_oracle_local_delta_frozen_proposal.py
│   └── train_oracle_local_delta_union_scope.py
│
├── checkpoints/
├── .gitignore
├── README.md
└── ...
```
