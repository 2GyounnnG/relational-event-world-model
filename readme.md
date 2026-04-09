# Relational / Event-Centric World Model

Stage-1 synthetic graph-event world model.

## 1. Project goal

This repository studies a narrow question in a controlled setting:

> In a structured synthetic graph world with local events, when does local event-centric modeling help over a plain global graph transition baseline?

The project is intentionally staged. The current repository is still **strictly Stage 1**:

- synthetic graph-event environment
- structured graph state input
- no raw images
- no LLM integration
- no real-world data
- no learned proposal yet
- no causal-consistency regularization yet
- no hypergraph rewrite formulation yet

## 2. Current Stage-1 dataset

Implemented event types:

1. `node_state_update`
2. `edge_add`
3. `edge_delete`
4. `motif_type_flip`

Implemented dataset features:

- train / val / test splits
- variable-size graphs
- padded batching with `node_mask`
- one-event and two-event transitions
- independent two-event pairs
- `changed_nodes`
- `changed_edges`
- `event_scope_union_nodes`
- `event_scope_union_edges`

A key Stage-1 distinction is that the dataset stores both:

- the **changed region** (what actually changed), and
- the **event scope** (the local region associated with the event, including context)

This is important for oracle-local experiments.

## 3. What is implemented

### Global baselines

- global whole-graph baseline
- typed global baseline with:
  - node type head
  - node state head
  - edge head

### Oracle-local baselines

- oracle local typed rewrite baseline
- oracle local delta-edge rewrite baseline
- merge-back into full graph output
- scope-only and full-graph evaluation

### Diagnostics and evaluation

- oracle merge-back sanity check
- delta-edge breakdown evaluation
- changed-vs-context edge analysis
- typed flip diagnostics (`eval_type_breakdown.py`)

## 4. Main current findings

### 4.1 Oracle-local edge rewriting is learnable

Under oracle event scope, local rewrite dynamics are clearly learnable.

### 4.2 Edge prediction is still the main Stage-1 bottleneck

Node state and node type are comparatively stable. Edge prediction remains the main challenge.

### 4.3 The original local-edge failure mode was delete collapse

Early oracle-local edge runs showed that `edge_add` was much easier than `edge_delete`.
The delta `keep/add/delete` formulation plus delete-aware weighting fixed the worst collapse.

### 4.4 The current edge tension is no longer “can delete be learned?”

The current tension is:

> how to improve `keep` / `context` behavior without giving back too much `delete`.

### 4.5 `motif_type_flip` is a distinct typed failure mode

Typed diagnostics showed that the main type-side error is not general node-type prediction.
The real failure mode is **true type-flip targets** (`current_type != target_type`), where both global and oracle-local typed baselines show a strong copy-current-type bias.

### 4.6 Flip-aware supervision is much more effective under oracle-local modeling than under global modeling

Adding a flip-aware type loss weight only marginally improves the global typed baseline, but produces a much larger gain in the oracle-local typed baseline.
This suggests that locality helps not only edge rewriting, but also discrete type-transition disambiguation.

## 5. Current Stage-1 comparison view

### 5.1 Current edge comparison (oracle-local delta)

| Model | scope_edge | delta_all | keep | add | delete | changed | context | Current role |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Delta reference (`delta_delete_weight=3.0`) | 0.7179 | 0.7039 | 0.7125 | 0.9179 | 0.4708 | 0.6826 | 0.7374 | delete-oriented reference |
| Delta + `delta_keep_weight=1.10` | 0.7362 | 0.7207 | 0.7426 | 0.9370 | 0.4389 | 0.6719 | 0.7718 | **current lead candidate** |
| Delta + `delta_keep_weight=1.25` | 0.7410 | 0.7253 | 0.7639 | 0.9399 | 0.3861 | 0.6455 | 0.7940 | keep/context-oriented upper point |

Interpretation:

- `delta_delete_weight=3.0` remains a strong delete-oriented reference.
- A mild keep-aware weighting (`delta_keep_weight=1.10`) gives the best current balance among recent edge ablations.
- A stronger keep weight (`1.25`) improves keep/context further, but gives back too much delete.

### 5.2 Typed comparison focused on `motif_type_flip`

| Model | full | changed | flip_target | nonflip_changed | scope | single-event `motif_type_flip` flip | Current role |
|---|---:|---:|---:|---:|---:|---:|---|
| Global typed | 0.9724 | 0.5365 | 0.0000 | 1.0000 | — | 0.0000 | global typed reference |
| Global typed + `type_flip_weight=2.0` | 0.9716 | 0.5441 | 0.0181 | 0.9984 | — | 0.0152 | supervision-only check |
| Oracle-local typed | 0.9717 | 0.5483 | 0.0290 | 0.9969 | 0.9117 | 0.0273 | oracle-local typed reference |
| Oracle-local typed + `type_flip_weight=1.5` | 0.9706 | 0.5609 | 0.0634 | 0.9906 | 0.9082 | 0.0697 | **balanced typed candidate** |
| Oracle-local typed + `type_flip_weight=2.0` | 0.9644 | 0.6348 | 0.2699 | 0.9499 | 0.8888 | 0.2394 | aggressive flip-recovery variant |

Interpretation:

- The typed failure is highly concentrated in **true flip targets** rather than in non-flip changed nodes.
- The global typed model remains strongly biased toward copying the current type, even with flip-aware weighting.
- Under oracle-local modeling, flip-aware weighting becomes much more effective.
- `type_flip_weight=1.5` currently looks like the cleaner tradeoff; `2.0` serves as proof that the failure mode is correctable.

## 6. Explored but not currently pursued

The following directions were explored and are not current mainline directions:

- model-wide dropout on the oracle-local delta line
- edge-head-only dropout on the oracle-local delta line
- 3-way keep/add/delete checkpoint selection for the delta line

These variants tended to push the system toward delete / changed sensitivity at the expense of keep / context balance.

## 7. Recommended entry points

### Train typed global baseline

```bash
python train/train_baseline_global.py \
  --train_path data/graph_event_train.pkl \
  --val_path data/graph_event_val.pkl
```

### Train oracle-local typed baseline

```bash
python train/train_oracle_local.py \
  --train_path data/graph_event_train.pkl \
  --val_path data/graph_event_val.pkl
```

### Train oracle-local delta baseline

```bash
python train/train_oracle_local_delta_earlystop.py \
  --train_path data/graph_event_train.pkl \
  --val_path data/graph_event_val.pkl \
  --delta_delete_weight 3.0
```

### Evaluate type-flip diagnostics

```bash
python train/eval_type_breakdown.py \
  --model_kind oracle_local_typed \
  --checkpoint_path checkpoints/oracle_local_rewrite_typed/best.pt \
  --data_path data/graph_event_test.pkl \
  --split_name test
```

### Evaluate delta-edge breakdown

```bash
python train/eval_oracle_local_breakdown_delta.py \
  --checkpoint_path checkpoints/oracle_local_rewrite_delta/best.pt \
  --data_path data/graph_event_test.pkl \
  --split_name test
```

## 8. Current interpretation

At the current Stage-1 stopping point:

- local rewrite is viable in principle
- delete-aware edge modeling matters
- keep/context vs delete remains the main edge-side tradeoff
- `motif_type_flip` is a real typed failure mode rather than a cosmetic metric artifact
- flip-aware supervision helps much more under oracle-local typed modeling than under global typed modeling

This does **not** yet show a fully fair end-to-end local-vs-global win, because the oracle-local models still use privileged oracle scope. But it does show that event-centric locality carries real useful signal in the clean Stage-1 setting.

## 9. Not done yet

Still not implemented:

- learned event proposal / matcher
- proposal-vs-rewrite disentangling beyond oracle scope
- causal consistency / order-invariance regularization
- long-horizon rollout studies
- image-based input
- real-world data
- hypergraph rewrite formulation

## 10. Suggested near-term direction

Near term, the most useful direction is not to expand the problem setup, but to keep the Stage-1 story clean:

1. retain the current edge lead candidate (`delta_keep_weight=1.10`)
2. retain the current typed balanced candidate (`oracle_local_typed + type_flip_weight=1.5`)
3. use the more aggressive typed variant (`type_flip_weight=2.0`) as a proof-of-correctability reference
4. avoid new architecture changes before the Stage-1 local-vs-global comparison story is fully documented
