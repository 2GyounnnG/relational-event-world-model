# Step30 Encoder Phase Design

## Phase Question

Step30 asks whether the project can add a minimal encoder-front-end benchmark before any large end-to-end training:

Can a small encoder recover the current structured graph state from a weak, slot-aligned observation derived from the existing synthetic graph world?

This is the first encoder-phase benchmark. It is not intended to replace the current proposal/rewrite backend.

## Scope

In scope:

- structured synthetic graph-event world only
- one weak observation slot per existing node
- node identity and node ordering preserved
- weak observation to current structured graph recovery
- direct supervision of current node type, current node state, and current edge existence
- clean and noisy weak-observation variants
- lightweight metrics for recovery quality

Out of scope:

- raw image rendering
- real-world data
- object detection or tracking
- hypergraph reformulation
- LLM integration
- end-to-end proposal+rewrite training
- reopening Step22-29 noisy interaction local tweak lines

中文说明：Step30 不是要做真实视觉理解，而是先把“弱观测 -> 结构化图状态”的入口问题单独立起来。

## Weak Observation Format

Each sample keeps the existing clean graph-event fields and adds:

```text
weak_observation:
  slot_features: [N, F_obs]
  relation_hints: [N, N]
  slot_mask: [N]
  feature_names: list[str]
  variant: "clean" | "noisy"
  config: dict
```

The first version is slot-aligned:

- slot `i` corresponds to graph node `i`
- there is no permutation, detection, or tracking ambiguity
- relation hints are weak pairwise evidence, not a new ground-truth adjacency field

Default slot feature layout:

```text
type_hint_onehot[0:num_types]
type_observed_mask[1]
state_hint[state_dim]
state_observed_mask[state_dim]
```

## Observation Variants

### clean

The clean weak observation is still an observation, not a new target copy:

- node state is quantized
- relation evidence is represented as soft pairwise hints
- node identity/order remains aligned

### noisy

The noisy weak observation adds:

- type dropout
- type flips
- state noise
- state dropout
- stronger quantization
- relation dropout
- relation false positives
- relation-hint jitter

The latent current graph and next graph remain clean.

## Dataset Fields

Step30 datasets are normal pickle lists. Each item preserves the source sample fields when available:

- `graph_t`: clean current structured graph target
- `graph_t1`: clean next structured graph
- `events`
- `changed_nodes`
- `changed_edges`
- `event_scope_union_nodes`
- `event_scope_union_edges`

Step30-specific fields:

- `weak_observation`
- `step30_observation_variant`
- `step30_observation_config`
- `step30_source_sample_index`
- `step30_note`

## Recovery Task

Input:

- weak slot features
- weak relation hints

Targets:

- clean current node type from `graph_t["node_features"][:, 0]`
- clean current node state from `graph_t["node_features"][:, 1:]`
- clean current adjacency from `graph_t["adj"]`

The recovery target is the current graph, not the next graph. Later phases can feed recovered beliefs into proposal/rewrite, but Step30b keeps the task direct and inspectable.

## Initial Metrics

The Step30 recovery evaluator reports:

- node type accuracy
- node state MAE
- node state MSE
- edge accuracy
- edge precision
- edge recall
- edge F1
- breakdown by observation variant
- trivial weak-observation baseline when available

The trivial baseline decodes type from the type hint, state from the state hint, and edges from thresholded relation hints.

## Minimal Encoder Baseline

The baseline encoder is intentionally small:

- message passing over weak slot features and relation hints
- node type head
- node state head
- pairwise edge-existence head

This keeps the benchmark debuggable and close to the existing structured graph code.

## Later Frozen-Backend Roadmap

Step30 should progress in stages:

1. **Step30a data:** generate weak observations from existing graph states.
2. **Step30b recovery:** train/evaluate weak observation to current structured graph recovery.
3. **Step30c frozen backend:** feed recovered graph beliefs into the existing frozen proposal/rewrite backend.
4. **Step30d interface diagnosis:** measure whether backend errors come from encoder recovery, proposal scope, or rewrite.
5. **Later only if justified:** consider end-to-end encoder+backend training.

The current pack implements only Step30a/Step30b.
