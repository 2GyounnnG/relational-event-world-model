# Step30 rev13 Signed Pair Witness

## Question

Can a small additional weak observation cue provide genuinely new rescue-safety information for low-relation-hint true edges, without making the Step30 weak-observation benchmark trivially decodable?

This pack does not continue the rev7-rev12 rescue-decode micro-line. It keeps the evaluation recovery-first and does not run backend adapter/interface work.

## Mechanism

rev13 adds exactly one new weak pair-level observation field:

`signed_pair_witness`

The cue is synthetic, structured, noisy, and pair-level. It is separate from `relation_hints` and `pair_support_hints`.

Generation behavior:

- Positive-ish values weakly support true low-hint edge rescue.
- Negative-ish values weakly warn against unsafe false admissions.
- Near-zero values remain ambiguous.
- The cue has overlapping positive/negative distributions, dropout, flips, jitter, and quantization.
- It is not a clean adjacency copy.

Implementation detail:

- Existing rev6 `slot_features`, `relation_hints`, and `pair_support_hints` remain unchanged for matching seeds.
- `signed_pair_witness` is generated after the existing weak-observation channels so it does not perturb prior rev6 fields.
- The encoder consumes the cue in the pairwise edge head through `use_signed_pair_witness=True`.
- The recovery objective is the rev6-style objective: relation-logit residual, pair-support hints, and missed-edge recovery loss.
- No selective-rescue decode was used for the retained rev13 recovery eval.

## Data

Generated files:

- `data/graph_event_step30_weak_obs_rev13_train.pkl`
- `data/graph_event_step30_weak_obs_rev13_val.pkl`
- `data/graph_event_step30_weak_obs_rev13_test.pkl`

Split composition:

| Split | Samples | Clean | Noisy |
|---|---:|---:|---:|
| train | 20000 | 10000 | 10000 |
| val | 4000 | 2000 | 2000 |
| test | 4000 | 2000 | 2000 |

Test event counts:

| Event family | Count |
|---|---:|
| edge_add | 1374 |
| edge_delete | 1448 |
| motif_type_flip | 1110 |
| node_state_update | 1284 |

## Weak-Cue Sanity

The cue is not trivially separable. Test-set noisy distributions remain highly overlapping:

- `hint_missed_true_edge`: mean positive but broad, with many zero/negative values.
- `unsafe_rescue_false`: mean negative but broad, with many zero/positive values.

The trivial baseline that averages relation, pair support, and witness does not become strong:

- noisy edge F1: `0.4826`
- overall edge F1: `0.5496`

This confirms the cue is not simply exposing clean adjacency.

## Recovery Results

Hard decode:

- clean threshold: `0.50`
- noisy threshold: `0.55`
- selective rescue: off

| Row | Overall Edge F1 | Clean Edge F1 | Noisy Precision | Noisy Recall | Noisy Edge F1 |
|---|---:|---:|---:|---:|---:|
| rev6 reference | 0.7383 | 0.8215 | 0.6300 | 0.7072 | 0.6550 |
| rev13 encoder | 0.7349 | 0.8218 | 0.6026 | 0.7257 | 0.6481 |
| rev13 trivial | 0.5496 | 0.6166 | 0.4923 | 0.4970 | 0.4826 |

Event-family edge F1 for rev13 encoder:

| Event family | Edge F1 |
|---|---:|
| edge_add | 0.7360 |
| edge_delete | 0.7340 |
| motif_type_flip | 0.7370 |
| node_state_update | 0.7334 |

## Targeted Rescue Diagnostics

Noisy split:

| Row | Hint-Missed True Recall | Hint-Missed Avg Score | Hint-Supported FP Error | HM vs Hard-Neg Win Rate |
|---|---:|---:|---:|---:|
| rev6 reference | 0.2817 | 0.4366 | 0.4688 | 0.3476 |
| rev13 encoder | 0.3925 | 0.4790 | 0.4529 | 0.4535 |
| rev13 trivial | 0.5662 | 0.5178 | 0.5021 | 0.5466 |

Interpretation:

- rev13 does add real new rescue-safety signal: hint-missed recall and hint-missed-vs-hard-negative ordering improve over rev6.
- However, the improvement is not clean enough: noisy edge F1 falls below rev6, and precision drops materially.
- The trivial baseline gets a strong ordering signal on the specific hint-missed-vs-hard-negative diagnostic, but poor global edge recovery. This means the cue is informative but not directly sufficient as a full adjacency decoder.

## Backend Rerun Decision

Focused Step30c backend integration was not run.

Reason:

`rev13 improved hint-missed diagnostics but did not beat rev6 noisy edge F1 and did not produce a clean recovery gate.`

The rev13 cue is a real new observation/representation mechanism, not another decode tweak, but the current first implementation does not justify backend integration yet.

## Conclusion

rev13 partially validates the multi-cue pair-evidence direction. The new `signed_pair_witness` cue is non-leaky and improves the named rescue ranking bottleneck. It does not yet produce a better retained recovery point than rev6.

Recommended next action:

- Do not move to adapter/interface work yet.
- Do not resume rev7-rev12 rescue-decode micro-tweaks.
- If continuing Step30, revise the witness/encoder use so that the new cue improves global noisy edge F1, not only the targeted hint-missed diagnostic.
