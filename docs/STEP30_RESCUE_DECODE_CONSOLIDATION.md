# Step30 Rescue-Decode Consolidation

## Phase Question

Step30 introduced a weak-observation encoder phase before the existing frozen structured backend. After the initial recovery benchmark, the main difficulty became edge-side recovery under noisy weak observation, especially low-relation-hint true edges that the backend proposal needs for changed/add behavior.

This document consolidates the Step30 rescue/decode micro-line from rev6 through rev12 and records the current stop/park decision. It does not introduce a new experiment, backend adapter, soft adjacency interface, or proposal/rewrite joint training.

## Scope And Non-Scope

In scope:

- Consolidate Step30 rev6-rev12 rescue/decode findings.
- State which recovery/decode point remains the current reference.
- Explain why repeated small rescue-decode/scorer/auxiliary tweaks should be parked for now.
- Design the next stronger observation/encoder-side mechanism family.

Out of scope:

- Parked Step22-Step29 noisy interaction tuning.
- Backend adapter or interface work.
- Soft adjacency backend input.
- Proposal/rewrite joint finetuning.
- Raw images, real-world data, hypergraph, or LLM integration.

Stable defaults remain unchanged:

- Broad clean default: `W012`.
- Noisy broad default: `RFT1 + calibrated P2`.
- Interaction-aware alternative: `I1520`.
- Consistency reference: `C005`.
- Rollout-aware reference: `R050`.
- Retained noisy interaction-aware branch candidate: `Step26 proposal + RFT1`.

## Revision Summary

### rev6: first real missed-edge rescue signal

rev6 added `pair_support_hints` and was the first Step30 revision to produce a nontrivial upstream recovery gain over rev2 on noisy edge recovery. It substantially improved the named bottleneck, `hint_missed_true_edge`, and justified a focused frozen-backend rerun.

However, rev6 still left a downstream gap: the recovery win only partially transferred into frozen-backend behavior. It improved some missed-edge-related behavior but was not enough to move Step30 into adapter/interface work.

### rev7: global backend-aware threshold opened the gate too wide

rev7 showed that lower decode thresholds can improve proposal edge recall, out-of-scope miss, and add on noisy `RFT1 + calibrated P2`.

The problem was that the mechanism was too coarse. It over-admitted edges globally, causing large full/context/delete damage. This made clear that the correct direction was not another global threshold, but a selective rescue path.

### rev8: selective rescue was directionally real

rev8 introduced a narrow rescue channel for low-relation-hint candidates with pair-support evidence. It was clearly better aligned than rev7: it preserved part of rev7's proposal/OOS/add gains while recovering much of rev7's full/context/delete damage.

But rev8 still did not cleanly beat rev6 as an operating point. The rescue channel improved proposal/OOS/add relative to rev6, but still paid too much in full/context/delete.

### rev9: hand-built guard did not fix rescue precision

rev9 added a small guarded ranking rule on the rev8 rescue channel. The guard reused the same scalar signals and did not materially improve the operating point. This showed that another hand-tuned formula over the same weak evidence was unlikely to solve safe rescue admission.

### rev10: supervised rescue-safety scorer was still too weak

rev10 replaced the hand-built guard with a tiny supervised rescue-safety scorer over compact candidate features. The scorer improved rescue ranking only minimally. Validation AP and accepted rescue precision remained weak, and downstream behavior was not a clean improvement over rev8/rev9.

This suggested the bottleneck was not just a missing linear separator over the existing scalar features.

### rev11: richer post-hoc candidate context still did not unlock a clean point

rev11 trained a tiny scorer over richer frozen rev6 candidate context, including endpoint/pair features where available. It did not improve rescue acceptance precision, noisy edge F1, or noisy `RFT1 + calibrated P2` proposal/OOS/add relative to rev10.

The result suggested that the frozen recovery representation itself was not exposing enough rescue-safety signal for post-hoc decoding to exploit.

### rev12: encoder-side rescue-safety auxiliary objective was also insufficient

rev12 added a small rescue-safety auxiliary head during encoder training. It did reduce some hint-supported false-positive error and recovered some delete behavior relative to rev10/rev11, but it gave back too much of the intended proposal/OOS/add rescue benefit.

The accepted rescue precision was worse than rev8/rev10/rev11, and the noisy `RFT1 + calibrated P2` proposal recall and OOS miss did not beat the prior rescue points.

## Compact Downstream Comparison

Noisy `RFT1 + calibrated P2` frozen-backend integration:

| Revision | Full | Context | Changed | Add | Delete | Proposal Edge Recall | OOS Miss |
|---|---:|---:|---:|---:|---:|---:|---:|
| rev6 | 0.7882 | 0.7962 | 0.3156 | 0.1861 | 0.4307 | 0.5851 | 0.5578 |
| rev8 | 0.7540 | 0.7613 | 0.3189 | 0.2453 | 0.3847 | 0.6067 | 0.5193 |
| rev10 | 0.7532 | 0.7605 | 0.3140 | 0.2421 | 0.3766 | 0.6105 | 0.5178 |
| rev12 | 0.7557 | 0.7628 | 0.3305 | 0.2303 | 0.4219 | 0.6024 | 0.5300 |

Interpretation:

- rev6 is the broadest and safest recovery/decode reference among these rows.
- rev8/rev10 improve proposal recall, OOS miss, and add relative to rev6, but full/context/delete degrade substantially.
- rev12 recovers some delete and changed-edge behavior relative to rev10, but gives back proposal recall, OOS miss, and add.
- No row is a clean retained operating point for frozen-backend Step30.

## Compact Rescue Diagnostics

Noisy recovery/rescue diagnostics:

| Revision | Hint-Missed True Recall | Hint-Supported FP Error | Accepted Rescue Precision |
|---|---:|---:|---:|
| rev6 | 0.2817 | 0.4688 | n/a |
| rev8 | 0.4536 | 0.4688 | 0.1641 |
| rev10 | 0.4564 | 0.4688 | 0.1668 |
| rev12 | 0.4197 | 0.4306 | 0.1522 |

Interpretation:

- rev6 first established real low-hint true-edge rescue.
- rev8/rev10 increased missed-edge admission, but did not reduce hint-supported false-positive error.
- rev12 reduced hint-supported false-positive error, but reduced missed-edge recall and accepted rescue precision.
- The line repeatedly fails to separate safe rescue candidates from unsafe false admissions well enough.

## Park / Retain Decision

Current retained Step30 recovery/decode reference:

`rev6 missed-edge evidence recovery with retained rev6 decode settings remains the current best Step30 recovery reference.`

Status:

- This is a diagnostic/reference point, not a stable backend operating point.
- It is useful because it introduced the first real missed-edge rescue signal and remains the broadest recovery/decode reference.
- It is not enough to justify adapter/interface work because downstream frozen-backend gains are partial and unstable.

Park decision:

`The rev7-rev12 rescue-decode micro-line should be parked for the current phase.`

Reason:

- Selective rescue is directionally real.
- But repeated small decode, guard, scorer, richer-context scorer, and auxiliary-head tweaks did not produce a clean operating point.
- Proposal recall / OOS / add gains are repeatedly bought with full/context/delete damage.
- Post-hoc rescue scoring over current features is too weak.
- Small encoder-side rescue-safety shaping is also too weak.
- The bottleneck is no longer "one more clever decode tweak"; it is insufficient observation/representation signal for safe rescue decisions.

Chinese summary:

当前这条线已经证明「选择性救边」方向是真的，但也证明了现有信号不够。再调一个阈值、再换一个小 scorer、再加一个轻量 aux head，大概率只是在 recall 和 false admission 之间来回搬损失，不会产生新的干净 operating point。

## Next Mechanism Design Memo

### Missing Information

The current weak observation exposes two useful but insufficient pair-level signals:

- `relation_hint`: still dominates ranking and remains too close to the original ambiguous relation evidence.
- `pair_support_hint`: helps low-hint true-edge rescue, but does not reliably distinguish safe rescues from unsafe false admissions.

What is missing is an independent weak signal about rescue safety:

- Whether a low-relation-hint pair has positive evidence from another weak source.
- Whether pair-support is corroborated by endpoint-compatible evidence rather than being generic noisy support.
- Whether admitting the edge is structurally plausible for the current weak observation, not just high-scoring under the existing edge head.

The current encoder can learn "this pair may be an edge" better than rev2, but not "this low-hint rescue is safe enough to admit."

### Stronger Observation/Representation Family

Recommended family:

`multi-cue pair evidence for rescue safety`

This remains synthetic, structured, weak-observation-only, and node-identity-aligned. It does not add images, detection, tracking, hypergraph structure, LLMs, backend adapters, or backend joint training.

The family should add a small number of independent weak pair cues that are not direct adjacency labels. The cue should be designed to help distinguish:

- safe low-hint true-edge rescues
- unsafe high-support false admissions

Candidate cue design:

- A signed `pair_witness_hint` channel with weak positive and weak negative evidence in one compact scalar or two compact channels.
- The positive side should be more likely for true low-hint edges, especially where relation_hint is weak but the synthetic process leaves another weak trace.
- The negative side should be more likely for unsafe false admissions, especially where pair_support is high but not corroborated.
- The channel should be noisy, dropped out, quantized, and imperfect so trivial decode remains a meaningful floor.

This is meaningfully different from rev6-rev12:

- rev6 added pair support that helped rescue but not safety.
- rev8-rev11 tried to choose better with the same frozen signals.
- rev12 tried to shape representation with the same observation information.
- The proposed family changes the observation evidence available to the encoder, so the model has a new basis for rescue-safety separation.

### Smallest First Experiment

Run exactly one small first experiment in this new family:

`Step30 rev13 signed_pair_witness benchmark probe`

Minimal setup:

- Keep the rev6 benchmark family and encoder architecture recognizable.
- Add one inspectable weak pair cue: `pair_witness_hint`.
- Generate it only from the existing synthetic graph process; do not expose clean adjacency.
- Apply dropout/noise/quantization so trivial decode remains non-perfect.
- Train a rev6-style encoder that consumes `relation_hint`, `pair_support_hint`, and `pair_witness_hint`.
- Keep the ordinary edge recovery objective and one modest rescue-safety auxiliary objective if needed.
- Do not add backend adapters or soft adjacency.

Primary evaluation gates before any Step30c rerun:

- Noisy edge F1 must clearly beat rev6.
- Hint-missed true-edge recall must remain high.
- Hint-supported false-positive error must improve relative to rev8/rev10.
- Accepted rescue precision must improve materially over rev10.
- The improvement must not come from a global over-admission pattern.

Only if those gates pass should a focused frozen-backend integration rerun be justified.

## Recommended Next Action

Run exactly one small first experiment in the new stronger mechanism family:

`Step30 rev13: add a noisy signed_pair_witness weak observation cue and evaluate recovery/rescue diagnostics first.`

Do not proceed to adapter/interface work yet, and do not continue the rev7-rev12 rescue-decode micro-line without a genuinely stronger observation/representation signal.
