# Step32 Target-Identity Evidence Prototype Spec

## 1. Why This Mechanism Now

Current status is conservative. The sandbox line is preserve-and-stop. Step33 implementation is closed. Step22-29 local tweaking remains parked. Step32 is the leading next-phase candidate, but it is not formalized as the official next-phase entry point because it remains threshold-sensitive, below the retained Step30 rev6 recovery reference, and backend transfer is closed.

The sandbox lesson matters because it isolated a concrete mechanism gap: broad locality was not enough. The first local operator had oracle event scope and still did not beat the monolithic baseline. It became clearly better only after direct event-target identity was provided inside the broader scope. The Step32 analogue is not to import sandbox oracle masks, but to ask whether rendered / image-like observations need an explicit representation of target evidence inside a broad candidate region.

This is more promising than scale-only or threshold-only continuation because the Step32 docs already show that pure scale helped one operating point without improving the best calibrated reference. Threshold adjustment is also closed for current comparisons. The remaining blocker is a representation mechanism, not a bigger model or another operating-point sweep.

## 2. Mechanism Definition

**Target-identity evidence inside candidate region** means a representation that separates three roles:

- **Broad candidate region:** the coarse set of candidate objects, relation slots, or local graph elements that may contain the recovery target.
- **Target identity evidence inside that region:** an observation-derived salience signal over those candidates that says which candidate object or relation appears most responsible for the change.
- **Final recovery prediction:** the downstream graph/event recovery output evaluated under the fixed Step32 protocol.

The preferred formulation is a per-candidate salience representation over existing candidate slots. Each candidate object or relation inside the broad region receives an evidence value derived from the rendered / image-like observation and local residual cues, and the recovery head consumes that evidence as representation context.

This mechanism is not:

- an oracle target mask
- a changed-region label used as model input
- a new backend-transfer attempt
- a threshold retuning exercise
- a scale-only model increase
- a new rendered-observation family by itself
- a Step33 rewrite continuation
- a proposal-discovery module for the sandbox line

## 3. What Must Remain Fixed

The Step32 fixed protocol remains unchanged:

- default threshold: `clean=0.50`, `noisy=0.50`
- calibrated threshold: `clean=0.30`, `noisy=0.30`

Retained Step32 references remain unchanged:

- `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30`
- `step32_rendered_bridge_candidate_scale @ clean=0.50, noisy=0.50`

The comparison frame remains unchanged:

- compare against Step30 rev6 retained recovery reference
- compare against Step31 `step31_simple_late_fusion`
- compare against trivial rendered baselines
- preserve split calibrated/default Step32 references

Closed boundaries remain closed:

- backend transfer remains closed
- repo-wide defaults remain unchanged
- Step33 implementation remains closed
- Step22-29 local tweaking remains closed
- no formal Step32 promotion is implied

## 4. What Is Allowed To Change

Only one observation/representation-level mechanism is allowed to change: candidate slots inside the Step32 bridge may receive target-identity evidence derived from the same controlled synthetic rendered / image-like observation setting.

The prototype boundary does not allow broad architecture search. It does not allow a new scale regime, a new threshold protocol, a new rendering expansion as the main claim, backend transfer, real data, raw images, hypergraphs, or LLM integration.

## 5. Candidate Evidence Sources

Bounded evidence forms:

- **Object-level salience inside candidate region:** a score or feature indicating which candidate object visually carries change evidence.
- **Relation/slot salience inside candidate region:** a score or feature indicating which candidate pair, edge, or relation slot visually carries change evidence.
- **Target-specific residual evidence inside candidate region:** a local before/after or clean/noisy residual cue associated with each candidate object or relation slot.

Preferred evidence form for this prototype:

**Relation/slot salience with target-specific residual support.**

This is the tightest match to the observed mechanism gap. The sandbox failure was clearest when a broad scope contained multiple plausible edges but only one edge was the direct target. Step32 should therefore specify candidate relation slots as the primary carrier of target-identity evidence, with object salience used only as supporting context when the target is node-like.

## 6. Leakage Guardrails

The evidence must be computed from the allowed Step32 observation representation, not from ground-truth changed masks, event labels, or oracle target identities.

Too much supervision or hidden leakage would include:

- feeding ground-truth changed node or edge masks as input
- feeding oracle event target ids as input
- deriving salience directly from recovery labels rather than observation cues
- constructing candidate slots using test-time target labels
- selecting the target candidate with any rule that depends on the ground-truth final graph
- evaluating only examples where the evidence source already encodes the answer

Labels that may be used only for evaluation:

- clean/noisy recovery targets
- changed node and changed edge labels
- event or recovery class labels
- oracle target identities, if available for diagnostic measurement

Those labels may measure whether target salience aligns with the true target. They must not be consumed as model inputs.

## 7. Prototype Structure

Exactly one bounded prototype is specified:

**Relation-slot target-evidence Step32 bridge.**

Input:

- the existing Step32 synthetic rendered / image-like observation substrate
- the existing candidate region representation used by the retained Step32 bridge line
- no ground-truth target mask or changed-region label as input

Intermediate representation:

- candidate object slots and candidate relation slots inside the broad candidate region
- one target-evidence value or small evidence feature per candidate relation slot
- optional object-slot support evidence only where needed to contextualize relation slots

Output:

- the same final recovery predictions used by the retained Step32 evaluation protocol
- optional diagnostic salience alignment metrics may be defined later, but they are not new success metrics for this spec

Comparison rows:

- retained `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30`
- retained `step32_rendered_bridge_candidate_scale @ clean=0.50, noisy=0.50`
- trivial rendered baselines
- Step30 rev6 retained recovery reference
- Step31 `step31_simple_late_fusion`

What stays fixed:

- thresholds
- train/val/test comparison protocol
- recovery target definition
- retained references
- backend-transfer status
- repo-wide defaults
- closed Step33 and Step22-29 boundaries

What changes:

- only the intermediate Step32 representation gains relation-slot target evidence inside the candidate region

## 8. Success Criteria

Evidence would justify later implementation if the paper prototype can specify a non-leaky observation-derived relation-slot salience mechanism that is distinct from threshold tuning, scale, and oracle target masks.

For a later implementation to be considered successful, it would need to improve relative to current Step32 retained references under the fixed protocol:

- improve over `candidate_next` at calibrated `0.30 / 0.30`
- improve over `candidate_scale` at default `0.50 / 0.50`, accounting for the same-scale variance caveat
- reduce the gap to Step30 rev6, especially noisy F1
- avoid replacing the split calibrated/default references with an uncalibrated one-off row

Target-identity evidence would be doing real work if improvement is concentrated in cases where the broad candidate region contains multiple plausible targets and the salience representation correctly separates the actual target from nearby context. A result that improves only uniformly across all cases without target-disambiguation evidence would not strongly support this mechanism.

## 9. Failure Criteria

The hypothesis is not worth implementing if the paper specification cannot define a target-evidence signal that is observation-derived, bounded, and non-oracular.

It would look like threshold tuning or scale by another name if the proposed gain depends on:

- changing the fixed `0.50 / 0.50` or `0.30 / 0.30` thresholds
- adding capacity without changing the representation
- selecting a favorable seed as the main result
- adding a broader rendered observation family without target-evidence structure

It would indicate hidden oracle leakage if the evidence signal:

- matches ground-truth changed masks by construction
- consumes event target ids
- depends on the final graph target during input construction
- is only meaningful because labels are embedded into candidate slots

## 10. What Not To Do Next

Do not implement this prototype now.

Do not run training or evaluation.

Do not reopen backend transfer.

Do not continue Step32 by scale-only escalation.

Do not tune thresholds.

Do not reopen Step33 implementation.

Do not reopen Step22-29 local tweaking.

Do not reopen Step30 or Step31 micro-lines.

Do not convert sandbox lessons into repo-wide default changes.

Do not add real data, raw images, hypergraphs, LLM integration, or backend joint training.

## 11. Final Decision

This prototype is worth carrying forward to a later implementation-decision memo as a paper-only candidate.

The decisive reason is that it is the first Step32 mechanism proposal that directly targets the documented gap beyond scale and thresholding while respecting the sandbox lesson: broad candidate regions need target identity inside them. This spec does not approve implementation. It only defines the bounded mechanism clearly enough for a later memo to decide whether a prototype run is justified.

README should remain unchanged now.
