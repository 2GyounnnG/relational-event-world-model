# Step32 Target-Identity Implementation Decision

## 1. Current Step32 Status

Step32 remains the leading next-phase candidate, but it is not formalized as the official next-phase entry point.

Retained Step32 references:

- best calibrated isolated reference: `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30`
- best observed default-threshold isolated reference: `step32_rendered_bridge_candidate_scale @ clean=0.50, noisy=0.50`

What is still blocked:

- Step32 remains below the retained Step30 rev6 recovery reference.
- Step32 remains threshold-sensitive.
- default-threshold behavior remains seed-sensitive.
- scale alone did not improve the best calibrated reference.
- no concrete non-leaky observation/representation mechanism has been implemented or validated.

What remains fixed:

- default thresholds: `clean=0.50`, `noisy=0.50`
- calibrated thresholds: `clean=0.30`, `noisy=0.30`
- split calibrated/default Step32 retained references
- Step30 rev6, Step31 `step31_simple_late_fusion`, and trivial rendered baselines as comparison rows
- backend transfer closed
- repo-wide defaults unchanged
- Step33 implementation closed
- Step22-29 local tweaking closed

Step32 is still not formalized because the current evidence supports retained isolated candidates, not a promoted system checkpoint. The line has real positive signal, but it has not crossed the Step30 rev6 reference and has not yet specified a mechanism strongly enough to justify reopening implementation.

## 2. Why Target-Identity Evidence Is The Best Candidate

Target-identity evidence is the current best mechanism candidate because it directly addresses the most concrete lesson from the sandbox line: broad candidate support is not enough when the model cannot identify the true target inside that support.

The sandbox result is specific. The first local operator had oracle event scope but did not beat the monolithic baseline. The bounded event-target-mask revision changed the result qualitatively. That does not mean Step32 should consume oracle masks. It means the next Step32 mechanism should ask whether the rendered / image-like bridge needs observation-derived target evidence inside a broader candidate region.

This is more specific than scale-only continuation because it changes the representation question rather than adding capacity. It is more specific than threshold-only continuation because the fixed Step32 protocol already closes threshold fiddling. It also avoids treating rendered-observation expansion itself as the mechanism.

## 3. What Remains Uncertain

The core uncertainty is whether target-identity evidence can be made genuinely observation-derived.

The current prototype spec names the right shape: relation-slot salience with target-specific residual support. But it does not yet define the actual salience construction tightly enough to prove that it avoids oracle leakage.

Remaining uncertainties:

- whether relation-slot salience can be computed from allowed rendered / image-like observations without using changed masks, event target ids, or final graph labels
- whether the candidate relation slots already carry hidden target information through their construction
- whether target-specific residual support is a legitimate observation cue or a proxy for label-derived change masks
- whether the mechanism is specific enough to implement without drifting into architecture search
- whether any eventual improvement would come from target disambiguation rather than extra capacity, favorable thresholds, or seed selection

The idea is promising, but the implementation boundary is not yet tight enough.

## 4. Formal Decision

Choose **(b): keep this as paper-only for now**.

Do not approve a Step32 implementation run yet.

The conservative reason is simple: the current docs identify the best mechanism candidate, but they do not fully specify a non-leaky observation-derived salience source. Approving implementation now would risk turning a good hypothesis into an underconstrained prototype, or worse, accidentally importing oracle target identity through the representation.

## 5. Preserve

Preserve these docs:

- `docs/STEP32_SOURCE_OF_TRUTH_CONSOLIDATION.md`
- `docs/STEP32_MECHANISM_GAP_DECISION_MEMO.md`
- `docs/STEP32_MECHANISM_HYPOTHESIS_SHORTLIST.md`
- `docs/STEP32_TARGET_IDENTITY_EVIDENCE_PROTOTYPE_SPEC.md`
- `docs/STEP32_TARGET_IDENTITY_IMPLEMENTATION_DECISION.md`
- `docs/SANDBOX_MVP_CONSOLIDATED_STATUS.md`
- `docs/NEXT_PHASE_ENTRY_PLAN_AFTER_STEP33.md`

Preserve these retained references:

- `step32_rendered_bridge_candidate_next @ clean=0.30, noisy=0.30`
- `step32_rendered_bridge_candidate_scale @ clean=0.50, noisy=0.50`
- Step30 rev6 retained recovery reference
- Step31 `step31_simple_late_fusion`
- trivial rendered baselines

Preserve current policy:

- Step32 remains leading but not formalized.
- backend transfer remains closed.
- fixed Step32 thresholds remain fixed.
- split calibrated/default Step32 references remain split.
- repo-wide defaults remain unchanged.
- Step33 and Step22-29 remain closed.

Future condition needed before implementation is reconsidered:

The next paper step must define exactly how relation-slot target salience is computed from allowed Step32 observations, and must show why that computation is not equivalent to feeding changed masks, event target ids, or final recovery labels. It should also define a diagnostic that would separate true target-disambiguation gains from generic capacity, threshold, or seed effects.

Until that condition is met, target-identity evidence remains a preserved hypothesis, not an approved implementation.

## 6. What Not To Do

Do not implement Step32 now.

Do not reopen backend transfer.

Do not tune thresholds.

Do not run scale-only Step32 continuation.

Do not add a new rendered-observation expansion as the main mechanism.

Do not reopen Step33 implementation.

Do not reopen Step22-29 local tweaking.

Do not reopen Step30 or Step31 micro-lines.

Do not convert sandbox lessons into repo-wide default changes.

Do not use real data, raw images, hypergraphs, LLM integration, or backend joint training.

## 7. Final Recommendation

Keep the Step32 target-identity evidence prototype paper-only for now.

It is the best mechanism candidate, and it should be preserved. It is not yet strong enough to justify a bounded implementation run because the current spec has not fully nailed down the non-leaky observation-derived relation-slot salience source. The next valid move is another narrow paper clarification of that evidence source, not implementation.

README should remain unchanged now.
