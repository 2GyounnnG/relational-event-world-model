# Step32 Mechanism Hypothesis Shortlist

## Current Status

The sandbox line is preserve-and-stop. It is a positive feasibility result for oracle-targeted local event rewriting and composition, not a mandate to continue sandbox expansion.

Step33 implementation is closed. It remains diagnostic-only and should not be reopened as the next implementation line.

Step32 is the leading candidate direction, but it is not formalized as the official next-phase entry point. The retained Step32 references remain isolated candidates under the fixed threshold protocol.

Step32 still lacks one concrete mechanism worth implementing. The current blocker is not threshold tuning, scale, backend transfer, or another rendered-observation expansion by itself; it is a missing observation/representation mechanism that could plausibly close the gap to the retained Step30 rev6 recovery reference.

## Shortlist

### 1. Target-Identity Evidence Inside Candidate Region

Definition: represent the rendered / image-like observation as a broad candidate event region plus an explicit target-identity evidence signal that marks which object or relation inside that region is most likely to be the actual changed target.

Why this fits Step32 better than Step33: Step32 is about observation and representation, and this hypothesis asks whether the observation bridge can expose target identity before recovery; Step33 was a local rewrite diagnostic and is already closed.

Current blocker addressed: broad candidate support may be insufficient when the model still cannot identify the changed subset inside the support region.

Risk: the mechanism can become too close to oracle target supervision if the evidence signal is specified as a label rather than as recoverable observation structure.

Reason it may fail: rendered cues may not contain enough reliable target-specific evidence to distinguish the true changed object or edge from nearby contextual objects.

### 2. Instance-Stable Object/Relation Tokens

Definition: convert the rendered / image-like observation into instance-stable object and relation tokens that preserve object identity and candidate edge identity across the synthetic observation before graph recovery.

Why this fits Step32 better than Step33: this is a representation bridge from image-like observations to graph/event recovery, while Step33's question was rewrite behavior after structured supports are already available.

Current blocker addressed: Step32 may be threshold-sensitive because the representation does not bind rendered evidence cleanly to graph identities, especially under noisy observations.

Risk: the representation could drift back toward structured multi-view hints and stop testing the rendered / image-like bridge question cleanly.

Reason it may fail: stable instance tokens may improve bookkeeping without adding enough event-change evidence to lift noisy F1 toward Step30 rev6.

### 3. Local Change-Evidence Residual Field

Definition: represent the image-like observation through a local residual evidence field that emphasizes signed before/after change cues around objects and relations, rather than relying only on absolute rendered scene features.

Why this fits Step32 better than Step33: Step32 needs a stronger observation substrate, and residual evidence is an observation-level mechanism; Step33-style rewrite composition is not the limiting question here.

Current blocker addressed: calibrated Step32 performance appears saturated, suggesting the current rendered bridge may not isolate the actual change signal strongly enough.

Risk: a residual field may overfit synthetic rendering artifacts and fail to preserve the mechanistic recovery comparison.

Reason it may fail: residual evidence can highlight nuisance motion or visibility differences without resolving which graph node or edge should be recovered.

## Ranking

1. **Target-Identity Evidence Inside Candidate Region**
2. **Instance-Stable Object/Relation Tokens**
3. **Local Change-Evidence Residual Field**

## Recommended Hypothesis

Recommend **Target-Identity Evidence Inside Candidate Region** as the one paper-only Step32 mechanism direction to carry forward.

This is the strongest candidate because it directly converts the sandbox lesson into a Step32 observation/representation hypothesis: a broad region can be valid but still inadequate unless the model has a way to identify the true target inside that region. The original sandbox local operator had oracle scope and still failed to beat monolithic; direct target-mask conditioning changed the result qualitatively. Step32 should now ask, on paper only, whether rendered / image-like observations need an analogous target-identity evidence mechanism rather than more scale, threshold adjustment, or broader candidate regions.

This recommendation is not an implementation plan. It does not reopen Step32 implementation, Step33, Step22-29, backend transfer, or training. The next memo should formalize only this hypothesis enough to decide whether it is worth a future bounded prototype.

README should remain unchanged now.
