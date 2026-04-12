# Step31 Learned-vs-Late Fusion Consolidation

Step31 validates synthetic multi-view observation as a stronger bridge family.
The learned-vs-late fusion follow-up then asks whether a learned multi-view
encoder can replace the simpler retained bridge.

Conclusion: not yet. `step31_simple_late_fusion` remains the retained Step31
backend-transfer reference. The learned-fusion line is informative, but the
Step31c-e micro-line should now be parked.

## What The Line Established

### Simple Late Fusion

Simple late fusion is currently strongest because it acts as an implicit
stabilizer. It averages independently trained single-view logits, which makes it
less reactive to disagreement-rich pairs. This gives the best recovery F1 and
the best full/context downstream transfer.

### Raw Learned Multi-View Encoder

The raw learned encoder is not weak. It is diagnostically important because it
keeps stronger proposal-sensitive behavior, especially under the noisy
`RFT1 + calibrated P2` backend. Its problem is over-admission: it assigns too
many positive edges in mid/disagreement regions, hurting precision and
full/context stability.

### Step31c

Step31c added a diagnostic disagreement-damped probe. It confirmed the main
failure mode: learned fusion over-admits positive edges in disagreement regions.
The probe narrowed the recovery gap substantially, but it was still a heuristic
and did not replace simple late fusion.

### Step31d

Step31d tried late-fusion-informed edge-head training. It showed that teacher
stability can be learned, but the objective was too blunt. It became too
conservative, improving precision while collapsing useful recall and worsening
proposal-sensitive transfer.

### Step31e

Step31e made the teacher objective asymmetric and recall-preserving. It avoided
Step31d's recall collapse and is the healthier teacher-training probe. However,
it still did not beat late fusion or Step31c on the retained downstream tradeoff.

## Compact Comparison

Recovery metrics are Step31 recovery-side test results. Transfer metrics are
noisy `RFT1 + calibrated P2` frozen-backend outputs.

| Row | Overall F1 | Noisy P | Noisy R | Noisy F1 | RFT full | RFT context | RFT PropR | RFT OOS miss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step30 rev6 reference | 0.7452 | 0.6306 | 0.7049 | 0.6657 | 0.7882 | 0.7962 | 0.5851 | 0.5578 |
| Step31 simple late fusion | 0.9198 | 0.8821 | 0.8501 | 0.8658 | 0.8740 | 0.8844 | 0.6820 | 0.5378 |
| Step31 learned encoder | 0.9034 | 0.7777 | 0.9141 | 0.8404 | 0.8572 | 0.8670 | 0.7036 | 0.5128 |
| Step31c damping probe | 0.9177 | 0.8449 | 0.8755 | 0.8599 | 0.8661 | 0.8760 | 0.6840 | 0.5328 |
| Step31d distilled probe | 0.9019 | 0.9235 | 0.7571 | 0.8321 | 0.8580 | 0.8669 | 0.6075 | 0.6056 |
| Step31e recall-preserving probe | 0.9175 | 0.8691 | 0.8498 | 0.8594 | 0.8664 | 0.8761 | 0.6678 | 0.5514 |

## Retain / Park Decision

Retain:

- Current Step31 backend-transfer reference: `step31_simple_late_fusion`
- Current learned-fusion diagnostic reference: `step31c_agreement_damped_encoder`

Park:

- The Step31c-e learned-fusion micro-line should be parked for the current phase.

Not yet justified:

- Adapter/interface work is still premature.
- Backend joint training is still not justified.
- Reopening Step30 micro-lines is not justified.

## Next-Step Decision

Chosen direction: Option 1, park learned-fusion micro-tweaks now and continue
with late fusion as the active bridge family.

Rationale:

- Step31 already produced the large phase-level win: multi-view evidence itself.
- Late fusion is simple, interpretable, and currently best on recovery plus
  backend transfer.
- Learned fusion is now well diagnosed but not retained.
- Step31c-e improved understanding more than operating point quality.
- Another tiny learned-fusion loss or admission tweak is unlikely to change the
  phase decision.

What would justify reopening learned fusion later:

- A qualitatively stronger learned-fusion phase, not another micro-loss.
- For example, a trained fusion model with explicit per-view reliability
  supervision, or a structured view-consensus latent trained as a first-class
  representation.
- That should be a separate phase, not Step31c-f style incremental tweaking.

## Recommended Next Action

Retain late fusion and move to Step32 as a synthetic rendered/image-like bridge
probe, using `step31_simple_late_fusion` as the structured multi-view reference.
