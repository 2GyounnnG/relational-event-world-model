# Project Status: Steps 2-6

## 1. Project scope

This project is still operating inside a structured synthetic graph-event world. The working stack is a learned-scope bridge: a proposal model predicts event scope, and an oracle-local rewrite model predicts the next graph state under proposal conditioning.

The project is not yet working on raw images, real-world datasets, hypergraphs, language grounding, or LLM-integrated reasoning. The current results should be read as controlled structural-world results, not as real-world transfer claims.

## 2. Step 2 summary

### Central bottleneck

Step 2 focused on learned-scope robust rewrite behavior. The central tension was the keep/delete tradeoff under imperfect proposal masks: stronger edit sensitivity tended to help changed-edge and delete behavior, but could also erode broad full-edge and context-edge stability.

### What was learned

The main result of Step 2 was that a stable broad default exists. `W012` emerged as the best general learned-scope rewrite candidate because it held the best overall balance across broad graph quality, edge stability, and robustness under the proposal/rewrite interface.

The project also established two informative alternatives:

- `WG025` as the strongest edit-preserving alternative
- `DR005` as a delete-rescue anchor

These alternatives were useful because they made the tradeoff frontier explicit. They did not displace `W012` as the broad default.

### Why Step 2 is considered sufficiently closed

Step 2 is treated as sufficiently closed for now because the main tradeoff line is well mapped. `W012` is strong enough to serve as the broad default, `WG025` captures the more edit-preserving direction, and `DR005` anchors the delete-rescue direction. More local retuning inside this line is unlikely to change the project direction materially without a new substrate or new phase objective.

## 3. Step 3 summary

### Why exact reverse-order final-state was vacuous

The first exact-matched reverse-order study used the same base state and the same independent event pair in both orders. That dataset was technically correct, but the evaluation was vacuous: independent events commute to the same final target when the model receives equivalent information, so exact matched final-state reverse-order gaps collapsed to zero.

That result was still useful because it ruled out a misleading Step 3 direction. Exact final-state reverse-order matching is not a nontrivial sequential consistency test in this regime.

### Why sequential composition became the real substrate

Step 3 became meaningful only after shifting to sequential composition. The key substrate used:

- `S0`
- `SA`
- `SB`
- `SAB`

This made it possible to test:

- first-step quality
- second-step quality
- path-level consistency across `A -> B` and `B -> A`

That sequential composition setup was non-vacuous and showed real path-level order sensitivity.

### What `C005` does and does not solve

`C005` is the current Step 3 consistency candidate. It comes from a light consistency objective on the sequential composition substrate. It improved path-gap behavior, especially on harder edit-sensitive metrics, but it did not become a new broad default. The gains came with some degradation in base step quality, and the consistency benefit did not cleanly transfer into rollout stability later on.

So `C005` is best understood as a useful consistency reference and a valid Step 3 candidate, not as a replacement for the broad default.

## 4. Step 4 summary

### Rollout degradation findings

Step 4 introduced short-horizon autoregressive rollout evaluation. The main finding was that rollout degradation is real but not catastrophic. The failure pattern was not sudden collapse. Instead, the system showed conservative drift:

- changed-edge accuracy stayed weak
- add behavior collapsed toward zero
- context-edge accuracy stayed relatively high

This made short-horizon rollout a real but manageable problem.

### What `R050` improves

`R050` is the current rollout-aware candidate. It improved rollout-sensitive edit behavior, especially on changed-edge, delete, and add recovery relative to plain `W012`, but it did so by sacrificing some broad stability on full-edge and context-edge metrics.

### Why `R050` is the current Step 4 candidate

`R050` is not the broad system default, but it is the correct Step 4 rollout-aware reference because it is the clearest point on the rollout-aware side of the tradeoff. It demonstrates that rollout-aware supervision helps, even though the resulting model is not broadly safer than `W012`.

## 5. Step 5 summary

### What changed in the multievent regime

Step 5 moved from simpler sequences into a more complex three-event structural regime. The new benchmark mixed:

- fully independent sequences
- partially dependent sequences
- strongly interacting sequences

This made the project test multi-event interaction structure rather than only single-step behavior or simple two-event composition.

### Why interaction complexity is the main bottleneck

The family transferred into this regime: the models did not collapse, and the learned-scope bridge remained usable. However, the main limiter was not horizon length by itself. The strongest bottleneck was event interaction complexity. Earlier events changed local conditions for later events, and that exposed limits in the proposal/rewrite interface more clearly than simple sequence length did.

### Roles of `W012` and `I1520`

`W012` remained the safest broad-transfer default in Step 5 because it preserved the strongest full-graph and context-edge stability.

`I1520` became the interaction-aware alternative. It improved changed-edge, delete, and interaction-sensitive behavior, but it gave back some of the broader stability that keeps `W012` as the safest default.

## 6. Step 6 summary

### Noisy structured observation findings

Step 6 introduced noisy structured observation: the latent world stayed clean and synthetic, but the model received corrupted structured graph observations as input. This created the first benchmark where the graph input was imperfect even though the clean target next state stayed unchanged.

The family remained usable under noise. The main bottleneck shifted toward proposal robustness under noisy observation, especially on the edge side, and then later toward proposal/rewrite coupling under noisy observation.

### Proposal-side progression

Proposal-side work progressed in four clear stages:

1. Noisy-observation proposal training helped.
   - `P2` became the best proposal-only checkpoint.

2. Global threshold calibration mattered.
   - With global threshold calibration, `P2` became the system-level best proposal front-end.
   - The working thresholds are:
     - `node_threshold = 0.15`
     - `edge_threshold = 0.10`

3. Regime-aware thresholding did not help further.
   - Per-regime operating points for `N1`, `N2`, and `N3` produced no gain over the global calibrated thresholds.

4. Temperature scaling did not help further.
   - Post-hoc temperature scaling improved some conservatism-related metrics but hurt the downstream edit-sensitive metrics that mattered most.

The conclusion is that proposal learning is sufficiently mature for now, and the final proposal front-end is the calibrated `P2` stack rather than a more elaborate calibration variant.

### Why calibrated `P2` is the final front-end for now

Calibrated `P2` is the final noisy-observation proposal front-end for now because:

- proposal-only noisy training helped
- global threshold calibration gave a real downstream improvement
- regime-aware thresholds added nothing further
- temperature scaling added nothing further

That makes `P2 + (0.15, 0.10)` the cleanest stable front-end to carry forward.

### Rewrite-side progression

On the rewrite side, the project then tested noisy-observation adaptation while holding the calibrated proposal front-end fixed.

`RFT1` is the rewrite-only noisy-adaptation candidate. It improved on the calibrated `P2 + W012` baseline in a balanced way and became the best Step 6 main candidate.

The project then tested light joint noisy proposal+rewrite fine-tuning. `J05` showed signal: changed-edge and add behavior improved, which confirms that proposal/rewrite coupling matters. But `J05` did not beat `RFT1` overall. The gains looked like a different tradeoff, not a clear system-level improvement.

### Why `RFT1 + calibrated P2` is the current Step 6 main candidate

`RFT1 + calibrated P2` is the current Step 6 main candidate because it is the best balanced noisy-observation system found so far:

- better than plain `W012 + calibrated P2`
- better overall than the first light joint line
- still simple and faithful to the existing proposal/rewrite interface

At this point, the remaining Step 6 gap appears to be more about deeper proposal/rewrite coupling than about another isolated proposal calibration trick.

## 7. Current stable defaults

### Broad default

- `W012`

### Step-specific defaults

- Step 3 consistency candidate: `C005`
- Step 4 rollout-aware candidate: `R050`
- Step 5 interaction-aware alternative: `I1520`
- Step 6 noisy structured observation main candidate: `RFT1 + calibrated P2`

### Strongest alternatives and anchors

- strongest edit-preserving Step 2 alternative: `WG025`
- delete-rescue anchor: `DR005`
- informative but not current Step 6 default: `J05`

## 8. What is intentionally NOT being done next

The project is not continuing local tuning on Steps 2-6 right now. In particular, it is intentionally not reopening:

- Step 2 keep/delete/rescue retuning
- Step 3 exact reverse-order final-state matching as a main direction
- Step 4 small rollout-loss sweeps around the existing short-horizon line
- Step 5 more local interaction-weight sweeps inside the current first interaction-aware line
- Step 6 threshold-only or temperature-only proposal calibration retuning

The goal now is consolidation and a clean phase handoff, not more micro-optimization inside already-mapped tradeoff lines.

## 9. Next-phase entry point

The next phase should start from the consolidated defaults rather than from reopened tuning loops. A future phase should treat the current system family as established:

- `W012` as the broad default
- `C005` as the sequential composition consistency reference
- `R050` as the rollout-aware reference
- `I1520` as the interaction-aware alternative
- `RFT1 + calibrated P2` as the noisy structured observation stack

Future work should begin by posing a new phase question on top of these defaults, not by re-running old local tradeoff searches unless a clearly new substrate requires them.
