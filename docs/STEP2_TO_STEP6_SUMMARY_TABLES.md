# Step 2-6 Summary Tables

## Step-level summary

| Step | Main question | Current default | Strongest alternative | Key bottleneck | Status |
|---|---|---|---|---|---|
| 2 | Can the learned-scope bridge support a robust broad rewrite default under imperfect scope? | `W012` | `WG025` (`DR005` as delete-rescue anchor) | keep/delete/edit tradeoff at the proposal/rewrite interface | sufficiently closed for now |
| 3 | What is the right non-vacuous consistency substrate, and what light consistency line is worth keeping? | `C005` | `W012` as the stronger non-consistency baseline | sequential composition consistency without giving back too much step quality | partially closed |
| 4 | How much autoregressive degradation appears under short-horizon rollout, and does rollout-aware tuning help? | `R050` | `W012` as the safer broad baseline | conservative drift under rollout, especially changed-edge and add behavior | partially closed |
| 5 | Does the method family transfer to a more complex 3-event structural regime? | `W012` | `I1520` | event interaction complexity | partially closed |
| 6 | Can the learned-scope bridge remain effective under noisy structured observation? | `RFT1 + calibrated P2` | `W012 + calibrated P2` (`J05` as the main joint signal) | noisy observation robustness and proposal/rewrite coupling | sufficiently closed for now |

## Model and recipe roles

| Model / recipe | Role in project | Strengths | Weaknesses |
|---|---|---|---|
| `W012` | broad default learned-scope rewrite model | strongest overall balance; safest broad transfer; stable full-edge/context-edge behavior | less edit-sensitive than the more aggressive alternatives |
| `WG025` | strongest edit-preserving Step 2 alternative | better edit-preserving behavior on harder edge changes | gives back too much broad stability to replace `W012` |
| `DR005` | delete-rescue anchor | strongest reference point for rescue-oriented delete behavior | rescue-specific; not a broad default |
| `C005` | Step 3 sequential composition consistency candidate | improves path-gap behavior on the real sequential consistency substrate | not a broad default; some step-quality degradation; consistency benefit did not transfer cleanly to rollout |
| `R050` | Step 4 rollout-aware candidate | improves rollout-sensitive changed/delete/add behavior | weaker full-edge/context-edge stability than `W012` |
| `I1520` | Step 5 interaction-aware alternative | better interaction-sensitive changed-edge/delete behavior | weaker broad-transfer stability than `W012` |
| `P2 + calibrated thresholds` | final proposal-side noisy-observation front-end | best proposal-side noisy front-end; threshold calibration gives real downstream gains | still recall-heavy; proposal-side maturity does not remove rewrite-side noise gap |
| `RFT1 + calibrated P2` | current Step 6 main candidate | best balanced noisy structured observation stack so far | still incremental rather than decisive; deeper coupling gap remains |
| `J05` | first light joint noisy proposal+rewrite line | shows real joint coupling signal; helps changed-edge/add behavior | does not beat `RFT1 + calibrated P2` overall; mostly a different tradeoff |
