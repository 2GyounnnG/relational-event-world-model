# Step31 Synthetic Multi-View Observation Bridge Probe

## Scope

Step31 introduces one new synthetic observation family: multiple independent weak structured views of the same graph-event recovery target.

It does not reopen parked Step22-Step30 micro-lines, does not add backend joint training, does not move to adapter/interface work, and does not introduce real images, real-world data, hypergraphs, or LLM integration.

## Data

Each raw graph-event sample is expanded into clean/noisy variants as before, but each variant now contains three weak observation views:

| view | profile | intended difference |
| --- | --- | --- |
| 0 | `relation_focus` | relation hints are slightly more reliable; support/bundle evidence is noisier |
| 1 | `support_focus` | pair support is slightly more reliable; relation and witness evidence are noisier |
| 2 | `evidence_focus` | signed witness and bundle channels are slightly more reliable; relation/support are noisier |

All views observe the same clean `graph_t` target, but use independent corruption masks, jitter, flips, dropout, and quantization. The views do not expose clean adjacency. The first view is also stored in the old `weak_observation` slot so a Step30-style single-view baseline can be trained on the same Step31 data.

Generated splits:

| split | samples | clean | noisy | views |
| --- | ---: | ---: | ---: | ---: |
| train | 20000 | 10000 | 10000 | 3 |
| val | 4000 | 2000 | 2000 | 3 |
| test | 4000 | 2000 | 2000 | 3 |

## Model

The Step31 encoder is intentionally small:

- shared per-view `SimpleGraphEncoder`,
- node fusion from mean/std of per-view node latents,
- edge head over fused node-pair latents plus mean/std/min/max relation and support evidence,
- mean/std signed witness and pair-evidence bundle features,
- relation-logit residual over the mean relation/support hint.

This tests the multi-view evidence substrate, not a large new architecture family.

## Baselines

| baseline | description |
| --- | --- |
| Step30 rev6 reference | fixed retained Step30 recovery reference |
| single-view baseline | Step30-style model trained only on Step31 view 0 |
| trivial multi-view | deterministic mean-fusion of relation/support/witness/bundle hints |
| simple late fusion | apply the single-view model independently to all views, then average logits |
| Step31 multi-view encoder | learned shared-view encoder plus explicit fusion |

## Recovery Results

| row | overall P | overall R | overall F1 | clean F1 | noisy P | noisy R | noisy F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step30 rev6 reference | 0.7196 | 0.7727 | 0.7452 | 0.8282 | 0.6306 | 0.7049 | 0.6657 |
| single-view baseline | 0.7395 | 0.8211 | 0.7782 | 0.8648 | 0.6432 | 0.7616 | 0.6974 |
| trivial multi-view | 0.9455 | 0.7846 | 0.8576 | 0.9598 | 0.9676 | 0.5811 | 0.7261 |
| simple late fusion | 0.9320 | 0.9078 | 0.9198 | 0.9732 | 0.8821 | 0.8501 | 0.8658 |
| Step31 multi-view encoder | 0.8608 | 0.9504 | 0.9034 | 0.9707 | 0.7777 | 0.9141 | 0.8404 |

## Targeted Diagnostics

Noisy split only:

| row | hint-missed recall | hint-supported FP error | rescue P | rescue R | agreement precision | disagreement precision | disagreement FP rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Step30 rev6 reference | 0.2817 | 0.4688 | n/a | n/a | n/a | n/a | n/a |
| single-view baseline | 0.5844 | 0.3290 | 0.4785 | 0.6851 | 0.7881 | 0.5341 | 0.2308 |
| trivial multi-view | 0.1472 | 0.0544 | 0.8812 | 0.2353 | 0.9748 | 0.9570 | 0.0071 |
| simple late fusion | 0.5042 | 0.2431 | 0.6936 | 0.7311 | 0.9284 | 0.8305 | 0.0617 |
| Step31 multi-view encoder | 0.7227 | 0.4062 | 0.5460 | 0.9219 | 0.8819 | 0.6771 | 0.1634 |

## Event-Family Edge F1

| row | edge_add | edge_delete | motif_type_flip | node_state_update |
| --- | ---: | ---: | ---: | ---: |
| single-view baseline | 0.7795 | 0.7767 | 0.7788 | 0.7776 |
| trivial multi-view | 0.8578 | 0.8568 | 0.8636 | 0.8555 |
| simple late fusion | 0.9223 | 0.9188 | 0.9214 | 0.9166 |
| Step31 multi-view encoder | 0.9061 | 0.9028 | 0.9038 | 0.9007 |

## Diagnosis

Synthetic multi-view is a real stronger bridge family. Even trivial mean-fusion beats the Step30 rev6 reference, and learned/simple multi-view baselines improve substantially over the Step31 single-view baseline.

The strongest result is not the custom Step31 fusion encoder. Simple late fusion of independently trained single-view predictions is best on this first probe, with noisy F1 `0.8658` versus `0.8404` for the learned multi-view encoder. This is useful: it says the main win comes from independent view evidence and cross-view averaging, not from a complex fusion architecture.

The learned Step31 encoder is more recall-heavy. It achieves very high hint-missed recall (`0.7227`) and rescue recall (`0.9219`), but pays more precision cost than simple late fusion. Cross-view agreement is highly precision-supportive, and disagreement remains a meaningful risk signal.

The trivial baseline is strong but still clearly below learned/simple fusion. The bridge does not collapse into clean-adjacency leakage, but the first Step31 view setup is much easier than Step30 single-view because independent views reduce corruption variance dramatically.

## Gate

Step31 clears the recovery-side rev6 gate, but no backend integration was run in this task. The right immediate interpretation is: continue in the synthetic multi-view direction, but keep the next step recovery-side and sanity-focused because the simple late-fusion baseline currently dominates the custom learned fusion encoder.
