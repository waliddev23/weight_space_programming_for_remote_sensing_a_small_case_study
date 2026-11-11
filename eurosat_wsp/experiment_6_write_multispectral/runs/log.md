# W3 run log: multispectral (13 bands)

Claude Fable 5.2 in claude.ai, code execution, web search and memory off. One chat for all rounds.
Same protocol as W2. The model sees only practice-set feedback (W2's 1,000 validation images). Test scores (5,400 images, identical to the RGB test set) are recorded here and never shown to it.

## Data

- Source: `giswqs/EuroSAT_MS`, re-split onto the RGB experiments' exact images by filename.
- Test: all 5,400 images matched, labels identical to RGB. Practice set: all 1,000 W2 practice images matched.
- No multispectral version exists for `SeaLake_2335` (train) and `SeaLake_509` (validation); both dropped.
- Band order verified against the data: corr(B08, B8A) = 0.983, cirrus B10 mean reflectance 0.0013.

## Baselines (computed 2026-11-11, before any attempt)

| Reference | RGB (W2) | Multispectral (W3) |
|---|---|---|
| Best constant guess | 11.3% | 11.3% |
| Random weights, 3 seeds | 8.1% | 8.7% |
| Nearest centroid on mean colour / mean spectrum | 30.4% | 48.6% |
| Same network trained with AdamW, 3 seeds | 78.7% | 81.5% (79.8, 81.2, 83.6) |
| Fable, round 0 | 40.3% | 51.8% |
| Fable, round 3 | 49.2% | 59.5% |

## Rounds

| Round | Practice accuracy (shown to model) | Test accuracy (never shown) | Test balanced accuracy | Notes |
|---|---|---|---|---|
| 0 | 52.2% | 51.8% | 51.9% | No feedback yet. Above the 48.6% mean-spectrum centroid and above W2 round 0 (40.3%). Design: raw bands, water / vegetation / bare indices, visible and NIR texture, per-cell thresholds, coarse edges, Gaussian nearest-prototype classifier with estimated class profiles. SeaLake 100% and Pasture 91% on practice; Highway (8%) and Residential (9%) mostly confused with each other and Industrial. |
| 1 | 60.1% | 59.0% | 58.8% | Prototypes only, conv layers unchanged. Lowered the texture profiles of built-up classes (Residential and Industrial had fallen into Highway), raised Forest's NIR and vegetation profile to match Pasture's, widened the vegetation spread, and penalised over-predicted Pasture. Forest 67% → 96%, Industrial 40% → 65% on practice. |
| 2 | 61.3% | 60.4% | 59.9% | Prototypes and biases only. Gave Residential more vegetation and coarser edges than Highway, and rebalanced biases (Pasture and Industrial down, Residential up). Residential 20% → 41% on practice, but Industrial 65% → 53%, now mistaken for Residential. Only the code was transferred. |
| 3 | 60.3% | 59.5% | 59.1% | Separated Industrial and Residential spectrally rather than by texture: Residential made darker and greener, the texture gap narrowed, biases rebalanced. On the test set Residential stayed at 41% and Industrial at 53%, while Annual Crop fell from 52% to 40%; overall a small step back from round 2. |

## Outcome against the pre-registered criteria

| Criterion | Result |
|---|---|
| Round 0 above the 48.6% mean-spectrum centroid | **Met:** 51.8% |
| Feedback helps: round 3 above round 0 | **Met:** 59.5% against 51.8% (peak 60.4% at round 2) |
| Spectral knowledge helps: each round above the RGB run (W2) | **Met:** 51.8% vs 40.3% at round 0, 59.5% vs 49.2% at round 3 |

## Reading the result

- **Higher in absolute terms, but much of the gain is in the data.** The extra bands lift the trivial mean-spectrum classifier from 30.4% to 48.6%. Fable's margin over that trivial classifier was smaller with 13 bands (+3.2 points at round 0, +10.9 after feedback) than with RGB (+9.9 and +18.8).
- **Closer to what training achieves.** After feedback Fable reached 73% of the trained network's accuracy with 13 bands (59.5 of 81.5), against 63% with RGB (49.2 of 78.7).
- **Natural classes are nearly solved, built-up classes are not.** On the test set after round 3: SeaLake 99%, Forest 96%, River 83%, Pasture 76%, but Highway 27%, Residential 41%, Industrial 53%. Feedback moved errors between the built-up classes without separating them; the round-0 feature design was never changed.
- **Feedback saturates after one round.** +7.2 points after round 1, then +1.4 and -0.9.

## Follow-up: Fable's network as a starting point for training (W3-init)

Pre-registered in `docs/next_steps.md` before any run. Fable's final W3 network (round 3) against Kaiming initialisation, trained with the W3 ceiling recipe at three label budgets, 3 seeds each. "Rescaled" divides Fable's classifier by one constant (scores with unit spread on 1,000 training images), which leaves its predictions unchanged but un-saturates the softmax: Fable's classifier weights reach 6,000 against about 0.1 for standard initialisation. Code: `init_compare.py`; all runs: `init_compare/results.json`.

| Training images | Starting point | Test accuracy, mean (per seed) | Val. accuracy after epoch 1 | Verdict vs Kaiming |
|---|---|---|---|---|
| 100% (16,199) | Kaiming | 82.8% (82.5, 82.3, 83.5) | 56.5% | |
| 100% (16,199) | Fable, as written | 82.5% (80.4, 80.2, 86.9) | 55.8% | no clear difference (-0.3) |
| 100% (16,199) | Fable, rescaled | **86.3%** (86.0, 86.7, 86.3) | 59.9% | **helps (+3.6)** |
| 10% (1,620) | Kaiming | 69.4% (69.6, 69.3, 69.4) | 23.1% | |
| 10% (1,620) | Fable, as written | 64.3% (73.0, 60.0, 60.0) | 35.8% | no clear difference (-5.1) |
| 10% (1,620) | Fable, rescaled | **76.2%** (73.2, 77.1, 78.2) | 37.0% | **helps (+6.8)** |
| 1% (163) | Kaiming | 63.7% (63.4, 62.6, 65.1) | 11.7% | |
| 1% (163) | Fable, as written | 66.1% (65.4, 60.8, 72.0) | 32.0% | no clear difference (+2.4) |
| 1% (163) | Fable, rescaled | **67.5%** (64.7, 68.4, 69.4) | 23.7% | **helps (+3.8)** |

Reading:
- **Rescaled, Fable's network is a better starting point at every budget, on every seed.** The largest gain is with 10% of the labels (+6.8 points); with all labels it still adds +3.6.
- **As written, it is unstable.** Its oversized classifier saturates the softmax: some seeds do well, others stall near Fable's untrained 59.5% (two 10% runs end at 60.0%). Its value lies in what it encodes, not in its exact scale.
- **Head start.** After one epoch, Fable's networks are well ahead of Kaiming when labels are scarce (e.g. 32% against 12% at 1%).
- **Caveats.** Three seeds per cell; the Kaiming result here (82.8%) differs from the W3 ceiling (81.5%) because this experiment keeps data on the GPU in half precision, so comparisons are only made within this table. The rescaling was decided before any run, but it is still a correction applied by us, not by the model.
