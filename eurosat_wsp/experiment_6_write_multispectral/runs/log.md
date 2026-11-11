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
