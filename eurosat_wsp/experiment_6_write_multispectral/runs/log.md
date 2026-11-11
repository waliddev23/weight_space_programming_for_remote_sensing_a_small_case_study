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
| Fable, round 0 | 40.3% | |
| Fable, round 3 | 49.2% | |

## Rounds

| Round | Practice accuracy (shown to model) | Test accuracy (never shown) | Test balanced accuracy | Notes |
|---|---|---|---|---|
