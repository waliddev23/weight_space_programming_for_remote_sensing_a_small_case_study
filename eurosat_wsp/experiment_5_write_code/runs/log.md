# W2 run log

Claude Fable 5.2 in claude.ai, code execution, web search and memory off. One chat for all rounds.
The model sees only the practice-set feedback (1,000 validation images). Test scores (5,400 images) are recorded here and never shown to it.

| Round | Practice accuracy (shown to model) | Test accuracy (never shown) | Test balanced accuracy | Notes |
|---|---|---|---|---|
| 0 | 40.2% | 40.3% | 39.3% | No feedback yet. Above the 30.4% colour baseline, below W1's 49.4%. |
| 1 | 47.0% | 45.5% | 45.8% | Same features; only class centroids and spreads re-tuned from the round 0 feedback. Forest 33% → 93% and Pasture 9% → 63% on practice; SeaLake now confused with Forest. Only the code was transferred, not the explanation. |
| 2 | 49.0% | 47.5% | 47.3% | Added a new feature (very smooth blocks, for water) and per-class bias corrections for over- and under-predicted classes. SeaLake 25% → 76% on practice; Forest dropped to 62% (now confused with SeaLake). Only the code was transferred. |
| 3 | 49.4% | 49.2% | 48.7% | Small targeted changes, explained by the model from the confusion matrix: moved the Highway centroid "darker, bluer and less textured, where the true highways were actually going", partly reverted the Forest/SeaLake shift, tightened the coarse-edge spread, re-balanced class biases. |

## Outcome against the pre-registered criteria

| Criterion | Result |
|---|---|
| Round 0 (no feedback) above the 30.4% colour baseline | **Met:** 40.3% |
| Feedback helps: round 3 above round 0 | **Met:** 49.2% against 40.3%, rising every round |
| Depth helps: any round above W1's 49.4% | **Not met:** best 49.2%, a tie within noise |

The trained version of the same network reaches 78.7%, so the hand-written network reaches about 63% of what training achieves. Every round kept the same feature design; feedback was used to re-estimate class profiles and biases, not to change what the network computes. The extra layer was used to compute texture and edge statistics, but the model's limited knowledge of the true class statistics, not network capacity, appears to be the bottleneck.

Reference points: random weights 8.1%, mean-RGB centroid 30.4%, W1 attempt 1 (618 weights, by hand) 49.4%, this network trained normally 78.7%.

Round 0 design (Fable's words): an interpretable feature extractor (raw colour, eight rectified gradient channels, bright pixels; then brightness, blue minus red, green excess, texture energy, smooth and textured block fractions, coarse edges, bright and dark fractions) feeding a nearest-centroid classifier whose class centroids are its own estimates of EuroSAT statistics.

Round 0 main errors: Forest predicted as SeaLake (67 of 100), Pasture predicted as Forest (52 of 100), Highway predicted as River (33 of 100).
