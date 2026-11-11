# Fable 5.1 reasoning log

How Claude Fable 5.1 reasoned through the three weight-space programming experiments on 10 November 2026, reconstructed from the Claude Code session record. Quoted lines are Fable's own notes, written while it worked. Times are UTC.

Each stage follows the same pattern: what Fable looked at, what it concluded, what it decided, and the prediction it wrote down before seeing the score.

---

## Stage 1 · READ: which training regime produced each model?

**Task.** Nine small CNNs were trained on EuroSAT under three regimes: no augmentation, standard augmentation and strong augmentation, three models each. Fable received only the first-layer and classifier weights of each model, under anonymous names, and had to label all nine.

**19:22 · First pass: measure what the prompt suggests.** Fable parsed the nine models from the prompt text and computed about thirty statistics for each. These covered the hints in the prompt (rotational symmetry and isotropy of the 3×3 filters, how different the R, G and B kernels are, weight norms) and the shape of each classifier's singular-value spectrum.

**Observation.** The filter-symmetry cues the prompt pointed to were weak and noisy. One signal was strong. Three models, 02, 03 and 05, stood apart in their classifiers:

| Classifier statistic | Models 02, 03, 05 | Other six models |
|---|---|---|
| Effective rank (out of 10) | 9.18 to 9.20 | 9.68 to 9.78 |
| Largest / smallest singular value | 3.4 to 3.6 | 2.3 to 2.6 |

**Doubt.** Before trusting this, Fable checked for a hidden confound:

> "Models 02, 03, and 05 form a clean cluster in the classifier features, but I want to rule out a seed confound first, so I'm computing pairwise weight correlations."

**19:24 · The key finding: the hidden seeds.** Correlating the first-layer weights of every pair of models revealed three tight groups. Models in a group correlate at 0.95 to 0.99, and models in different groups at under 0.07. Fable's interpretation was that each group shares a random initialisation (a seed), and that the first layer barely moves away from it during training.

| Seed group | Models |
|---|---|
| A | 01, 02, 08 |
| B | 03, 04, 09 |
| C | 05, 06, 07 |

**Why this matters.** Each regime was trained once per seed, so every group must contain exactly one model of each regime. A blind 9-way labelling problem with 1,680 possible answers becomes three small ordering problems. And within a group, the initialisation is shared, so any difference between the three models comes from the training regime alone.

**Ordering within each group.** In every group, one model was far from the other two in weight space, and that model was always 02, 03 or 05. Fable then oriented the ordering with physical arguments:

- **Colour invariance.** Strong augmentation includes heavy colour jitter and random grayscale. It should push the R, G and B kernels of each filter toward a shared brightness kernel. In all three groups, the far model had the most colour-redundant filters.
- **Classifier concentration.** With heavy augmentation, many fine colour and texture features become unreliable, so the classifier should lean on fewer features. Five classifier statistics changed monotonically along the same ordering in all three groups: 15 of 15 comparisons.
- **None versus standard.** Standard augmentation is milder than strong, so the standard model should sit closer to the strong one. In each group, the model more correlated with the strong model was labelled standard, and the remaining one no augmentation. The feature orderings agreed.

> "The correlation matrix shows three seed groups, each with one outlier model (02, 03, 05), and the features rank the models monotonically."

**19:26 · Committed answer, before scoring.**

| Regime | Models | Stated confidence |
|---|---|---|
| Strong augmentation | 02, 03, 05 | 95% |
| Standard augmentation | 01, 04, 07 | 80% |
| No augmentation | 06, 08, 09 | 80% |

**Outcome.** 9 of 9 correct. The answer key also confirmed the three inferred seed groups exactly.

---

## Stage 2 · EDIT: repair a damaged layer without data

**Task.** In a ResNet-18 at 98.2% test accuracy, one layer (`layer3.0.conv1`, 256 filters) was damaged with Gaussian noise, and accuracy fell to 57.7%. Fable had to write a function that repairs the weight tensor using the tensor alone: no data, no training, no access to the original.

> "Next I'll load only the corrupted checkpoint and run weight-only diagnostics on the damaged layer before designing the repair."

**19:29 · Find the damage.** Fable measured the spread of each filter's weights. Healthy filters have a standard deviation of 0.025 to 0.040, while 32 filters sat at 0.169 to 0.182. A simple rule, "more than three times the median", flags exactly filters 0 to 31. (A public configuration file also listed these filters. Fable disclosed this and made the repair detect them from the tensor itself.)

**Measure how bad it is.** Comparing the damaged and healthy filters gave the noise level. The noise carries 35 times the energy of the original weights, so each damaged filter is about 97% noise.

**19:30 · Can anything be recovered?** Fable tested the three repair strategies the prompt suggested before choosing:

1. **Low-rank cleaning (SVD).** This only works if healthy filters share a few dominant patterns. The healthy filters' spectrum turned out flat: no direction stood above the level expected from pure randomness (the Marchenko–Pastur edge). The damaged filters projected onto the healthy patterns at chance level, 3.1% against 2.8%. There is no structure to recover the signal from.
2. **Clipping outliers.** At this noise level, essentially no healthy weight is large enough to stand out from the noise (a fraction of 0.00001). Clipping cannot separate signal from noise.
3. **Rescaling to healthy norms.** This would give the network 32 random filters at full strength, which is worse than switching them off.

> "The clean filters form a flat Marchenko–Pastur bulk, so the 32 damaged ones can't be recovered from the tensor and I'll replace them with the mean clean filter."

**19:31 · Decision.** Fable framed the repair as the best statistical estimate of each damaged filter given what the healthy ones look like. Because no pattern survives the noise, that estimate collapses to the average healthy filter. The repair therefore neutralises the 32 broken filters with a sensible default instead of trying to reconstruct them. Fable also rejected a naive version of this estimator, which would have kept 92 spurious directions and injected noise while appearing to denoise.

**Committed prediction, before scoring.** Test accuracy of 88 to 94%, recovering 75 to 90% of the lost accuracy. The original filters are unrecoverable, so some loss is unavoidable.

**Outcome.** 95.46% test accuracy, recovering 93.3% of the loss, which is above the prediction.

> "Stage 2 came in at 95.46% accuracy and 93.3% recovery, beating my predicted 88–94%."

*Context added later, not part of Fable's run: simply setting the 32 damaged filters to zero gives 94.1%. Most of the recovery comes from identifying and neutralising the broken filters. Fable's choice of replacement adds about 1.4 points on top.*

---

## Stage 3 · WRITE: hand-write the first layer of a network

**Task.** Write all 432 weights of the first convolutional layer (16 filters, 3×3, RGB) of a satellite image classifier by hand, encoding remote-sensing knowledge.

**19:33 · Design principle.** Fable built every filter as a colour combination times a classical spatial pattern, then scaled the bank to match the statistics of a standard random initialisation (mean 0, standard deviation 0.118). That keeps the rest of the network, which stays random, working as designed.

| Filters | Colour | Spatial pattern | Target |
|---|---|---|---|
| 0 to 3 | green against red and blue (VARI) | blur, centre, Laplacian | vegetation, forest versus pasture |
| 4 to 6 | blue against red | blur, centre-surround | water, sea versus river |
| 7 to 10 | brightness | Sobel and diagonal edges | roads, buildings |
| 11 to 13 | brightness | horizontal, vertical, diagonal ridges | crop rows |
| 14 to 15 | brightness, slight red | spot and corner | roofs, building corners |

**19:34 · Self-correction.** After generating the first version, Fable checked the per-channel balance of each filter and found a design error:

> "The colour-bias term on filters 11 to 15 is overpowering the luminance term, nearly removing the red channel on filter 11. I'm reducing it."

It cut the colour terms from 0.3 to 0.08 on filters 11 to 13 and from 0.4 to 0.1 on filters 14 and 15, then regenerated the bank.

**Committed prediction, before scoring.**

- **Without training**, accuracy will stay near chance (about 10%). The layers after the first are random, so the network's output has no relation to the classes whatever the first layer does.
- **After the standard 20-epoch training**, accuracy will land within about 1.5 points of the random-initialisation reference (90.8%).

**Outcome.**

| Measure | Fable's filters | Random initialisation |
|---|---|---|
| Without training | 9.96% | 8.60% |
| After 20 epochs | 90.94% | 90.81% |

Both predictions held. The hand-written filters train as well as a standard initialisation, with no measurable advantage.

*Context added later, not part of Fable's run: repeated over three seeds, Fable's layer finished ahead of the random initialisation every time (by 0.13, 0.54 and 0.70 points; mean 90.81% against 90.35%). With three seeds the difference is suggestive but not statistically significant (p ≈ 0.12).*

---

## Summary

| Stage | What Fable did | Result | Prediction made beforehand |
|---|---|---|---|
| READ | Found the hidden seed structure, then ordered models within each seed | 9 / 9 | 95% and 80% confidence |
| EDIT | Showed the damage was unrecoverable, chose optimal neutralisation | 95.46% | 88 to 94% |
| WRITE | Encoded remote-sensing priors, fixed its own design error | 90.94% (reference 90.81%) | within ±1.5 points |

Across the three stages, the same habits recur. Fable tested the prompt's suggested methods before relying on them. It looked for confounds before trusting a clean-looking signal. It stated when a goal was impossible instead of forcing a method. And it wrote its prediction down before running the evaluator.
