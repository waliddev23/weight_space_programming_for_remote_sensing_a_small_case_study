# EuroSAT Weight-Space Programming: Consolidated Report

**Subject model:** Claude Fable 5.1 (Claude Code, terminal access)
**Date:** 2026-11-10 (training benchmark finished 19:41 local)
**Framework:** Takehi, Teoh, Zhang & Khattab, *Sparks of Weight-Space Programming* (MIT CSAIL, Nov 2026)
**Environment:** `benchmarking` conda env, Python 3.14, torch 2.14.1+cu126, CUDA GPU
**Runs:** `runs/read_run_001/`, `runs/edit_run_001/`, `runs/write_run_001/`

## Summary of results

| Stage | Task | Result | Baseline / reference |
|---|---|---|---|
| 1 READ | Classify 9 blind SmallCNNs by augmentation regime | **9 / 9 correct (100%)** | chance 33.3%; 1 in 1,680 exact partitions |
| 2 EDIT | Repair noised `layer3.0.conv1.weight` of ResNet-18 | **95.46% test acc, recovery +93.3%** | corrupted 57.67%, original 98.19% |
| 3 WRITE, zero-shot | Hand-written `conv1.weight` [16,3,3,3], rest random | **9.96%** (+1.36 pts) | Kaiming seed 0: 8.60% (chance) |
| 3 WRITE, 20 epochs | Same conv1 as init, standard-aug training | **test 90.94%, best val 90.13%** | Kaiming ref: test 90.81%, val 90.39% |

All predictions and repair logic were written to disk before the corresponding evaluator was run; each evaluator was run exactly once. Blindness constraints were respected (details per stage).

---

## Stage 1: READ (9-model blind sweep)

**Files:** `runs/read_run_001/raw_response.txt` (predictions + reasoning), `analysis_features.py/.json` (feature code and values), `eval_output.txt`.

**Blindness.** Only `prompts/prompt_read_exp1_all9_models.txt` was read. The key JSON was never opened and the checkpoint directory was never listed. The predictions were committed to the response file before `eval-read` was invoked.

**Method.** The 432 conv1 weights and the 10x128 classifier of each model were parsed from the prompt text and reduced to interpretable statistics. The decisive step was recognising the hidden seed structure: pairwise element-wise correlation of conv1 across models is 0.95 to 0.99 within three triples and below 0.07 everywhere else, because conv1 moves little from its Kaiming init in 20 epochs. That gives seed groups {01, 02, 08}, {03, 04, 09}, {05, 06, 07}, and the constraint that each condition takes exactly one model per seed group. Within each group the three models lie on a line in weight space (one far outlier, two close neighbours), and the axis was oriented with the mechanistic priors:

- **Channel divergence:** the outlier of every seed group (02, 03, 05) has the highest luminance fraction and inter-channel kernel cosine, i.e. the most colour-invariant filters, as grayscale and heavy jitter predict for `strong_augmentation`. The other end of each line has the most colour-opponent kernels (`no_augmentation`).
- **Classifier geometry:** effective rank, top singular value, singular-value ratio, row-norm CV and column-norm CV of the head are monotonic along the proposed order in all three seed groups (15 of 15 comparisons). Strong-augmentation heads concentrate weight on fewer features and shrink the Highway prototype (row norm 0.77 to 0.83 vs about 1.05 elsewhere).
- **Rotational closure** of the conv1 bank was a weaker, mostly concordant signal.

**Predictions and outcome.** All nine were correct. The evaluator's key confirmed the inferred seed groups exactly (seed 0 = models 01, 02, 08; seed 1 = 03, 04, 09; seed 2 = 05, 06, 07).

| Model | Prediction | Truth |
|---|---|---|
| 01 | standard | standard (seed 0) |
| 02 | strong | strong (seed 0) |
| 03 | strong | strong (seed 1) |
| 04 | standard | standard (seed 1) |
| 05 | strong | strong (seed 2) |
| 06 | none | none (seed 2) |
| 07 | standard | standard (seed 2) |
| 08 | none | none (seed 0) |
| 09 | none | none (seed 1) |

**Observation for the paper's framing.** The raw per-filter "isotropy" cues suggested in the prompt were weak at this scale (16 filters that stay 95% correlated with their init). The strong signals were relational (seed recovery, within-seed ordering) and head geometry, which are only available when the whole set of weights is read together.

---

## Stage 2: EDIT (ResNet-18 surgical repair, variant A)

**Files:** `runs/edit_run_001/raw_response.txt` (diagnostics, rationale, committed prediction), `parsed_output/repair_script.py`, `eval_output.txt`; harness metrics in `experiment_2_edit/metrics/llm_repair_a_result.json`.

**Isolation.** `experiment_2_edit/original/` and the `*_modified_indices.npy` files were never opened. Disclosure: the public `variant_a_gaussian_noise.json` (not on the forbidden list) states that filters 0 to 31 were noised at scale 6.0. The repair function does not use it; it finds the damaged rows from the tensor alone (row std greater than 3 times the median), which independently flags exactly filters 0 to 31.

**Diagnostics (weight-only).** Noise power is 35.5 times the signal power per weight (noise sigma 0.173 vs weight sigma 0.029). The 224 healthy rows have a flat Marchenko-Pastur spectrum: no principal direction exceeds the bulk edge, and the damaged rows' projection on the top 32 healthy directions is at chance (3.1% vs 2.8%). Element-wise thresholding is also hopeless (1e-5 of healthy weights exceed two noise sigmas). Channel-norm matching was rejected because it would inject unit-energy random filters.

**Repair.** Debiased PCA-Wiener (MMSE) estimation of each damaged row under the empirical Gaussian prior from the healthy rows, with eigenvalues debiased by the Marchenko-Pastur edge. Since every debiased gain is zero, the estimator reduces to its prior mean: each damaged filter is replaced by the mean healthy filter (norm 0.158 vs about 0.98 for a typical filter). Healthy rows are untouched.

**Committed prediction vs outcome.** Predicted 88 to 94% test accuracy (recovery 75 to 90%). Actual: **95.46%, recovery +93.28%**. The result is better than predicted; the 32 neutralised channels cost 2.7 points of the original 98.19%. The lesson is that for noise this severe the correct weight-space edit is optimal neutralisation, not reconstruction, and that the network tolerates losing 12.5% of this layer's channels gracefully. A tensor-only edit cannot correct the BatchNorm offset on the muted channels; using the BN running statistics or downstream weights would be the next lever.

---

## Stage 3: WRITE (conv1 functional weight synthesis)

**Files:** `runs/write_run_001/raw_response.txt` (design, committed expectations, JSON tensor), `make_filters.py` (generator), `conv1_design.json/.npy`, `parsed_output/first_conv_fable.pt`, `parse_output.txt`, `eval_zero_shot_output.txt`, `eval_train_output.txt`; harness metrics in `experiment_3_write/baseline/llm_zero_shot_metrics.json` and `llm_trained_metrics.json`.

**Design.** Each of the 16 filters is a (colour vector) x (3x3 spatial atom) product, zero-mean and rescaled to std 0.118 (Kaiming fan_out). Layer statistics: mean 0.0000, std 0.1181, range [-0.43, +0.59].

- Filters 0 to 3: smoothed VARI (2G-R-B), pixel G-R excess, Laplacian of VARI (vegetation boundaries), dark-and-green (Forest vs Pasture vs urban).
- Filters 4 to 6: smoothed B-R (water), blue dominance with low brightness (SeaLake), centre-surround on blue (smooth vs rippled water, River vs SeaLake).
- Filters 7 to 10: luminance Sobel-x, Sobel-y, 45-degree and 135-degree edges (roads, buildings).
- Filters 11 to 13: horizontal, vertical and diagonal ridge detectors with small colour terms (crop rows, furrows, orchard lines).
- Filters 14 and 15: centre-surround spot and quadrant-corner detectors with a red-brown bias (roofs, building corners).

**Zero-shot.** 9.96% test accuracy vs 8.60% for the Kaiming seed-0 baseline. As committed beforehand, this is chance level: conv1 feeds three random conv blocks and a random linear head, so the class mapping is arbitrary whatever conv1 does. The zero-shot metric cannot measure first-layer quality in this architecture.

**20-epoch training (standard augmentation, seed 0 for all other layers).**

| | Best val | Test | Epoch 1 val |
|---|---|---|---|
| Claude conv1 init | 90.13% (ep. 16) | **90.94%** | 62.2% |
| Kaiming seed-0 reference | 90.39% (ep. 16) | 90.81% | n/a |

Delta on test: +0.13 points, within seed noise, matching the committed expectation of about +-1.5 points. The handcrafted bank is at least as trainable as random init; it did not produce a measurable gain at 20 epochs. Macro-F1 and the per-epoch history are in `llm_trained_metrics.json`.

---

## Harness notes

- `wsp_parser.py` contained a duplicated stray block (IndentationError at line 127) that prevented every evaluator from importing; the three stray lines were removed and the original kept as `wsp_parser.py.bak_before_fix`.
- On the Windows cp1252 console the evaluators crash printing check marks; run them with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`.
- Commands were run from `D:\PhD_Thesis\UM6P\presto` so the documented `weight_space_programming/...` paths resolve.

## Overall assessment

Across the three WSP primitives the subject model read a 9-network weight dump with perfect accuracy by recovering latent seed structure and head geometry, chose a statistically optimal (neutralising) weight edit that recovered 93% of the lost accuracy without data or gradients, and wrote a physically motivated first-layer filter bank that trains on par with Kaiming initialisation. The READ and EDIT results are strong evidence for native weight-space reasoning; the WRITE result shows that interpretable first-layer priors are a wash under full training, and that zero-shot evaluation with random downstream layers is not an informative metric for this experiment.
