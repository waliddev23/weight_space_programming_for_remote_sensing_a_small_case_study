# Blog plan: an interactive explainer on GitHub Pages

Brief for building the article that presents this project. Written for the model and subagents that will build it, who do not have the history of the project: everything they need is in this repository, and this file says where.

**Goal.** A reader with no machine-learning background understands, in a few minutes and mostly through visuals, that a language model wrote working neural networks without data, what it got right, where it failed, and how it reasoned. Every visual answers one question in under five seconds.

## Story order

1. **Hook.** One visual and one sentence: *"No training data. No gradient descent. A language model wrote every weight of a satellite-image classifier, and it works."*
2. **Part 1: the model writes networks (W1 to W3).** What a network is, what Fable wrote, how it scored, how feedback changed it, RGB against 13 bands.
3. **Part 2: is the written network a good starting point? (W3-init, the ablation).** Fable against standard initialisation at three label budgets, and why the as-written version is unstable.
4. **Part 3: the earlier pilot (READ, EDIT, WRITE).** Reading training history from weights, repairing a ruined layer, writing a first layer.
5. **What it means, and what it does not.** Limitations, comparison with Takehi et al., links to the repository.

## Visuals, with their data

Every visual is built from files in this repository; no number is typed by hand.

### Hook

| Visual | Interaction | Data |
|---|---|---|
| Confusion matrix of Fable's W3 network evolving from round 0 to round 3, accuracy counter beside it | Play / step through rounds | `eurosat_wsp/experiment_6_write_multispectral/private/round_*.json` (`test.confusion_matrix`, `test.test_accuracy`) |

### Part 1: writing networks

| Visual | Interaction | Data |
|---|---|---|
| **What a network is.** Image → 16 detectors → averages → class scores, drawn as boxes with arrows | Hover a stage to see its size and role | Architecture in `w1.py` (TinyNet) and `w2.py` / `w3.py` (SmallNet) |
| **What the task looks like.** Ten example images per class, RGB and false colour | Click a class | *To export:* thumbnails from the test split (see "Assets to export") |
| **What Fable's detectors see.** A sample image and the response map of each first-layer detector | Click an image, click a detector | *To export:* activation maps from `runs/round_3/weights.npz` (W3) and `runs/w1_attempt_1/weights.pt` (W1) |
| **What Fable believed versus reality.** Per-class mean reflectance across the 13 bands, measured, with Fable's assumed class profiles overlaid | Toggle classes | Measured: *to export* from `eurosat_wsp/data/ms/`. Fable's values: the prototype table `T` in `runs/round_0/response.txt` and `round_3` (W3) |
| **Feedback rounds.** Accuracy per round, RGB and 13 bands, with the trivial baselines | Step through rounds; each step shows Fable's own explanation | `experiment_5_write_code/private/round_*.json`, `experiment_6_write_multispectral/private/round_*.json`, `private/baselines.json`; explanations in `runs/round_*/response.txt`; summary in each `runs/log.md` |
| **Nature solved, cities not.** Recall per class after round 3, natural against built-up classes | Hover a bar | `experiment_6_write_multispectral/private/round_3.json` (`test.per_class_recall`) |

### Part 2: the ablation

| Visual | Interaction | Data |
|---|---|---|
| **Fable against Kaiming.** Interactive version of `assets/fig7_write_init.png` | Switch label budget; hover a seed | `experiment_6_write_multispectral/init_compare/results.json` |
| **Training curves.** Validation accuracy per epoch for the three starting points | Switch budget and seed | `results.json`, field `val_history` |
| **Why "as written" is unstable.** Ten class scores turned into probabilities, with a temperature slider: at Fable's scale one class takes 100%, after rescaling the probabilities are graded | Slider | Scores of one example image from the W3 network, *to export*; scale factor as in `init_compare.py` (`fable_net`) |

### Part 3: the pilot

| Visual | Interaction | Data |
|---|---|---|
| **READ: seeds appear.** The 9×9 correlation matrix shown in anonymous order, then reordered into three blocks | Button "sort by seed", animated | Checkpoints in `experiment_1_read/checkpoints/`, key in `prompts/key_read_exp1_all9_models.json`, functions in `read_features.py`; static version `assets/fig1_read_seed_structure.png` |
| **READ: fingerprints.** Colour invariance against classifier rank, seeds joined by lines | Hover a model | `read_features.py` on the same checkpoints; static `assets/fig2_read_fingerprints.png` |
| **EDIT: the ruined model.** The damaged layer's 256 filters as a grid of tiles: original, noised (32 filters turn to static), repaired (replaced by the mean filter). Below: predictions on a few test images and the confusion matrix for original, damaged and repaired | Three-way toggle | *To export:* `experiment_2_edit/original/resnet18_original_state_dict.pt`, `corrupted/resnet18_corrupted_a_gaussian_noise.pt`, repair from `runs/edit_run_001/parsed_output/repair_script.py`; confusion matrices in `experiment_2_edit/corruption_configs/variant_a_gaussian_noise.json` and `metrics/` |
| **EDIT: all repair methods.** Interactive version of `assets/fig3_edit_methods.png` | Switch corruption type | `experiment_2_edit/metrics/edit_baselines.json` |
| **WRITE: the 16 filters** | Hover a filter for Fable's description | `runs/write_run_001/parsed_output/first_conv_fable.pt`, descriptions in `runs/write_run_001/raw_response.txt` |
| **How Fable reasoned.** A step-by-step timeline: what it looked at, what it doubted, what it decided, the prediction it wrote down before scoring | Click through steps | `eurosat_wsp/runs/fable_reasoning_log.md`; quotes and times from `runs/evidence/fable_reasoning_transcript.md` |

## Assets to export first

A single script, for example `site/build_data.py`, writes everything the page needs into `site/data/` as small JSON files and PNG thumbnails. Visual work starts only once this exists, so every component reads real data.

1. Test thumbnails: 10 per class, RGB and 13-band false colour (near infrared, red, green), 64×64 upscaled with nearest neighbour.
2. Activation maps of the first-layer detectors of W1 and W3 on those thumbnails.
3. Measured mean reflectance per class and band, from `eurosat_wsp/data/ms/train_x.npy` (divide by 10,000).
4. Fable's assumed prototypes per round, parsed from the W3 replies.
5. Confusion matrices, recalls and accuracies for every W2 and W3 round, test and practice.
6. W3-init results, including per-epoch validation curves.
7. The pilot READ correlation matrix and fingerprints, with the anonymous order and the seed order.
8. The EDIT layer as tile images for original, corrupted and repaired; predictions and confusion matrices for each.
9. Class scores of one example image from the W3 network, for the temperature demonstration.

## Work split for subagents

| Agent | Delivers | Depends on |
|---|---|---|
| **Data** | `site/build_data.py` and `site/data/` (list above) | the repository |
| **Design system** | One stylesheet: colours (take the palette in `eurosat_wsp/generate_figures.py`: blue `#2a78d6` for Fable, orange `#eb6834` for baselines and RGB, ink and grid colours), type scale, figure frame, caption style, light and dark mode | none |
| **Part 1 visuals**, **Part 2 visuals**, **Part 3 visuals** | One self-contained component per visual, reading only from `site/data/` | Data, Design system |
| **Writer** | The prose: short sentences, one idea each, every number taken from the data files, each section ending with what it does not show | the logs and `README.md` |
| **Integration and fact-check** | Assembles the page, tests it on a phone-width screen, checks every number in the text against `site/data/`, checks every link | all of the above |

## Rules

- **Every number comes from a file.** If a number in the text cannot be traced to `site/data/` or a result file, it is removed.
- **Say who did what.** The pilot (READ, EDIT, WRITE) is Claude Fable 5.1 with tools, on 10 November 2026. W1 to W3 and W3-init are Claude Fable 5.2 without tools, on 11 November 2026. Repair baselines, multi-seed training and all figures were added afterwards by the authors, with the results known.
- **Keep the caveats visible**, next to the claims they qualify: W1 is a single attempt; Fable used remembered knowledge of EuroSAT; the 13-band gain is partly in the data; rescaling was our correction, decided before the runs; three seeds; variant A has an easy baseline (zeroing gives 94.1%).
- **No hype words** ("breakthrough", "magic", "surgical"). Let the visuals carry the surprise.
- **Mobile first.** Most readers will arrive from a phone.
- **Static site.** Precomputed JSON plus client-side rendering; no server.

## Shipping

- **Site.** Static HTML, CSS and JavaScript in `site/`, published with GitHub Pages through a GitHub Actions workflow.
- **Social preview.** An `og:image` (the hook visual or `assets/fig7_write_init.png`) so links show a picture on LinkedIn.
- **Launch post.** A short LinkedIn post with a looping clip of the confusion matrix evolving and the link.

## Open decisions (owner: authors)

1. **Address:** a personal site (`<user>.github.io`, new repository) or a page of this repository.
2. **Byline and affiliation** for the article and a citation entry.
3. **Licence** for the code and the article.
