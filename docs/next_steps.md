# Next steps: experiments in Takehi et al.'s own terms

The pilot ([comparison](comparison_with_takehi_et_al.md)) showed that Fable 5.1 can analyse and diagnose weights with tools. It did not test the central claim of Takehi et al.: that a model can read and write working networks **with no tools, no data and no training**. The three experiments below test that claim directly on EuroSAT.

Each experiment has its success criteria and baselines fixed here, **before it is run**. Results will be reported against these criteria whatever they turn out to be, negative results included.

## Rules that apply to every experiment

1. **No tools means no tools.** The model is called through a plain chat or API request with tool use disabled, not inside Claude Code. It answers in a single reply.
2. **Same model.** Claude Fable 5.1, so results connect to the pilot.
3. **Two attempts per task,** both reported, as in Takehi et al.
4. **Keys stay outside the repository** until all answers are recorded. Answer keys, original checkpoints and trained ceilings are committed only afterwards.
5. **Full logs.** The exact prompt, the full response and the evaluation output are saved for every attempt.
6. **Every result is compared with a trivial baseline,** so that an apparent success can't come from the task being easy. The pilot showed why: zeroing broken filters already recovered most of the EDIT accuracy.

## Experiment W1 · Write a whole classifier from scratch

The most important of the three, and the closest match to Takehi et al.'s Experiment 5.

**Network.** Small enough to fit in a prompt and to be written by hand: 618 weights.

```text
input 64x64 RGB (ImageNet-normalised)
conv 3x3, 3 -> 16 channels, with bias      448 weights
ReLU
global average pooling                     16 features per image
linear 16 -> 10 classes, with bias          170 weights
```

**Task.** Fable writes all 618 weights in one reply. It receives the architecture, the input normalisation, the ten class names and a description of EuroSAT (Sentinel-2 RGB, 64×64 pixels, 10 m resolution). It receives no images, no statistics computed from the data and no feedback.

**Evaluation.** The network exactly as written, on the 5,400 test images, with no training.

**Baselines.**

| Baseline | Why it is there |
|---|---|
| Chance (10%) | The floor. |
| The same network with Kaiming initialisation | What random weights of this shape score. |
| Nearest class centroid on mean RGB, fitted on training data | A trivial colour classifier. If Fable doesn't beat it, its network adds nothing beyond average colour. |
| The same network trained with Adam (3 seeds) | The ceiling for this architecture. |

**Success criteria.**

- **Weak success:** test accuracy significantly above chance (binomial test, p < 0.01) on both attempts.
- **Strong success:** above the mean-RGB centroid baseline on both attempts.
- **Report regardless:** per-class accuracy and the confusion matrix, to see which classes the hand-written colour and texture logic actually separates.

**Relaxed variant W1b,** reported separately: Fable may also see per-class mean and standard deviation of R, G and B from the training set, a few numbers of "documentation". This tests whether the bottleneck is domain knowledge or the translation of knowledge into weights.

## Experiment R2 · Read what a network was trained to do

Matches Takehi et al.'s Experiment 1: open-ended, no options, no tools.

**Networks.** Six SmallCNNs, **each with its own random seed**, removing the shortcut from the pilot. Each is trained on a different binary task, one class or group against the rest:

| Network | Positive class |
|---|---|
| 1 | Water (River + SeaLake) |
| 2 | Forest |
| 3 | Built-up (Residential + Industrial) |
| 4 | Highway |
| 5 | Cropland (Annual Crop + Permanent Crop) |
| 6 | Pasture |

**Task.** For each network separately, Fable receives the first-layer and classifier weights as text, about 1,700 numbers, plus the architecture and a statement that it is a binary EuroSAT detector. It does not receive the list of tasks. Question: *"Which land-cover type does this network detect? Give your answer, your confidence and your reasoning."*

**Scoring.** Fixed in advance by an answer rubric. Correct: the class or an obvious synonym ("water" for network 1). Partly correct: the right family ("vegetation" for forest). Wrong: anything else. Chance for a free answer is hard to define, so a second scoring compares the answer against the six options after the fact, where chance is 1 in 6.

**Success criteria.** At least 4 of 6 correct on each attempt. Every reasoning trace is checked for whether it cites real features of the weights, such as the colour pattern of the first layer, or only guesses.

**Additional variant R2b.** The pilot's three-regime task, rerun with no tools and nine independent seeds. This measures how much of the pilot's 9 of 9 depended on Python and on the seed shortcut.

## Experiment E2 · Fix a network that behaves wrongly

Matches Takehi et al.'s Experiment 4: a behavioural flaw, fixed by editing weights, no data.

**Flawed network.** A SmallCNN trained on data with a planted shortcut: in the training set, River images are given a slight blue tint that the test set does not have. The network learns "blue tint means river", and on the clean test set it confuses rivers with highways and lakes. The flaw is confirmed before the experiment: River recall must fall at least 20 points below a cleanly trained network.

**Task.** Fable receives the weights, the architecture, a description of the symptom and the test confusion matrix, playing the role of the "recorded episode" in Takehi et al. It does not receive the cause. It must explain the failure and return an edited set of weights, in one reply, with no tools.

**Baselines.**

| Baseline | Why it is there |
|---|---|
| The flawed network | The starting point. |
| Raising the River output bias until River recall matches the clean network | The trivial fix. It rebalances the output without touching the cause. |
| Retraining only the classifier on clean data | The ceiling for a last-layer fix, with data. |
| A cleanly trained network | The ceiling overall. |

**Success criteria.** River recall improves by at least half the gap to the clean network, overall accuracy does not fall, and the result beats the bias-only fix. The explanation must name the cause (colour, the blue channel) to count as a diagnosis.

## Order and effort

| Step | Work | Model calls |
|---|---|---|
| 1. W1 harness: network, prompt, parser, evaluator, baselines | Small; no training except the 3 ceiling runs of a tiny network | 2 attempts (+2 for W1b) |
| 2. R2: train 6 binary networks (and 9 for R2b), build prompts and rubric | Moderate; 15 short training runs | 12 attempts (+18 for R2b) |
| 3. E2: build the flawed network, confirm the flaw, baselines | Moderate; 2 training runs and 1 head retraining | 2 attempts |

W1 comes first. It is the cheapest and the most decisive. A clearly-above-baseline score would be a result in Takehi et al.'s own terms on a new domain, and a failure would be an honest negative result that is just as worth reporting.
