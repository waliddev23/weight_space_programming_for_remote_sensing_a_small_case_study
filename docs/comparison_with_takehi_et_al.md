# How this project relates to Takehi et al. (2026)

This repository borrows the read, edit and write structure of Takehi, Teoh, Zhang and Khattab, [*Sparks of Weight-Space Programming*](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, November 2026). It is **not a replication** of their study. The protocols differ in ways that change what the results can show. This page states those differences so that no result here is read as confirming theirs.

## What they claim

A general-purpose language model can read and write neural network weights directly, guided only by a natural-language goal. In their strongest experiments, the model receives the weights as text and answers in a single reply, with **no tools**, no data, no gradients and no running of the network:

- **Read.** It names the game a policy was trained on, without being given the options.
- **Fix.** It repairs a policy that fails at a control task, by adding a small circuit to the weights.
- **Write.** It writes **every** weight of a game-playing network from scratch. That network, evaluated with no training, beats PPO trained for 100 million steps.

Their later experiments (6 to 8) add a Python sandbox and, in some settings, a small budget of forward passes.

## What this project did

Claude Fable 5.1 ran three experiments on EuroSAT satellite imagery inside Claude Code, with a Python environment available throughout.

| | Takehi et al. | This project | Consequence |
|---|---|---|---|
| **Tools** | None in experiments 1, 2, 4 and 5; a Python sandbox in 3 and 6 to 8. | Python throughout: correlations, SVD, generator scripts. | Comparable to their tool-assisted experiments only, not to their headline results. |
| **Read: question** | Open-ended ("which game?"), options not given. | Three named training regimes, three models each. | A constrained classification, easier than open-ended recognition. |
| **Read: shortcut** | Each network trained independently. | Models trained in groups sharing a random seed. | Fable found and exploited this structure. Without it the task is harder, and untested. |
| **Edit: what is broken** | A network that **behaves** wrongly (a policy stuck in a local minimum, a model that loops). | Random noise added to 32 filters of one layer. | Ours is a statistics problem. Fable solved it statistically, without reasoning about what the filters compute. |
| **Write: how much** | The **entire** network. | The first layer only: 432 of 98,954 weights. | Most of the network came from elsewhere. |
| **Write: evaluation** | Used as written, with **no training**. | The full network was then trained for 20 epochs. | Gradient descent did most of the work. This is not weight-space programming in their sense. |
| **Logs** | Full logs released. | Full session record and a reasoning log in `eurosat_wsp/runs/`. | Same standard. |

## What the results do and do not support

**Read (9 of 9 correct).** This is genuine weight-space reading: Fable inferred how each network was trained from its weights alone, without running any of them. It is closest to their Experiment 3, which also allowed Python on the weights. It does not test their no-tools claim, and the shared seeds made the problem easier than an open-ended question would be.

**Edit (95.5% test accuracy).** Fable's contribution was **diagnosis**. It found the damaged filters, showed they could not be reconstructed, and rejected the suggested methods, one of which would have made the network worse (45.8%). The edit itself replaces each damaged filter with the average healthy filter. Setting them to zero gives 94.1%. This shows good judgment about weights, not an understanding of what the network computes.

**Write (90.81% against 90.35% over three seeds).** This shows that domain knowledge can be written into a first layer without harming training, and possibly helping slightly (not significant, p ≈ 0.12). It does **not** test whether a model can write a working network, because training followed.

## How to describe this work

Accurate: *"A tool-assisted pilot study of weight-space reading and diagnosis on remote-sensing CNNs, inspired by Takehi et al."*

Not accurate: *"A replication of Takehi et al. on satellite imagery"*, or *"Fable wrote a satellite classifier by hand."*

## Follow-up experiments in their terms

Three follow-up experiments (W1 to W3, run with Claude Fable 5.2 in claude.ai with tools off) test the write claim in Takehi et al.'s own terms: whole networks written with no tools, no data and no training, evaluated exactly as written. They reached 49.4% (RGB, typed by hand), 49.2% (RGB, written as a program, after three feedback rounds) and 59.5% (13-band multispectral, after three rounds), against trivial colour or spectrum classifiers at 30.4% and 48.6%. Used as a training starting point, the multispectral network beat standard initialisation at every label budget once its classifier was rescaled. The read and edit claims have not yet been tested without tools. Designs, criteria and results are in [next_steps.md](next_steps.md) and the README.
