# Experiment W1: write a whole classifier, no tools, no training

Can a language model write **every** weight of a working EuroSAT classifier from scratch, in one reply, with no tools, no data and no training? This is the EuroSAT counterpart of Takehi et al.'s Experiment 5. The design, baselines and success criteria were fixed beforehand in [docs/next_steps.md](../../docs/next_steps.md).

## The network (618 weights)

```text
image 64x64 RGB
-> 16 filters of 3x3 over R, G, B (432 weights + 16 biases)
-> ReLU
-> average each filter's response over the whole image (16 numbers per image)
-> linear layer 16 -> 10 classes (160 weights + 10 biases)
-> predicted class = highest score
```

The model has to decide what each of the 16 filters should detect (a colour, an edge, a texture) and how each class should weigh those 16 detections. It is a tiny, readable program written in weights.

## How to run an attempt

**1. Open a new chat with Claude Fable 5.2, with tools off.** In claude.ai, start a new conversation, select Fable 5.2 (the version used for attempt 1), and switch off code execution (the analysis tool) and web search. The model must not be able to run code. Use a new chat for every attempt.

**2. Paste the whole prompt.**
- W1, attempt 1 and 2: `prompts/prompt_w1.txt`
- W1b, attempt 1 and 2: `prompts/prompt_w1b.txt` (the same, plus each class's average colour)

**3. Save the complete reply** (explanation and JSON) as a text file:

```text
runs/w1_attempt_1/response.txt
runs/w1_attempt_2/response.txt
runs/w1b_attempt_1/response.txt
runs/w1b_attempt_2/response.txt
```

Save it exactly as received, without editing. If the reply is cut off or contains no valid JSON, that attempt counts as a failure; do not ask the model to fix it.

**4. Parse and evaluate.** From the repository root (the paths after `--input` and `--weights` are relative to this folder):

```bash
python eurosat_wsp/experiment_4_write_full/w1.py parse --input runs/w1_attempt_1/response.txt
python eurosat_wsp/experiment_4_write_full/w1.py eval  --weights runs/w1_attempt_1/weights.pt
```

`eval` prints the test accuracy, balanced accuracy, recall per class, the baselines, and whether the pre-registered success criteria were met. It saves everything to `result.json` next to the weights.

## Baselines

Computed with `w1.py baselines` and kept out of git until all four attempts are recorded, so that nothing about the targets is public before the model answers.

| Baseline | Meaning |
|---|---|
| Best constant guess (11.3%) | Always predicting the largest test class. The real floor. |
| Kaiming initialisation | Random weights of the same network. |
| Mean-RGB nearest centroid | A trivial classifier on average image colour, fitted on training data. |
| Trained ceiling | The same network trained with AdamW, best of 30 epochs, 3 seeds. |

## Success criteria (pre-registered)

- **Weak success:** accuracy significantly above the best constant guess (binomial p < 0.01) on both attempts.
- **Strong success:** accuracy above the mean-RGB centroid on both attempts.

All four attempts are reported, whatever the outcome.
