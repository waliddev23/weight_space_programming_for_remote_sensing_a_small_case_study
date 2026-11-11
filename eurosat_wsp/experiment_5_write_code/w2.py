"""Experiment W2: the model writes a Python program that builds every weight of a
two-layer EuroSAT CNN, with no data and no training, then gets up to 3 rounds of
feedback on a small practice set. Takehi et al.'s Experiment 6, scaled down.

    python eurosat_wsp/experiment_5_write_code/w2.py prompt
    python eurosat_wsp/experiment_5_write_code/w2.py baselines
    python eurosat_wsp/experiment_5_write_code/w2.py run --round 0     # after saving the reply
    python eurosat_wsp/experiment_5_write_code/w2.py summary           # after the last round

For round k, save the model's full reply as runs/round_k/response.txt. `run` extracts
the code, executes it in a separate isolated Python process, evaluates the weights,
prints the practice feedback to paste back, and stores the test result privately.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
WSP_ROOT = HERE.parent
for p in (WSP_ROOT, WSP_ROOT / "experiment_4_write_full"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common as wsp
from w1 import load_split, predict, report
from wsp_parser import extract_code_block

RUNS = HERE / "runs"
PRIVATE = HERE / "private"            # test results and baselines, git-ignored until the end
MAX_ROUND = 3
SHAPES = {"conv1_weight": (16, 3, 3, 3), "conv1_bias": (16,),
          "conv2_weight": (32, 16, 3, 3), "conv2_bias": (32,),
          "fc_weight": (10, 32), "fc_bias": (10,)}
PRACTICE_PER_CLASS = 100


class SmallNet(nn.Module):
    """conv 3->16 -> ReLU -> avgpool 4 -> conv 16->32 -> ReLU -> global mean -> linear 32->10."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        h = F.avg_pool2d(F.relu(self.conv1(x)), 4)
        h = F.relu(self.conv2(h))
        return self.fc(h.mean(dim=(2, 3)))

    def load_written(self, w: dict) -> "SmallNet":
        with torch.no_grad():
            for name in ("conv1", "conv2", "fc"):
                getattr(self, name).weight.copy_(torch.as_tensor(w[f"{name}_weight"], dtype=torch.float32))
                getattr(self, name).bias.copy_(torch.as_tensor(w[f"{name}_bias"], dtype=torch.float32))
        return self


def practice_set():
    """100 validation images per class, fixed. The test set is never shown to the model."""
    x, y = load_split("validation")
    g = np.random.default_rng(0)
    idx = np.concatenate([g.choice(np.where(y.numpy() == c)[0], PRACTICE_PER_CLASS, replace=False)
                          for c in range(10)])
    return x[idx], y[idx]


# ── prompt ───────────────────────────────────────────────────────────────────
PROMPT = """You are an expert in remote sensing and neural networks.

# Task
Write a Python function that builds every weight of a small convolutional network that classifies EuroSAT satellite images into 10 land-cover classes, as accurately as possible. There is no training and no data. Your code cannot see any images and cannot run the network.

# Data
EuroSAT RGB: Sentinel-2 satellite image patches over Europe, 64 x 64 pixels at 10 m per pixel, true colour, seen from directly above.

Classes (output index: name):
{classes}

# Input
Each image is a tensor [3, 64, 64], channels R, G, B. Pixels are scaled to [0, 1] and normalised per channel:
  x_c = (pixel_c - mean_c) / std_c,  mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)

# Network (5,418 weights)
  h = relu(conv2d(x, conv1_weight, conv1_bias, 3x3, stride 1, padding 1))   # [16, 64, 64]
  h = avg_pool2d(h, 4)                                                         # [16, 16, 16]
  h = relu(conv2d(h, conv2_weight, conv2_bias, 3x3, stride 1, padding 1))    # [32, 16, 16]
  f = mean of h over all 16 x 16 positions                                     # [32]
  logits = fc_weight @ f + fc_bias                                             # [10]
  prediction = argmax(logits)

Shapes: conv1_weight [16,3,3,3], conv1_bias [16], conv2_weight [32,16,3,3], conv2_bias [32], fc_weight [10,32], fc_bias [10].
Convolution weights are indexed (output channel, input channel, row, column).

# What to return
One ```python code block defining build_weights() that returns a dict with exactly those six keys, each a numpy array of the given shape. Only numpy and math may be imported. The code must not read files, access the network or use randomness. Keep the code short; explain the design in a few sentences before it.

# Feedback
After your first answer you may receive up to {rounds} rounds of feedback. Each reports your network's accuracy, recall per class and confusion matrix on a fixed practice set of {n_practice} validation images. You may then send a revised build_weights(). The final evaluation uses a separate test set that you never see.
"""


def cmd_prompt(_args) -> None:
    classes = "\n".join(f"  {i}: {n}" for i, n in enumerate(wsp.CLASS_NAMES))
    text = PROMPT.format(classes=classes, rounds=MAX_ROUND, n_practice=10 * PRACTICE_PER_CLASS)
    (HERE / "prompts").mkdir(exist_ok=True)
    (HERE / "prompts" / "prompt_w2.txt").write_text(text, encoding="utf-8")
    print(f"wrote {HERE / 'prompts' / 'prompt_w2.txt'}")


# ── run one round ────────────────────────────────────────────────────────────
RUNNER = r"""
import json, sys, runpy
import numpy as np
ns = runpy.run_path(sys.argv[1])
w = ns["build_weights"]()
np.savez(sys.argv[2], **{k: np.asarray(v, dtype=np.float32) for k, v in w.items()})
"""
FORBIDDEN = ("open(", "import os", "import sys", "subprocess", "socket", "urllib", "requests",
             "torch", "datasets", "np.random", "random.", "__import__", "eval(", "exec(")


def build(round_dir: Path) -> dict:
    code = extract_code_block((round_dir / "response.txt").read_text(encoding="utf-8"), "python")
    hits = [f for f in FORBIDDEN if f in code]
    if hits:
        raise SystemExit(f"code uses forbidden constructs {hits}; the round counts as a failure")
    (round_dir / "build_weights.py").write_text(code, encoding="utf-8")
    out = round_dir / "weights.npz"
    proc = subprocess.run([sys.executable, "-I", "-c", RUNNER, str(round_dir / "build_weights.py"), str(out)],
                          capture_output=True, text=True, timeout=120, cwd=round_dir)
    if proc.returncode != 0:
        raise SystemExit("build_weights() failed:\n" + proc.stderr[-2000:])
    w = dict(np.load(out))
    for k, shape in SHAPES.items():
        if k not in w or w[k].shape != shape or not np.isfinite(w[k]).all():
            raise SystemExit(f"{k}: expected finite array of shape {shape}, got "
                             f"{None if k not in w else w[k].shape}")
    return w


def feedback_text(r: dict, rnd: int) -> str:
    names = [n[:10] for n in wsp.CLASS_NAMES]
    lines = [f"Practice set results (round {rnd}, {r['n']} validation images, 100 per class):",
             f"accuracy {r['test_accuracy'] * 100:.1f}%",
             "recall per class: " + ", ".join(f"{i} {v * 100:.0f}%" for i, v in
                                              enumerate(r["per_class_recall"].values())),
             "confusion matrix (rows = true class 0-9, columns = predicted 0-9):"]
    lines += [f"  {i}: " + " ".join(f"{v:3d}" for v in row) for i, row in enumerate(r["confusion_matrix"])]
    remaining = MAX_ROUND - rnd
    lines.append(f"You may send a revised build_weights() ({remaining} feedback round"
                 f"{'s' if remaining != 1 else ''} left)." if remaining > 0 else "This was the last round.")
    return "\n".join(lines)


def cmd_run(args) -> None:
    rd = RUNS / f"round_{args.round}"
    w = build(rd)
    device = wsp.get_device()
    net = SmallNet().load_written(w)
    xp, yp = practice_set()
    rp = report(yp.numpy(), predict(net, xp, device).numpy())
    xt, yt = load_split("test")
    rt = report(yt.numpy(), predict(net, xt, device).numpy())
    wsp.write_json({"round": args.round, "practice": rp, "test": rt, "evaluated_at": wsp.now()},
                   PRIVATE / f"round_{args.round}.json")
    fb = feedback_text(rp, args.round)
    (rd / "feedback.txt").write_text(fb + "\n", encoding="utf-8")
    print(fb)
    print(f"\n(test result stored privately in {PRIVATE.name}/round_{args.round}.json)")


def cmd_summary(_args) -> None:
    base = json.loads((PRIVATE / "baselines.json").read_text(encoding="utf-8"))
    print("round  practice  test   test balanced")
    for f in sorted(PRIVATE.glob("round_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        print(f"  {d['round']}     {d['practice']['test_accuracy'] * 100:5.1f}%  "
              f"{d['test']['test_accuracy'] * 100:5.1f}%  {d['test']['balanced_accuracy'] * 100:5.1f}%")
    print(f"\nKaiming {base['kaiming_mean'] * 100:.1f}% | trained ceiling {base['trained_mean'] * 100:.1f}% "
          f"| W1 attempt 1 (618 weights, by hand) 49.4%")


# ── baselines ────────────────────────────────────────────────────────────────
def cmd_baselines(args) -> None:
    device = wsp.get_device()
    xtr, ytr = load_split("train")
    xva, yva = load_split("validation")
    xte, yte = load_split("test")
    out = {"kaiming": [], "trained": []}
    for seed in range(3):
        torch.manual_seed(seed)
        net = SmallNet()
        for m in (net.conv1, net.conv2):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        out["kaiming"].append(report(yte.numpy(), predict(net, xte, device).numpy())["test_accuracy"])
    for seed in range(3):
        torch.manual_seed(seed)
        net = SmallNet().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        g = torch.Generator().manual_seed(seed)
        best, state = -1.0, None
        for _ in range(args.epochs):
            net.train()
            for idx in torch.randperm(len(xtr), generator=g).split(128):
                loss = F.cross_entropy(net(xtr[idx].to(device)), ytr[idx].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            va = float((predict(net, xva, device) == yva).float().mean())
            if va > best:
                best, state = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(state)
        acc = report(yte.numpy(), predict(net, xte, device).numpy())["test_accuracy"]
        out["trained"].append(acc)
        print(f"trained seed {seed}: test {acc:.4f}")
    out["kaiming_mean"] = float(np.mean(out["kaiming"]))
    out["trained_mean"] = float(np.mean(out["trained"]))
    out["created_at"] = wsp.now()
    wsp.write_json(out, PRIVATE / "baselines.json")
    print(f"Kaiming {out['kaiming']}, trained mean {out['trained_mean']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompt")
    r = sub.add_parser("run")
    r.add_argument("--round", type=int, required=True, choices=range(MAX_ROUND + 1))
    sub.add_parser("summary")
    b = sub.add_parser("baselines")
    b.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    {"prompt": cmd_prompt, "run": cmd_run, "summary": cmd_summary, "baselines": cmd_baselines}[args.cmd](args)


if __name__ == "__main__":
    main()
