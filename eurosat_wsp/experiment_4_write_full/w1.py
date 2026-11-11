"""Experiment W1: a language model writes every weight of a small EuroSAT classifier.

No tools, no data, no training. The network is evaluated exactly as written.
Pre-registered design: docs/next_steps.md (Experiment W1).

    python eurosat_wsp/experiment_4_write_full/w1.py prompt
    python eurosat_wsp/experiment_4_write_full/w1.py baselines
    python eurosat_wsp/experiment_4_write_full/w1.py parse --input runs/attempt_1/response.txt
    python eurosat_wsp/experiment_4_write_full/w1.py eval  --weights runs/attempt_1/weights.pt

Paths given to parse/eval are relative to this folder unless absolute.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
WSP_ROOT = HERE.parent
if str(WSP_ROOT) not in sys.path:
    sys.path.insert(0, str(WSP_ROOT))

import common as wsp
from wsp_parser import extract_code_block

PROMPTS = HERE / "prompts"
BASELINES = HERE / "baselines"          # git-ignored until all attempts are recorded
SHAPES = {"conv_weight": (16, 3, 3, 3), "conv_bias": (16,), "fc_weight": (10, 16), "fc_bias": (10,)}


# ── the network ──────────────────────────────────────────────────────────────
class TinyNet(nn.Module):
    """conv 3x3 (3->16, bias, padding 1) -> ReLU -> global average pool -> linear 16->10."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=True)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        h = F.relu(self.conv(x))
        return self.fc(h.mean(dim=(2, 3)))

    def load_written(self, w: dict) -> "TinyNet":
        with torch.no_grad():
            self.conv.weight.copy_(w["conv_weight"])
            self.conv.bias.copy_(w["conv_bias"])
            self.fc.weight.copy_(w["fc_weight"])
            self.fc.bias.copy_(w["fc_bias"])
        return self


# ── data: normalised tensors held in memory ──────────────────────────────────
def load_split(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    raw = wsp.load_raw()
    x, y = wsp.preload_split(raw[split])                       # uint8 N,64,64,3
    x = torch.from_numpy(x).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(wsp.MEAN).view(1, 3, 1, 1)
    std = torch.tensor(wsp.STD).view(1, 3, 1, 1)
    return (x - mean) / std, torch.from_numpy(y)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device) -> torch.Tensor:
    model.to(device).eval()
    out = [model(x[i:i + 512].to(device)).argmax(1).cpu() for i in range(0, len(x), 512)]
    return torch.cat(out)


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from scipy.stats import binomtest
    acc = float((y_true == y_pred).mean())
    k = int((y_true == y_pred).sum())
    recall = {wsp.CLASS_NAMES[c]: float((y_pred[y_true == c] == c).mean()) for c in range(10)}
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    majority = float(np.bincount(y_true, minlength=10).max() / len(y_true))   # best constant guess
    return {"test_accuracy": acc, "correct": k, "n": int(len(y_true)),
            "balanced_accuracy": float(np.mean(list(recall.values()))),
            "majority_class_rate": majority,
            "p_value_vs_majority": float(binomtest(k, len(y_true), majority, alternative="greater").pvalue),
            "per_class_recall": recall, "confusion_matrix": cm.tolist(),
            "predicted_class_counts": np.bincount(y_pred, minlength=10).tolist()}


# ── prompts ──────────────────────────────────────────────────────────────────
PROMPT = """You are an expert in remote sensing and neural networks.

# Task
Write, from scratch, every weight of a small neural network that classifies EuroSAT satellite images into 10 land-cover classes, as accurately as possible.

There is no training, no data, no feedback and no tools. You choose every number yourself, in one reply. The network will be evaluated exactly as you write it, on the EuroSAT test set.

# Data
EuroSAT RGB: Sentinel-2 satellite image patches over Europe, 64 x 64 pixels at 10 m per pixel, true colour (red, green, blue). Images are seen from directly above.

Classes (output index: name):
{classes}

# Input
Each image is a tensor of shape [3, 64, 64], channels in the order R, G, B. Pixel values are scaled to [0, 1] and then normalised per channel:
  x_c = (pixel_c - mean_c) / std_c
  mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)   for (R, G, B)
So a pixel with the average ImageNet colour has value 0 in every channel.
{extra}
# Network (618 weights)
  h      = conv2d(x, conv_weight, conv_bias, kernel 3x3, stride 1, padding 1)   # [16, 64, 64]
  h      = relu(h)
  f      = mean of h over all 64 x 64 positions                                  # [16]
  logits = fc_weight @ f + fc_bias                                                # [10]
  prediction = argmax(logits)

Shapes:
  conv_weight [16, 3, 3, 3]   (output channel, input channel R/G/B, row, column)
  conv_bias   [16]
  fc_weight   [10, 16]        (class, feature)
  fc_bias     [10]

# Output format
Explain your design briefly, then give all weights in a single ```json code block:

```json
{{
  "conv_weight": [[[[...3 numbers...], ...3 rows...], ...3 channels...], ...16 filters...],
  "conv_bias": [...16 numbers...],
  "fc_weight": [[...16 numbers...], ...10 rows...],
  "fc_bias": [...10 numbers...]
}}
```
Every number must be written out. Do not use ellipses, expressions or code.
"""


def class_colour_stats() -> str:
    x, y = load_split("train")
    lines = ["", "# Documentation: average colour per class (training set, normalised units as above)",
             "class: mean R, G, B | std R, G, B (std across pixels of all images of that class)"]
    for c, name in enumerate(wsp.CLASS_NAMES):
        xc = x[y == c]
        m = xc.mean(dim=(0, 2, 3)).tolist()
        s = xc.std(dim=(0, 2, 3)).tolist()
        lines.append(f"  {c} {name}: {m[0]:+.2f} {m[1]:+.2f} {m[2]:+.2f} | {s[0]:.2f} {s[1]:.2f} {s[2]:.2f}")
    return "\n".join(lines) + "\n"


def cmd_prompt(_args) -> None:
    PROMPTS.mkdir(parents=True, exist_ok=True)
    classes = "\n".join(f"  {i}: {n}" for i, n in enumerate(wsp.CLASS_NAMES))
    (PROMPTS / "prompt_w1.txt").write_text(PROMPT.format(classes=classes, extra=""), encoding="utf-8")
    (PROMPTS / "prompt_w1b.txt").write_text(PROMPT.format(classes=classes, extra=class_colour_stats()),
                                           encoding="utf-8")
    print(f"wrote {PROMPTS / 'prompt_w1.txt'} and prompt_w1b.txt")


# ── parse ────────────────────────────────────────────────────────────────────
def here(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else HERE / p


def cmd_parse(args) -> None:
    src = here(args.input)
    data = json.loads(extract_code_block(src.read_text(encoding="utf-8"), "json"))
    w = {}
    for key, shape in SHAPES.items():
        t = torch.tensor(data[key], dtype=torch.float32)
        if tuple(t.shape) != shape:
            raise SystemExit(f"{key}: expected shape {shape}, got {tuple(t.shape)}")
        if not torch.isfinite(t).all():
            raise SystemExit(f"{key}: contains NaN or Inf")
        w[key] = t
    out = here(args.save) if args.save else src.with_name("weights.pt")
    torch.save(w, out)
    n = sum(t.numel() for t in w.values())
    print(f"parsed {n} weights -> {out}")


# ── evaluate ─────────────────────────────────────────────────────────────────
def cmd_eval(args) -> None:
    path = here(args.weights)
    w = torch.load(path, map_location="cpu", weights_only=True)
    x, y = load_split("test")
    pred = predict(TinyNet().load_written(w), x, wsp.get_device())
    res = report(y.numpy(), pred.numpy())
    res["weights"] = str(path.relative_to(HERE)) if path.is_relative_to(HERE) else str(path)
    res["evaluated_at"] = wsp.now()
    out = path.with_name("result.json")
    wsp.write_json(res, out)

    print(f"\nTest accuracy:     {res['test_accuracy'] * 100:.2f}%  ({res['correct']}/{res['n']})")
    print(f"Balanced accuracy: {res['balanced_accuracy'] * 100:.2f}%")
    print(f"p vs best constant guess ({res['majority_class_rate'] * 100:.2f}%): "
          f"{res['p_value_vs_majority']:.2g}")
    print("Recall per class:")
    for name, r in res["per_class_recall"].items():
        print(f"  {name:24s} {r * 100:5.1f}%")
    b = BASELINES / "baselines.json"
    if b.exists():
        base = json.loads(b.read_text(encoding="utf-8"))
        print("\nBaselines:")
        print(f"  best constant guess        {res['majority_class_rate'] * 100:.2f}%")
        print(f"  Kaiming init (3 seeds)     {base['kaiming_mean'] * 100:.2f}%")
        print(f"  mean-RGB centroid          {base['centroid']['test_accuracy'] * 100:.2f}%")
        print(f"  trained ceiling (3 seeds)  {base['trained_mean'] * 100:.2f}%")
        weak = res["p_value_vs_majority"] < 0.01
        strong = res["test_accuracy"] > base["centroid"]["test_accuracy"]
        print(f"\nPre-registered criteria for this attempt: weak success {'MET' if weak else 'NOT met'}, "
              f"strong success {'MET' if strong else 'NOT met'}")
    print(f"saved {out}")


# ── baselines ────────────────────────────────────────────────────────────────
def cmd_baselines(args) -> None:
    device = wsp.get_device()
    xtr, ytr = load_split("train")
    xva, yva = load_split("validation")
    xte, yte = load_split("test")
    out = {}

    kaiming = []
    for seed in range(3):
        torch.manual_seed(seed)
        net = TinyNet()
        nn.init.kaiming_normal_(net.conv.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(net.conv.bias)
        kaiming.append(report(yte.numpy(), predict(net, xte, device).numpy())["test_accuracy"])
    out["kaiming"] = kaiming
    out["kaiming_mean"] = float(np.mean(kaiming))
    print(f"Kaiming init: {[f'{a:.4f}' for a in kaiming]}")

    # nearest class centroid on per-image mean colour
    ftr, fte = xtr.mean(dim=(2, 3)), xte.mean(dim=(2, 3))
    cent = torch.stack([ftr[ytr == c].mean(0) for c in range(10)])
    pc = torch.cdist(fte, cent).argmin(1)
    out["centroid"] = report(yte.numpy(), pc.numpy())
    print(f"mean-RGB centroid: {out['centroid']['test_accuracy']:.4f}")

    trained = []
    for seed in range(3):
        torch.manual_seed(seed)
        net = TinyNet().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        g = torch.Generator().manual_seed(seed)
        best_val, best_state = -1.0, None
        for epoch in range(args.epochs):
            net.train()
            for idx in torch.randperm(len(xtr), generator=g).split(128):
                loss = F.cross_entropy(net(xtr[idx].to(device)), ytr[idx].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            va = float((predict(net, xva, device) == yva).float().mean())
            if va > best_val:
                best_val, best_state = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(best_state)
        r = report(yte.numpy(), predict(net, xte, device).numpy())
        trained.append(r["test_accuracy"])
        print(f"trained seed {seed}: val {best_val:.4f} test {r['test_accuracy']:.4f}")
    out["trained"] = trained
    out["trained_mean"] = float(np.mean(trained))
    out["trained_setup"] = {"optimizer": "AdamW", "lr": 3e-3, "weight_decay": 1e-4, "batch": 128,
                            "epochs": args.epochs, "schedule": "cosine", "augmentation": "none",
                            "selection": "best validation accuracy"}
    out["created_at"] = wsp.now()
    wsp.write_json(out, BASELINES / "baselines.json")
    print(f"saved {BASELINES / 'baselines.json'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prompt", help="write prompt_w1.txt and prompt_w1b.txt")
    p = sub.add_parser("parse", help="parse a model response into weights.pt")
    p.add_argument("--input", required=True)
    p.add_argument("--save")
    e = sub.add_parser("eval", help="evaluate written weights on the test set, no training")
    e.add_argument("--weights", required=True)
    b = sub.add_parser("baselines", help="chance, Kaiming, mean-RGB centroid, trained ceiling")
    b.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    {"prompt": cmd_prompt, "parse": cmd_parse, "eval": cmd_eval, "baselines": cmd_baselines}[args.cmd](args)


if __name__ == "__main__":
    main()
