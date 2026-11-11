"""Experiment W3: W2 repeated on 13-band multispectral EuroSAT.

Same network as W2 except that the first layer reads 13 Sentinel-2 bands instead of R, G, B.
Same rules, same practice set (1,000 validation images), same 3 feedback rounds. The images
are exactly the RGB experiments' train / validation / test images, matched by filename.

    python eurosat_wsp/experiment_6_write_multispectral/w3.py prepare     # once, after download
    python eurosat_wsp/experiment_6_write_multispectral/w3.py prompt
    python eurosat_wsp/experiment_6_write_multispectral/w3.py baselines
    python eurosat_wsp/experiment_6_write_multispectral/w3.py run --round 0
    python eurosat_wsp/experiment_6_write_multispectral/w3.py summary
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
for p in (WSP_ROOT, WSP_ROOT / "experiment_4_write_full", WSP_ROOT / "experiment_5_write_code"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import common as wsp
from w1 import report
from w2 import FORBIDDEN, MAX_ROUND, PRACTICE_PER_CLASS, RUNNER, feedback_text
from wsp_parser import extract_code_block

DATA = WSP_ROOT / "data" / "ms"           # aligned uint16 arrays, git-ignored (about 2.9 GB)
RUNS = HERE / "runs"
PRIVATE = HERE / "private"
MS_DATASET = "giswqs/EuroSAT_MS"
N_BANDS = 13
SCALE = 10000.0                           # Sentinel-2 L1C digital numbers -> top-of-atmosphere reflectance
SHAPES = {"conv1_weight": (16, N_BANDS, 3, 3), "conv1_bias": (16,),
          "conv2_weight": (32, 16, 3, 3), "conv2_bias": (32,),
          "fc_weight": (10, 32), "fc_bias": (10,)}
# EuroSAT stores the bands in this order (B8A last); confirmed against the data in `prepare`.
BANDS = [("B01", "coastal aerosol", 443, 60), ("B02", "blue", 490, 10), ("B03", "green", 560, 10),
         ("B04", "red", 665, 10), ("B05", "red edge 1", 705, 20), ("B06", "red edge 2", 740, 20),
         ("B07", "red edge 3", 783, 20), ("B08", "near infrared", 842, 10), ("B09", "water vapour", 945, 60),
         ("B10", "cirrus", 1375, 60), ("B11", "shortwave infrared 1", 1610, 20),
         ("B12", "shortwave infrared 2", 2190, 20), ("B8A", "narrow near infrared", 865, 20)]


class SmallNetMS(nn.Module):
    """W2's network with a 13-band first layer."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(N_BANDS, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        h = F.avg_pool2d(F.relu(self.conv1(x)), 4)
        h = F.relu(self.conv2(h))
        return self.fc(h.mean(dim=(2, 3)))

    def load_written(self, w):
        with torch.no_grad():
            for name in ("conv1", "conv2", "fc"):
                getattr(self, name).weight.copy_(torch.as_tensor(w[f"{name}_weight"], dtype=torch.float32))
                getattr(self, name).bias.copy_(torch.as_tensor(w[f"{name}_bias"], dtype=torch.float32))
        return self


# ── data ─────────────────────────────────────────────────────────────────────
def _stem(name: str) -> str:
    return Path(name).stem


def cmd_prepare(_args) -> None:
    """Re-split the multispectral images onto the RGB experiments' exact splits, by filename."""
    from datasets import load_dataset
    rgb = wsp.load_raw()
    ms = load_dataset(MS_DATASET).with_format("numpy")
    where = {}
    for split in ms:
        for i, f in enumerate(ms[split]["filename"]):
            where[_stem(f)] = (split, i)
    DATA.mkdir(parents=True, exist_ok=True)
    rgb_names = wsp.CLASS_NAMES
    for split in ("train", "validation", "test"):
        names = [_stem(f) for f in rgb[split]["filename"]]
        rgb_y = np.asarray(rgb[split]["label"])
        missing = [n for n in names if n not in where]
        if split == "validation":   # remember W2's practice images by name before anything is dropped
            g = np.random.default_rng(0)
            pidx = np.concatenate([g.choice(np.where(rgb_y == c)[0], PRACTICE_PER_CLASS, replace=False)
                                   for c in range(10)])
            practice_names = [names[i] for i in pidx]
        if missing and split == "test":
            raise SystemExit(f"test: {len(missing)} RGB images have no multispectral match, e.g. {missing[:3]}")
        if missing:   # train / validation only; drop and record them (the test set must be complete)
            keep = [n not in where for n in names]
            keep = ~np.array(keep)
            names = [n for n, k in zip(names, keep) if k]
            rgb_y = rgb_y[keep]
            wsp.write_json({"split": split, "dropped": missing}, DATA / f"{split}_dropped.json")
            print(f"{split}: dropped {len(missing)} image(s) with no multispectral version: {missing}")
        x = np.empty((len(names), N_BANDS, 64, 64), dtype=np.uint16)
        y = np.empty(len(names), dtype=np.int64)
        for j, n in enumerate(names):
            s, i = where[n]
            row = ms[s][int(i)]
            x[j] = row["image"]
            y[j] = int(row["label"])
        if not (y == rgb_y).all():
            raise SystemExit(f"{split}: labels differ between RGB and multispectral after matching")
        np.save(DATA / f"{split}_x.npy", x)
        np.save(DATA / f"{split}_y.npy", y)
        print(f"{split}: {len(y)} images matched by filename, labels identical to RGB")
        if split == "validation":
            pos = {n: i for i, n in enumerate(names)}
            kept = [pos[n] for n in practice_names if n in pos]
            lost = [n for n in practice_names if n not in pos]
            np.save(DATA / "practice_idx.npy", np.array(kept))
            print(f"practice set: {len(kept)} of W2's 1,000 practice images available"
                  + (f"; missing {lost}" if lost else ""))
    # band-order check: B8A (last) should track B08, B10 (cirrus) should be near zero over land
    x = np.load(DATA / "validation_x.npy", mmap_mode="r")[:500].astype(np.float32) / SCALE
    m = x.mean(axis=(0, 2, 3))
    corr = np.corrcoef(x[:, 7].ravel(), x[:, 12].ravel())[0, 1]
    print("mean reflectance per band:", " ".join(f"{b[0]}={v:.3f}" for b, v in zip(BANDS, m)))
    print(f"corr(B08, B8A) = {corr:.3f} (expect > 0.95); B10 cirrus mean = {m[9]:.4f} (expect < 0.01)")
    _ = rgb_names


def load_ms(split: str):
    return np.load(DATA / f"{split}_x.npy", mmap_mode="r"), torch.from_numpy(np.load(DATA / f"{split}_y.npy"))


def batch(x, idx, device):
    return torch.from_numpy(np.ascontiguousarray(x[idx]).astype(np.float32) / SCALE).to(device)


@torch.no_grad()
def predict(model, x, device, idx=None):
    model.to(device).eval()
    idx = np.arange(len(x)) if idx is None else idx
    return torch.cat([model(batch(x, idx[i:i + 256], device)).argmax(1).cpu() for i in range(0, len(idx), 256)])


def practice_idx(_y: torch.Tensor) -> np.ndarray:
    """W2's practice images, matched by filename (saved by `prepare`)."""
    return np.load(DATA / "practice_idx.npy")


# ── prompt ───────────────────────────────────────────────────────────────────
PROMPT = """You are an expert in remote sensing and neural networks.

# Task
Write a Python function that builds every weight of a small convolutional network that classifies EuroSAT multispectral satellite images into 10 land-cover classes, as accurately as possible. There is no training and no data. Your code cannot see any images and cannot run the network.

# Data
EuroSAT multispectral: Sentinel-2 Level-1C image patches over Europe, 64 x 64 pixels, all 13 spectral bands, seen from directly above. Bands at 20 m and 60 m resolution are resampled to the 10 m grid.

Classes (output index: name):
{classes}

# Input
Each image is a tensor [13, 64, 64]. Channel order and band (central wavelength, native resolution):
{bands}
Values are top-of-atmosphere reflectance: the Level-1C digital number divided by 10000, so most values lie between 0 and 1. No other normalisation is applied.

# Network (6,858 weights)
  h = relu(conv2d(x, conv1_weight, conv1_bias, 3x3, stride 1, padding 1))   # [16, 64, 64]
  h = avg_pool2d(h, 4)                                                         # [16, 16, 16]
  h = relu(conv2d(h, conv2_weight, conv2_bias, 3x3, stride 1, padding 1))    # [32, 16, 16]
  f = mean of h over all 16 x 16 positions                                     # [32]
  logits = fc_weight @ f + fc_bias                                             # [10]
  prediction = argmax(logits)

Shapes: conv1_weight [16,13,3,3], conv1_bias [16], conv2_weight [32,16,3,3], conv2_bias [32], fc_weight [10,32], fc_bias [10].
Convolution weights are indexed (output channel, input channel, row, column).

# What to return
One ```python code block defining build_weights() that returns a dict with exactly those six keys, each a numpy array of the given shape. Only numpy and math may be imported. The code must not read files, access the network or use randomness. Keep the code short; explain the design in a few sentences before it.

# Feedback
After your first answer you may receive up to {rounds} rounds of feedback. Each reports your network's accuracy, recall per class and confusion matrix on a fixed practice set of {n_practice} validation images. You may then send a revised build_weights(). The final evaluation uses a separate test set that you never see.
"""


def cmd_prompt(_args) -> None:
    classes = "\n".join(f"  {i}: {n}" for i, n in enumerate(wsp.CLASS_NAMES))
    bands = "\n".join(f"  {i:2d}: {b} {name} ({nm} nm, {res} m)" for i, (b, name, nm, res) in enumerate(BANDS))
    text = PROMPT.format(classes=classes, bands=bands, rounds=MAX_ROUND, n_practice=10 * PRACTICE_PER_CLASS)
    (HERE / "prompts").mkdir(exist_ok=True)
    (HERE / "prompts" / "prompt_w3.txt").write_text(text, encoding="utf-8")
    print(f"wrote {HERE / 'prompts' / 'prompt_w3.txt'}")


# ── one round ────────────────────────────────────────────────────────────────
def build(rd: Path) -> dict:
    code = extract_code_block((rd / "response.txt").read_text(encoding="utf-8"), "python")
    hits = [f for f in FORBIDDEN if f in code]
    if hits:
        raise SystemExit(f"code uses forbidden constructs {hits}; the round counts as a failure")
    (rd / "build_weights.py").write_text(code, encoding="utf-8")
    proc = subprocess.run([sys.executable, "-I", "-c", RUNNER, str(rd / "build_weights.py"), str(rd / "weights.npz")],
                          capture_output=True, text=True, timeout=120, cwd=rd)
    if proc.returncode != 0:
        raise SystemExit("build_weights() failed:\n" + proc.stderr[-2000:])
    w = dict(np.load(rd / "weights.npz"))
    for k, shape in SHAPES.items():
        if k not in w or w[k].shape != shape or not np.isfinite(w[k]).all():
            raise SystemExit(f"{k}: expected finite array of shape {shape}, got {None if k not in w else w[k].shape}")
    return w


def cmd_run(args) -> None:
    rd = RUNS / f"round_{args.round}"
    net = SmallNetMS().load_written(build(rd))
    device = wsp.get_device()
    xv, yv = load_ms("validation")
    pi = practice_idx(yv)
    rp = report(yv.numpy()[pi], predict(net, xv, device, pi).numpy())
    xt, yt = load_ms("test")
    rt = report(yt.numpy(), predict(net, xt, device).numpy())
    wsp.write_json({"round": args.round, "practice": rp, "test": rt, "evaluated_at": wsp.now()},
                   PRIVATE / f"round_{args.round}.json")
    fb = feedback_text(rp, args.round)
    (rd / "feedback.txt").write_text(fb + "\n", encoding="utf-8")
    print(fb)
    print(f"\n(test result stored privately in {PRIVATE.name}/round_{args.round}.json)")


def cmd_summary(_args) -> None:
    base = json.loads((PRIVATE / "baselines.json").read_text(encoding="utf-8"))
    rgb = {}
    for f in sorted((WSP_ROOT / "experiment_5_write_code" / "private").glob("round_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rgb[d["round"]] = d["test"]["test_accuracy"]
    print("round  practice  test (13 bands)  test (RGB, W2)")
    for f in sorted(PRIVATE.glob("round_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        r = rgb.get(d["round"])
        print(f"  {d['round']}     {d['practice']['test_accuracy'] * 100:5.1f}%       "
              f"{d['test']['test_accuracy'] * 100:5.1f}%          {'' if r is None else f'{r * 100:5.1f}%'}")
    print(f"\nconstant {base['constant'] * 100:.1f}% | random {base['kaiming_mean'] * 100:.1f}% | "
          f"mean-spectrum centroid {base['centroid'] * 100:.1f}% | trained ceiling {base['trained_mean'] * 100:.1f}%")


# ── baselines ────────────────────────────────────────────────────────────────
def cmd_baselines(args) -> None:
    device = wsp.get_device()
    xtr, ytr = load_ms("train")
    xva, yva = load_ms("validation")
    xte, yte = load_ms("test")
    out = {"constant": float(np.bincount(yte.numpy()).max() / len(yte)), "kaiming": [], "trained": []}

    def mean_spectrum(x):
        return torch.cat([batch(x, np.arange(i, min(i + 512, len(x))), "cpu").mean(dim=(2, 3))
                          for i in range(0, len(x), 512)])
    ftr, fte = mean_spectrum(xtr), mean_spectrum(xte)
    cent = torch.stack([ftr[ytr == c].mean(0) for c in range(10)])
    out["centroid"] = float((torch.cdist(fte, cent).argmin(1) == yte).float().mean())
    print(f"mean-spectrum centroid: {out['centroid']:.4f}")

    for seed in range(3):
        torch.manual_seed(seed)
        net = SmallNetMS()
        for m in (net.conv1, net.conv2):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(m.bias)
        out["kaiming"].append(float((predict(net, xte, device) == yte).float().mean()))
    for seed in range(3):
        torch.manual_seed(seed)
        net = SmallNetMS().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        g = torch.Generator().manual_seed(seed)
        best, state = -1.0, None
        for _ in range(args.epochs):
            net.train()
            for idx in torch.randperm(len(ytr), generator=g).split(128):
                idx = np.sort(idx.numpy())
                loss = F.cross_entropy(net(batch(xtr, idx, device)), ytr[idx].to(device))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            va = float((predict(net, xva, device) == yva).float().mean())
            if va > best:
                best, state = va, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        net.load_state_dict(state)
        acc = float((predict(net, xte, device) == yte).float().mean())
        out["trained"].append(acc)
        print(f"trained seed {seed}: test {acc:.4f}")
    out["kaiming_mean"] = float(np.mean(out["kaiming"]))
    out["trained_mean"] = float(np.mean(out["trained"]))
    out["created_at"] = wsp.now()
    wsp.write_json(out, PRIVATE / "baselines.json")
    print(json.dumps({k: v for k, v in out.items() if k != "created_at"}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    sub.add_parser("prompt")
    r = sub.add_parser("run")
    r.add_argument("--round", type=int, required=True, choices=range(MAX_ROUND + 1))
    sub.add_parser("summary")
    b = sub.add_parser("baselines")
    b.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    {"prepare": cmd_prepare, "prompt": cmd_prompt, "run": cmd_run, "summary": cmd_summary,
     "baselines": cmd_baselines}[args.cmd](args)


if __name__ == "__main__":
    main()
