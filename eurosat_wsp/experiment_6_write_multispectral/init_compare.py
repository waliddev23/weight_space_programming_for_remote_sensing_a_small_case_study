"""Experiment W3-init: Fable's W3 network as a starting point for training, against Kaiming.

Pre-registered in docs/next_steps.md (Experiment W3-init).

    python eurosat_wsp/experiment_6_write_multispectral/init_compare.py          # all runs (resumable)
    python eurosat_wsp/experiment_6_write_multispectral/init_compare.py --summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import w3
from w3 import SmallNetMS, load_ms

wsp = w3.wsp
OUT = HERE / "init_compare" / "results.json"
FABLE_ROUND = 3
INITS = ["kaiming", "fable_as_written", "fable_rescaled"]
BUDGETS = {1.0: 30, 0.1: 60, 0.01: 200}          # fraction of training images -> epochs
SEEDS = [0, 1, 2]


def to_tensor(x_mm, device, dtype):
    out = torch.empty((len(x_mm), *x_mm.shape[1:]), dtype=dtype, device=device)
    for i in range(0, len(x_mm), 1024):
        out[i:i + 1024] = torch.from_numpy(np.asarray(x_mm[i:i + 1024]).astype(np.float32) / w3.SCALE).to(device, dtype)
    return out


@torch.no_grad()
def accuracy(net, x, y):
    """Data may live on the GPU or in host memory; batches are moved to the model's device."""
    net.eval()
    dev = next(net.parameters()).device
    correct = 0
    for i in range(0, len(x), 512):
        correct += (net(x[i:i + 512].to(dev).float()).argmax(1) == y[i:i + 512].to(dev)).sum().item()
    return correct / len(x)


def fable_net(rescale: bool, x_cal, device) -> SmallNetMS:
    w = dict(np.load(HERE / "runs" / f"round_{FABLE_ROUND}" / "weights.npz"))
    net = SmallNetMS().load_written(w)
    if rescale:
        with torch.no_grad():
            net.eval()
            net.to(device)
            logits = torch.cat([net(x_cal[i:i + 256].to(device).float()) for i in range(0, len(x_cal), 256)])
            c = 1.0 / logits.std().item()
            net.fc.weight.mul_(c)
            net.fc.bias.mul_(c)
    return net


def kaiming_net(seed: int) -> SmallNetMS:
    torch.manual_seed(seed)
    net = SmallNetMS()
    for m in (net.conv1, net.conv2):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        nn.init.zeros_(m.bias)
    return net


def subset(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    if frac >= 1.0:
        return np.arange(len(y))
    g = np.random.default_rng(int(round(frac * 10000)) * 10 + seed)
    idx = [g.choice(np.where(y == c)[0], max(1, int(round(frac * (y == c).sum()))), replace=False) for c in range(10)]
    return np.sort(np.concatenate(idx))


def train(net, xtr, ytr, xva, yva, xte, yte, epochs, seed, device):
    net = net.to(device)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator().manual_seed(seed)
    best, state, history = -1.0, None, []
    for _ in range(epochs):
        net.train()
        for idx in torch.randperm(len(ytr), generator=g).split(128):
            idx = idx.to(xtr.device)
            loss = F.cross_entropy(net(xtr[idx].to(device).float()), ytr[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
        va = accuracy(net, xva, yva)
        history.append(va)
        if va > best:
            best, state = va, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return {"test_accuracy": accuracy(net, xte, yte), "best_val": best, "val_epoch1": history[0],
            "val_history": history}


def run_all() -> None:
    device = wsp.get_device()
    free = torch.cuda.mem_get_info()[0] if device.type == "cuda" else 0
    # About 2.9 GB in half precision: kept on the GPU when it fits, otherwise in host memory
    data_device = device if free > 4.5e9 else torch.device("cpu")
    print(f"compute on {device}, data on {data_device}", flush=True)
    raw = {s: load_ms(s) for s in ("train", "validation", "test")}
    x = {s: to_tensor(raw[s][0], data_device, torch.float16) for s in raw}
    y = {s: raw[s][1].to(data_device) for s in raw}
    ytr_np = raw["train"][1].numpy()
    cal = x["train"][np.random.default_rng(0).choice(len(ytr_np), 1000, replace=False)]

    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    for frac in sorted(BUDGETS):
        for seed in SEEDS:
            idx = torch.from_numpy(subset(ytr_np, frac, seed)).to(data_device)
            xtr, ytr = x["train"][idx], y["train"][idx]
            for init in INITS:
                key = f"{frac}|{seed}|{init}"
                if key in results:
                    continue
                t0 = time.time()
                net = kaiming_net(seed) if init == "kaiming" else fable_net(init == "fable_rescaled", cal, device)
                r = train(net, xtr, ytr, x["validation"], y["validation"], x["test"], y["test"], BUDGETS[frac], seed,
                          device)
                r.update({"budget": frac, "n_train": int(len(idx)), "seed": seed, "init": init,
                          "epochs": BUDGETS[frac], "seconds": round(time.time() - t0, 1)})
                results[key] = r
                OUT.parent.mkdir(exist_ok=True)
                OUT.write_text(json.dumps(results, indent=1), encoding="utf-8")
                print(f"budget {frac:>4} seed {seed} {init:17s} test {r['test_accuracy'] * 100:5.1f}%  "
                      f"val@1 {r['val_epoch1'] * 100:5.1f}%  ({r['seconds']}s)", flush=True)
    summary()


def summary() -> None:
    results = json.loads(OUT.read_text(encoding="utf-8"))
    lines = ["| Training images | Starting point | Test accuracy, mean (per seed) | Val. accuracy after epoch 1 | Verdict vs Kaiming |",
             "|---|---|---|---|---|"]
    for frac in sorted(BUDGETS, reverse=True):
        acc = {i: [results[f"{frac}|{s}|{i}"]["test_accuracy"] * 100 for s in SEEDS
                   if f"{frac}|{s}|{i}" in results] for i in INITS}
        ep1 = {i: [results[f"{frac}|{s}|{i}"]["val_epoch1"] * 100 for s in SEEDS
                   if f"{frac}|{s}|{i}" in results] for i in INITS}
        n = next((r["n_train"] for r in results.values() if r["budget"] == frac), "?")
        for i in INITS:
            if len(acc[i]) < len(SEEDS):
                continue
            verdict = ""
            if i != "kaiming" and len(acc["kaiming"]) == len(SEEDS):
                diffs = [a - k for a, k in zip(acc[i], acc["kaiming"])]
                if np.mean(diffs) > 0 and all(d > 0 for d in diffs):
                    verdict = f"helps ({np.mean(diffs):+.1f})"
                elif all(d < 0 for d in diffs):
                    verdict = f"hurts ({np.mean(diffs):+.1f})"
                else:
                    verdict = f"no clear difference ({np.mean(diffs):+.1f})"
            lines.append(f"| {frac * 100:g}% ({n}) | {i.replace('_', ' ')} | {np.mean(acc[i]):.1f}% "
                         f"({', '.join(f'{a:.1f}' for a in acc[i])}) | {np.mean(ep1[i]):.1f}% | {verdict} |")
    table = "\n".join(lines)
    (OUT.parent / "summary.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run_all()
