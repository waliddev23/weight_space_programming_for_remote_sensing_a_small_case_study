"""Experiment 1 (READ): one small CNN, three augmentation policies, three seeds.

Every run uses the same architecture, the same official splits, the same optimizer,
LR, epoch budget and seed list. The *only* thing that changes between conditions is
the augmentation policy (none / standard / strong).

Outputs (per model):
    experiment_1_read/checkpoints/cnn_<condition>_seed<s>.pt   bare state_dict
    experiment_1_read/configs/cnn_<condition>_seed<s>.json     full config
    experiment_1_read/metrics/cnn_<condition>_seed<s>.json     val/test metrics

Run:
    python eurosat_wsp/run_experiment_1.py --condition standard
    python eurosat_wsp/run_experiment_1.py --all        # 9 models, ~30-40 min on a 3060
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import common as wsp

CONDITIONS = ("none", "standard", "strong")


# --------------------------------------------------------------------------- data


def load_data() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Decode all three official splits once per process (uint8, native 64x64)."""
    raw = wsp.load_raw()
    data = {}
    for split in ("train", "validation", "test"):
        t0 = time.time()
        data[split] = wsp.preload_split(raw[split])
        x, y = data[split]
        print(f"  loaded {split:10s} {x.shape} labels {y.min()}..{y.max()} "
              f"({time.time() - t0:.1f}s)", flush=True)
    return data


def make_loaders(data, policy: str, seed: int, batch_size: int,
                 max_train_samples: int | None = None):
    x_tr, y_tr = data["train"]
    if max_train_samples is not None and max_train_samples < len(y_tr):
        keep = np.random.RandomState(seed).permutation(len(y_tr))[:max_train_samples]
        keep.sort()
        x_tr, y_tr = x_tr[keep], y_tr[keep]

    train_ds = wsp.ArrayDataset(x_tr, y_tr, wsp.build_train_transform(policy))
    val_ds = wsp.ArrayDataset(*data["validation"], wsp.build_eval_transform())
    test_ds = wsp.ArrayDataset(*data["test"], wsp.build_eval_transform())

    train_loader = wsp.make_loader(train_ds, batch_size=batch_size, shuffle=True,
                                   seed=seed, num_workers=wsp.TRAIN_CONFIG["num_workers"])
    val_loader = wsp.make_loader(val_ds, batch_size=256, shuffle=False,
                                 num_workers=wsp.TRAIN_CONFIG["num_workers"])
    test_loader = wsp.make_loader(test_ds, batch_size=256, shuffle=False,
                                  num_workers=wsp.TRAIN_CONFIG["num_workers"])
    return train_loader, val_loader, test_loader, len(train_ds)


# -------------------------------------------------------------------------- train


def run_one(condition: str, seed: int, args, data, device) -> dict:
    name = f"cnn_{condition}_seed{seed}"
    print(f"\n=== {name} | aug={condition} | seed={seed} ===", flush=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_loader, val_loader, test_loader, n_train = make_loaders(
        data, condition, seed, args.batch_size, args.max_train_samples)

    model = wsp.build_smallcnn(seed=seed).to(device)
    params = wsp.count_parameters(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = wsp.make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    eval_criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "lr": [], "epoch_seconds": []}
    best_val_acc, best_epoch, best_state = -1.0, -1, None

    for epoch in range(args.epochs):
        t0 = time.time()
        tr_loss, tr_acc = wsp.train_one_epoch(model, train_loader, criterion,
                                              optimizer, scaler, device, use_amp)
        va_loss, va_acc, _, _ = wsp.evaluate(model, val_loader, eval_criterion,
                                             device, use_amp)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["lr"].append(lr_now)
        history["epoch_seconds"].append(time.time() - t0)

        print(f"epoch {epoch + 1:02d}/{args.epochs} | train {tr_loss:.4f}/{tr_acc:.3f} | "
              f"val {va_loss:.4f}/{va_acc:.3f} | lr {lr_now:.2e} | "
              f"{history['epoch_seconds'][-1]:.1f}s", flush=True)

        if va_acc > best_val_acc:
            best_val_acc, best_epoch = float(va_acc), epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"  -> best so far ({best_val_acc:.3f})", flush=True)

    # ---- evaluate the best (validation-selected) weights on val + held-out test
    model.load_state_dict(best_state)
    model.eval()
    val_loss, val_acc, val_pred, val_true = wsp.evaluate(
        model, val_loader, eval_criterion, device, use_amp)
    test_loss, test_acc, test_pred, test_true = wsp.evaluate(
        model, test_loader, eval_criterion, device, use_amp)
    val_metrics = wsp.classification_metrics(val_true, val_pred)
    test_metrics = wsp.classification_metrics(test_true, test_pred)

    ckpt_path = wsp.EXP1_DIR / "checkpoints" / f"{name}.pt"
    wsp.save_state_dict(best_state, ckpt_path)

    config = {
        "model_name": name,
        "experiment": "experiment_1_read",
        "condition": condition,
        "seed": seed,
        "architecture": {
            "class": "SmallCNN",
            "definition": "eurosat_wsp/common.py::SmallCNN",
            "channels": list(wsp.CHANNELS),
            "first_conv": {
                "in_channels": 3, "out_channels": wsp.FIRST_CONV_OUT,
                "kernel_size": 3, "padding": 1, "bias": True,
                "weight_shape": [wsp.FIRST_CONV_OUT, 3, 3, 3],
                "num_weights": wsp.FIRST_CONV_OUT * 3 * 3 * 3,
                "num_biases": wsp.FIRST_CONV_OUT,
            },
            "dropout": wsp.TRAIN_CONFIG["dropout"],
            "parameters": params,
            "init": "Kaiming normal (fan_out, relu) convs, BN weight=1/bias=0, "
                    "PyTorch-default linear head",
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
            "train_samples": n_train,
            "device": str(device),
            "amp": use_amp,
            "epoch_seconds": history["epoch_seconds"],
        },
        "augmentation": {
            "policy": condition,
            "transforms": wsp.describe_policy(condition),
            "eval_transforms": ["ToTensor", "Normalize"],
        },
        "data": {
            "dataset": wsp.HF_DATASET,
            "split_files": ["data/splits/train.json", "data/splits/validation.json",
                            "data/splits/test.json"],
            "img_size": wsp.IMG_SIZE,
            "mean": list(wsp.MEAN),
            "std": list(wsp.STD),
        },
        "checkpoint": f"experiment_1_read/checkpoints/{name}.pt",
        "checkpoint_format": "bare state_dict, CPU, no metadata",
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "created_at": wsp.now(),
    }
    wsp.write_json(config, wsp.EXP1_DIR / "configs" / f"{name}.json")

    metrics = {
        "model_name": name,
        "condition": condition,
        "seed": seed,
        "parameters": params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "val": {"loss": val_loss, **val_metrics},
        "test": {"loss": test_loss, **test_metrics},
        "history": history,
        "total_train_seconds": float(sum(history["epoch_seconds"])),
        "checkpoint": f"experiment_1_read/checkpoints/{name}.pt",
        "checkpoint_sha256": wsp.sha256_file(ckpt_path),
        "created_at": wsp.now(),
    }
    wsp.write_json(metrics, wsp.EXP1_DIR / "metrics" / f"{name}.json")

    print(f"--- {name}: best val {best_val_acc:.3f} (ep {best_epoch}) | "
          f"test {test_acc:.3f} | macro F1 {test_metrics['macro_f1']:.3f}", flush=True)

    del model, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 1 (READ) training")
    parser.add_argument("--condition", choices=CONDITIONS, action="append",
                        help="repeatable; default = every condition")
    parser.add_argument("--all", action="store_true", help="train all 9 models")
    parser.add_argument("--seeds", type=int, nargs="+", default=wsp.TRAIN_CONFIG["seeds"])
    parser.add_argument("--epochs", type=int, default=wsp.TRAIN_CONFIG["epochs"])
    parser.add_argument("--batch-size", type=int, default=wsp.TRAIN_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=wsp.TRAIN_CONFIG["lr"])
    parser.add_argument("--weight-decay", type=float, default=wsp.TRAIN_CONFIG["weight_decay"])
    parser.add_argument("--label-smoothing", type=float,
                        default=wsp.TRAIN_CONFIG["label_smoothing"])
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="smoke-test only: subsample the training split")
    args = parser.parse_args()

    conditions = list(CONDITIONS) if (args.all or not args.condition) else args.condition
    device = wsp.get_device()
    print(f"device: {device} | conditions: {conditions} | seeds: {args.seeds} | "
          f"epochs: {args.epochs}", flush=True)

    t0 = time.time()
    data = load_data()
    print(f"data ready in {time.time() - t0:.0f}s", flush=True)

    for condition in conditions:
        for seed in args.seeds:
            run_one(condition, seed, args, data, device)

    print(f"\nfinished in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
