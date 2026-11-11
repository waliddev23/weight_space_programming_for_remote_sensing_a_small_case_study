"""Initialization interface for Experiment 3 (WRITE).

Accepts externally generated weights for the small CNN's first convolutional layer
(`conv1`, 16 x 3 x 3 x 3 = 432 weights + 16 biases) and drops them into the shared
architecture. Every other layer keeps the documented Kaiming initialization for the
requested seed.

Examples
--------
Check a file without building anything:
    python eurosat_wsp/external_init.py --weights weights.npy --check-only

Build a full model with external first-conv weights:
    python eurosat_wsp/external_init.py --weights weights.pt --seed 0 \
        --save experiment_3_write/baseline/smallcnn_external_seed0.pt

Score the resulting (untrained) model on the held-out test split:
    python eurosat_wsp/external_init.py --weights weights.pt --seed 0 --eval-test
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import common as wsp


def describe(weights: torch.Tensor) -> dict:
    stats = wsp.tensor_stats(weights)
    stats["expected_numel"] = wsp.FIRST_CONV_OUT * 3 * 3 * 3
    stats["is_finite"] = bool(torch.isfinite(weights).all())
    return stats


def build_with_external_init(weights_path: str | Path, seed: int = 0,
                             num_classes: int = wsp.NUM_CLASSES,
                             save: str | Path | None = None) -> wsp.SmallCNN:
    """Build SmallCNN(seed) and replace only ``conv1.weight`` with external values."""
    model = wsp.build_smallcnn(seed=seed, num_classes=num_classes)
    weights = wsp.load_external_first_conv(weights_path)
    wsp.apply_external_first_conv(model, weights)
    if save is not None:
        wsp.save_state_dict(wsp.state_dict_cpu(model), save)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 3 first-conv init interface")
    parser.add_argument("--weights", required=True,
                        help=".pt/.pth/.npy/.npz with a (16, 3, 3, 3) tensor")
    parser.add_argument("--seed", type=int, default=0,
                        help="seed for the remaining layers' Kaiming init")
    parser.add_argument("--save", default=None, help="where to write the full state_dict")
    parser.add_argument("--check-only", action="store_true",
                        help="validate the tensor and print stats, build nothing")
    parser.add_argument("--eval-test", action="store_true",
                        help="evaluate the resulting model on the held-out test split")
    args = parser.parse_args()

    weights = wsp.load_external_first_conv(args.weights)
    print("weights:", describe(weights))

    expected = (wsp.FIRST_CONV_OUT, 3, 3, 3)
    if tuple(weights.shape) != expected:
        raise SystemExit(f"wrong shape: expected {expected}, got {tuple(weights.shape)}")
    if not torch.isfinite(weights).all():
        raise SystemExit("weights contain NaN or Inf")
    if args.check_only:
        print("shape and finiteness OK")
        return

    model = build_with_external_init(args.weights, seed=args.seed, save=args.save)
    replaced = (model.conv1.weight.detach().cpu() == weights).all().item()
    print(f"conv1 replaced: {replaced} | params: {wsp.count_parameters(model):,}"
          f" | seed: {args.seed}")
    if args.save:
        print("saved", args.save)

    if args.eval_test:
        device = wsp.get_device()
        model.to(device).eval()
        data = wsp.preload_split(wsp.load_raw()["test"])
        loader = wsp.make_loader(wsp.ArrayDataset(*data, wsp.build_eval_transform()),
                                 batch_size=256, shuffle=False)
        loss, acc, pred, true = wsp.evaluate(
            model, loader, nn.CrossEntropyLoss(), device, device.type == "cuda")
        print(f"test loss {loss:.4f} | test accuracy {acc:.3f} (untrained)")


if __name__ == "__main__":
    main()
