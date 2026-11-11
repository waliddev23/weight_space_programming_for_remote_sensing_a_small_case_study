"""Experiment 2 (EDIT): ResNet-18 ground truth plus three corrupted copies.

Preparation only - no LLM repair is attempted here.

* The original checkpoint is the one already trained in this project
  (notebooks/train_resnet18.ipynb -> checkpoints/resnet18_eurosat.pt, 98.0% val).
  It is copied byte-for-byte into experiment_2_edit/original/ and its sha256 is
  recorded; the source file is never modified.
* Every corrupted variant is derived from a fresh CPU copy of the *weights*, so the
  original file on disk is untouched.

Variants (all target a single, clearly defined conv layer):
    A  gaussian_noise   controlled Gaussian noise added to selected filters
    B  pruning          a fixed percentage of weights in selected filters set to 0
    C  filter_removal   selected whole filters set to 0

Run:
    python eurosat_wsp/prepare_experiment_2.py --calibrate   # strength sweep
    python eurosat_wsp/prepare_experiment_2.py               # final artifacts
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

import common as wsp

SOURCE_CKPT = wsp.PROJECT_ROOT / "checkpoints" / "resnet18_eurosat.pt"
ORIGINAL_DIR = wsp.EXP2_DIR / "original"
CORRUPTED_DIR = wsp.EXP2_DIR / "corrupted"
CONFIG_DIR = wsp.EXP2_DIR / "corruption_configs"
METRICS_DIR = wsp.EXP2_DIR / "metrics"

IMG_SIZE = 224                      # the pretrained ResNet wants 224x224
TARGET_LAYER = "layer3.0.conv1.weight"   # (256, 128, 3, 3), 294,912 weights
CORRUPTION_SEED = 12345

# Chosen after the calibration sweep (recorded in corruption_configs/calibration.json).
CORRUPTION_SPECS = {
    "A": {
        "name": "gaussian_noise",
        "description": "add Gaussian noise scaled to the layer's weight std",
        "filters": list(range(0, 32)),          # 32 of 256 output filters
        "noise_scale": 0.0,                     # filled in by --calibrate
    },
    "B": {
        "name": "pruning",
        "description": "set a fixed percentage of the selected weights to zero",
        "filters": list(range(0, 96)),          # 96 of 256 output filters
        "prune_fraction": 0.0,                  # filled in by --calibrate
    },
    "C": {
        "name": "filter_removal",
        "description": "set selected whole filters (all their weights) to zero",
        "filters": list(range(64, 96)),         # 32 of 256 output filters
    },
}

# Calibration sweep (val accuracy after corruption is measured for each setting).
CALIBRATION_GRID = {
    "A": {"noise_scale": [4.0, 6.0, 8.0, 12.0, 16.0]},
    "B": {"prune_fraction": [0.9, 0.95, 0.99, 1.0]},
    "C": {"filters": [list(range(0, 16)), list(range(0, 32)),
                      list(range(0, 64)), list(range(0, 96))]},
}
CALIBRATION_TARGET_VAL_ACC = 0.60   # clearly degraded, still far above chance


# ------------------------------------------------------------------------- model


def build_resnet18(num_classes: int = wsp.NUM_CLASSES) -> nn.Module:
    model = models.resnet18(weights=None)      # weights come from the checkpoint
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_loaders(data):
    train_tf = wsp.transforms.Compose([
        wsp.transforms.Resize(IMG_SIZE),
        wsp.transforms.CenterCrop(IMG_SIZE),
        wsp.transforms.ToTensor(),
        wsp.transforms.Normalize(wsp.MEAN, wsp.STD),
    ])
    val_ds = wsp.ArrayDataset(*data["validation"], train_tf)
    test_ds = wsp.ArrayDataset(*data["test"], train_tf)
    return (
        wsp.make_loader(val_ds, batch_size=128, shuffle=False),
        wsp.make_loader(test_ds, batch_size=128, shuffle=False),
    )


@torch.no_grad()
def accuracy(model, loader, device) -> tuple[float, float, np.ndarray, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    return wsp.evaluate(model, loader, criterion, device, device.type == "cuda")


# -------------------------------------------------------------------- corruption


def _check_layer(sd: dict[str, torch.Tensor]) -> torch.Tensor:
    if TARGET_LAYER not in sd:
        raise KeyError(f"{TARGET_LAYER} not in checkpoint")
    return sd[TARGET_LAYER]


def _filter_indices(shape: tuple[int, ...], filters: list[int]) -> np.ndarray:
    """Flat indices of every element belonging to the selected output filters."""
    per_filter = int(np.prod(shape[1:]))
    return np.concatenate([np.arange(f * per_filter, (f + 1) * per_filter) for f in filters])


def corrupt(state: dict[str, torch.Tensor], variant: str, spec: dict,
            seed: int = CORRUPTION_SEED) -> tuple[dict[str, torch.Tensor], np.ndarray, dict]:
    """Return (corrupted copy, modified flat indices in the target tensor, details)."""
    sd = {k: v.clone() for k, v in state.items()}
    w = _check_layer(sd)
    shape = tuple(w.shape)
    filters = list(spec["filters"])
    details: dict = {
        "variant": variant,
        "strategy": spec["name"],
        "target_layer": TARGET_LAYER,
        "layer_shape": list(shape),
        "selected_filters": filters,
        "num_filters": len(filters),
        "seed": seed,
    }
    g = torch.Generator().manual_seed(seed)

    if spec["name"] == "gaussian_noise":
        scale = float(spec["noise_scale"])
        noise = torch.randn((len(filters), *shape[1:]), generator=g) * (scale * float(w.std()))
        with torch.no_grad():
            w[filters] = w[filters] + noise
        modified = _filter_indices(shape, filters)
        details["noise_scale"] = scale
        details["noise_std"] = scale * float(w.std())
        details["layer_weight_std"] = float(state[TARGET_LAYER].std())

    elif spec["name"] == "pruning":
        frac = float(spec["prune_fraction"])
        per_filter = int(np.prod(shape[1:]))
        n_zero_per_filter = int(round(frac * per_filter))
        keep_local = []
        local_zeroed = []
        for f in filters:
            perm = torch.randperm(per_filter, generator=g)[:n_zero_per_filter]
            local_zeroed.append(perm.numpy())
            keep_local.append(f)
        idx = np.concatenate([f * per_filter + local for f, local in zip(keep_local, local_zeroed)])
        with torch.no_grad():
            flat = w.view(-1)
            flat[torch.from_numpy(idx)] = 0.0
        modified = np.sort(idx)
        details["prune_fraction"] = frac
        details["zeroed_per_filter"] = n_zero_per_filter

    elif spec["name"] == "filter_removal":
        with torch.no_grad():
            w[filters] = 0.0
        modified = _filter_indices(shape, filters)

    else:
        raise ValueError(f"unknown corruption strategy {spec['name']!r}")

    sd[TARGET_LAYER] = w

    changed = int((sd[TARGET_LAYER] != state[TARGET_LAYER]).sum())
    details.update({
        "modified_elements": int(modified.size),
        "elements_changed_vs_original": changed,
        "fraction_of_layer": float(modified.size / state[TARGET_LAYER].numel()),
        "fraction_of_model_weights": float(
            modified.size / sum(v.numel() for v in state.values() if v.is_floating_point())),
    })
    return sd, modified, details


# --------------------------------------------------------------------------- main


def prepare_original(device) -> dict[str, torch.Tensor]:
    """Copy the trained checkpoint (byte-identical) and return its weights."""
    ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
    dst = ORIGINAL_DIR / "resnet18_original.pt"
    src_hash = wsp.sha256_file(SOURCE_CKPT)

    if dst.exists():
        if wsp.sha256_file(dst) != src_hash:
            raise RuntimeError(f"{dst} differs from {SOURCE_CKPT}; refusing to overwrite")
        print(f"original already present and identical: {dst}")
    else:
        shutil.copy2(SOURCE_CKPT, dst)
        print(f"copied {SOURCE_CKPT} -> {dst}")

    ckpt = torch.load(dst, map_location="cpu", weights_only=True)
    state = {k: v.clone() for k, v in ckpt["model"].items()}

    # A bare state_dict for the LLM-facing copy, next to the archived artifact.
    bare = ORIGINAL_DIR / "resnet18_original_state_dict.pt"
    wsp.save_state_dict(state, bare)

    wsp.write_json({
        "source_checkpoint": str(SOURCE_CKPT.relative_to(wsp.PROJECT_ROOT)),
        "source_sha256": src_hash,
        "archived_copy": "experiment_2_edit/original/resnet18_original.pt",
        "archived_copy_sha256": wsp.sha256_file(dst),
        "bare_state_dict": "experiment_2_edit/original/resnet18_original_state_dict.pt",
        "bare_state_dict_sha256": wsp.sha256_file(bare),
        "provenance": (
            "trained by notebooks/train_resnet18.ipynb: ImageNet-pretrained resnet18, "
            "10-class head, 224x224 inputs, 10 epochs (4 frozen + 6 fine-tune), "
            "AdamW + cosine schedule, official EuroSAT splits"
        ),
        "reported_val_acc": float(ckpt["val_acc"]),
        "img_size": int(ckpt["img_size"]),
        "classes": list(ckpt["classes"]),
        "num_tensors": len(state),
        "num_parameters": int(sum(v.numel() for v in state.values())),
        "note": "the archived copy is byte-identical to the training artifact; "
                "corruptions are applied to in-memory copies of 'bare_state_dict'",
        "created_at": wsp.now(),
    }, ORIGINAL_DIR / "provenance.json")
    return state


def evaluate_state(state, val_loader, test_loader, device) -> dict:
    model = build_resnet18().to(device)
    model.load_state_dict(state)
    model.eval()
    val_loss, val_acc, vp, vt = accuracy(model, val_loader, device)
    test_loss, test_acc, tp, tt = accuracy(model, test_loader, device)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "val": {"loss": val_loss, "accuracy": val_acc, **wsp.classification_metrics(vt, vp)},
        "test": {"loss": test_loss, "accuracy": test_acc, **wsp.classification_metrics(tt, tp)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Experiment 2 (EDIT) preparation")
    parser.add_argument("--calibrate", action="store_true",
                        help="sweep corruption strengths and record val accuracy")
    parser.add_argument("--force", action="store_true",
                        help="recreate corrupted checkpoints even if they exist")
    args = parser.parse_args()

    device = wsp.get_device()
    print(f"device: {device}", flush=True)

    data = wsp.preload_all(wsp.load_raw())
    val_loader, test_loader = build_loaders(data)

    state = prepare_original(device)
    print(f"original weights: {len(state)} tensors, "
          f"{sum(v.numel() for v in state.values()):,} parameters", flush=True)

    # ---- original accuracy: the ground truth for every later comparison
    print("\n[evaluating original] ...", flush=True)
    original_metrics = evaluate_state(state, val_loader, test_loader, device)
    wsp.write_json({
        "model": "resnet18_original",
        "checkpoint": "experiment_2_edit/original/resnet18_original_state_dict.pt",
        "checkpoint_sha256": wsp.sha256_file(
            ORIGINAL_DIR / "resnet18_original_state_dict.pt"),
        "uncorrupted": True,
        "img_size": IMG_SIZE,
        **original_metrics,
        "created_at": wsp.now(),
    }, METRICS_DIR / "original.json")
    print(f"original: val {original_metrics['val']['accuracy']:.4f} | "
          f"test {original_metrics['test']['accuracy']:.4f}", flush=True)

    specs = {k: dict(v) for k, v in CORRUPTION_SPECS.items()}
    chosen_path = CONFIG_DIR / "chosen_strengths.json"
    if chosen_path.exists() and not args.calibrate:
        chosen = wsp.read_json(chosen_path)
        for variant, values in chosen.items():
            specs[variant].update(values)
        print(f"loaded chosen strengths from {chosen_path}", flush=True)

    # ---- optional calibration sweep
    if args.calibrate:
        sweep = {}
        best = {}
        for variant, grid in CALIBRATION_GRID.items():
            sweep[variant] = []
            best_gap = float("inf")
            for key, values in grid.items():
                for value in values:
                    spec = dict(specs[variant])
                    spec[key] = value
                    sd, _, details = corrupt(state, variant, spec)
                    model = build_resnet18().to(device)
                    model.load_state_dict(sd)
                    model.eval()
                    _, val_acc, _, _ = accuracy(model, val_loader, device)
                    del model
                    gap = abs(val_acc - CALIBRATION_TARGET_VAL_ACC)
                    record = {
                        "setting": {key: value},
                        "val_accuracy": val_acc,
                        "elements_changed": details["elements_changed_vs_original"],
                    }
                    sweep[variant].append(record)
                    print(f"  [{variant}] {key}={value} -> val {val_acc:.4f}", flush=True)
                    if gap < best_gap:
                        best_gap = gap
                        best[variant] = {"setting": {key: value}, "val_accuracy": val_acc}
            print(f"[{variant}] chosen: {best[variant]}", flush=True)

        wsp.write_json({
            "target_val_accuracy": CALIBRATION_TARGET_VAL_ACC,
            "sweep": sweep,
            "chosen": best,
            "seed": CORRUPTION_SEED,
            "created_at": wsp.now(),
        }, CONFIG_DIR / "calibration.json")

        chosen = {v: dict(best[v]["setting"]) for v in best}
        wsp.write_json(chosen, chosen_path)
        for variant, values in chosen.items():
            specs[variant].update(values)

    # ---- final corrupted checkpoints
    summary = {}
    for variant, spec in specs.items():
        if "noise_scale" in spec and spec["noise_scale"] <= 0:
            raise SystemExit("run --calibrate first: noise_scale is unset")
        if "prune_fraction" in spec and spec["prune_fraction"] <= 0:
            raise SystemExit("run --calibrate first: prune_fraction is unset")

        sd, modified, details = corrupt(state, variant, spec)
        ckpt_path = CORRUPTED_DIR / f"resnet18_corrupted_{variant.lower()}_{spec['name']}.pt"

        if ckpt_path.exists() and not args.force:
            print(f"{ckpt_path.name} exists - skipping (use --force to rebuild)")
        else:
            wsp.save_state_dict(sd, ckpt_path)

        index_file = CONFIG_DIR / f"variant_{variant.lower()}_modified_indices.npy"
        np.save(index_file, modified)

        print(f"\n[evaluating variant {variant}: {spec['name']}] ...", flush=True)
        metrics = evaluate_state(sd, val_loader, test_loader, device)

        config = {
            "variant": variant,
            "name": spec["name"],
            "description": spec["description"],
            "experiment": "experiment_2_edit",
            "target_layer": TARGET_LAYER,
            "layer_shape": list(state[TARGET_LAYER].shape),
            "selected_output_filters": list(spec["filters"]),
            "strategy_params": {k: v for k, v in spec.items()
                                if k not in {"name", "description", "filters"}},
            "seed": CORRUPTION_SEED,
            "modified_elements": details["elements_changed_vs_original"],
            "fraction_of_layer": details["fraction_of_layer"],
            "fraction_of_model_weights": details["fraction_of_model_weights"],
            "modified_indices_file":
                f"experiment_2_edit/corruption_configs/variant_{variant.lower()}_modified_indices.npy",
            "checkpoint": f"experiment_2_edit/corrupted/{ckpt_path.name}",
            "checkpoint_sha256": wsp.sha256_file(ckpt_path),
            "original_checkpoint": "experiment_2_edit/original/resnet18_original_state_dict.pt",
            "original_checkpoint_sha256": wsp.sha256_file(
                ORIGINAL_DIR / "resnet18_original_state_dict.pt"),
            "accuracy_before": original_metrics,
            "accuracy_after": metrics,
            "created_at": wsp.now(),
        }
        wsp.write_json(config, CONFIG_DIR / f"variant_{variant.lower()}_{spec['name']}.json")
        wsp.write_json({
            "model": f"resnet18_corrupted_{variant.lower()}_{spec['name']}",
            "variant": variant,
            "strategy": spec["name"],
            "checkpoint": config["checkpoint"],
            "checkpoint_sha256": config["checkpoint_sha256"],
            "uncorrupted": False,
            **metrics,
            "created_at": wsp.now(),
        }, METRICS_DIR / f"corrupted_{variant.lower()}.json")

        summary[variant] = {
            "name": spec["name"],
            "strategy_params": config["strategy_params"],
            "modified_elements": details["elements_changed_vs_original"],
            "val_accuracy": metrics["val"]["accuracy"],
            "test_accuracy": metrics["test"]["accuracy"],
            "checkpoint": config["checkpoint"],
        }
        print(f"variant {variant}: val {metrics['val']['accuracy']:.4f} | "
              f"test {metrics['test']['accuracy']:.4f} | "
              f"{details['elements_changed_vs_original']:,} weights changed", flush=True)

    wsp.write_json({
        "original": {
            "checkpoint": "experiment_2_edit/original/resnet18_original_state_dict.pt",
            "val_accuracy": original_metrics["val"]["accuracy"],
            "test_accuracy": original_metrics["test"]["accuracy"],
        },
        "corrupted": summary,
        "seed": CORRUPTION_SEED,
        "created_at": wsp.now(),
    }, CONFIG_DIR / "corruption_summary.json")
    print("\nwrote experiment_2_edit/ artifacts")


if __name__ == "__main__":
    main()
