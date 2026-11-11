"""Weight-Space Programming Serializer for EuroSAT models.

Translates PyTorch model weights into structured, human-and-LLM-readable text
following the format used by Takehi et al. (MIT CSAIL, Nov 2026).

Capabilities:
1. Focused serialization: conv1 (432 weights) + classifier head (1,280 weights) (~4k tokens)
2. Full serialization: All 98,954 weights of SmallCNN (~245k tokens) for Claude 1M context
3. Blind A/B pairing: Randomizes two models (e.g. none vs strong), generates an anonymous
   comparison file, and saves a separate answer key.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure eurosat_wsp root is in sys.path
WSP_ROOT = Path(__file__).resolve().parent
if str(WSP_ROOT) not in sys.path:
    sys.path.insert(0, str(WSP_ROOT))

import common as wsp


def format_vector(tensor_1d: torch.Tensor, precision: int = 3) -> str:
    """Format a 1D tensor as space-separated floats."""
    return " ".join(f"{val:.{precision}f}" for val in tensor_1d.flatten().tolist())


def serialize_conv2d(name: str, weight: torch.Tensor, precision: int = 3) -> list[str]:
    """Format a 4D Conv2d weight [C_out, C_in, kh, kw] into structured lines."""
    lines = []
    cout, cin, kh, kw = weight.shape
    channel_labels = ["R", "G", "B"] if cin == 3 else [f"c{c}" for c in range(cin)]

    for f_idx in range(cout):
        for c_idx in range(cin):
            c_tag = channel_labels[c_idx] if c_idx < len(channel_labels) else f"c{c_idx}"
            for r_idx in range(kh):
                row_vals = format_vector(weight[f_idx, c_idx, r_idx, :], precision=precision)
                lines.append(f"{name} filter {f_idx:02d} chan {c_tag} row {r_idx}  {row_vals}")
    return lines


def serialize_linear(name: str, weight: torch.Tensor, precision: int = 3) -> list[str]:
    """Format a 2D Linear weight [out_features, in_features] into row lines."""
    lines = []
    out_features, _ = weight.shape
    for r_idx in range(out_features):
        row_vals = format_vector(weight[r_idx, :], precision=precision)
        class_hint = f" ({wsp.CLASS_NAMES[r_idx]})" if out_features == wsp.NUM_CLASSES and r_idx < len(wsp.CLASS_NAMES) else ""
        lines.append(f"{name} row {r_idx:02d}{class_hint}  {row_vals}")
    return lines


def serialize_tensor(name: str, tensor: torch.Tensor, precision: int = 3) -> list[str]:
    """Serialize any tensor depending on its dimensionality."""
    t = tensor.detach().cpu().float()
    if t.ndim == 4:
        return serialize_conv2d(name, t, precision=precision)
    elif t.ndim == 2:
        return serialize_linear(name, t, precision=precision)
    elif t.ndim == 1:
        return [f"{name}  {format_vector(t, precision=precision)}"]
    else:
        flat_vals = format_vector(t.flatten(), precision=precision)
        return [f"{name} (shape {list(t.shape)})  {flat_vals}"]


def serialize_state_dict(state_dict: dict[str, torch.Tensor],
                         mode: str = "focused",
                         precision: int = 3) -> str:
    """Serialize a model state_dict.
    
    Modes:
      'focused': conv1.weight, conv1.bias, and classifier linear layer (~4k tokens)
      'full': all model weights (~245k tokens)
    """
    lines = []
    keys_to_export = []

    if mode == "focused":
        keys_to_export = [
            "conv1.weight",
            "conv1.bias",
            "classifier.2.weight",
            "classifier.2.bias",
        ]
    elif mode == "full":
        keys_to_export = list(state_dict.keys())
    else:
        raise ValueError(f"Unknown mode: {mode}. Choose 'focused' or 'full'.")

    total_numel = 0
    for k in keys_to_export:
        if k not in state_dict:
            continue
        tensor = state_dict[k]
        total_numel += tensor.numel()
        lines.append(f"# --- {k} (shape: {list(tensor.shape)}, numel: {tensor.numel()}) ---")
        lines.extend(serialize_tensor(k, tensor, precision=precision))
        lines.append("")

    header = [
        f"# Model Weights Serialization (Mode: {mode}, Precision: {precision} decimals)",
        f"# Total Parameters Exported: {total_numel:,}",
        f"# Generated: {wsp.now()}",
        "",
    ]
    return "\n".join(header + lines)


def create_blind_ab_comparison(path_a: Path, path_b: Path,
                               label_a: str, label_b: str,
                               mode: str = "focused",
                               seed: int | None = 42) -> tuple[str, dict]:
    """Randomly assign model 1 and model 2 to 'Model A' and 'Model B' to produce
    a blind A/B comparison string and a corresponding answer key."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    models = [
        {"path": path_a, "condition": label_a},
        {"path": path_b, "condition": label_b},
    ]
    rng.shuffle(models)

    state_a = wsp.load_state_dict(models[0]["path"])
    state_b = wsp.load_state_dict(models[1]["path"])

    text_a = serialize_state_dict(state_a, mode=mode)
    text_b = serialize_state_dict(state_b, mode=mode)

    combined = [
        "=" * 80,
        "=== MODEL A ===",
        "=" * 80,
        text_a,
        "\n" + "=" * 80,
        "=== MODEL B ===",
        "=" * 80,
        text_b,
    ]

    key = {
        "model_a": {
            "source": str(models[0]["path"]),
            "condition": models[0]["condition"],
        },
        "model_b": {
            "source": str(models[1]["path"]),
            "condition": models[1]["condition"],
        },
        "mode": mode,
        "seed": seed,
        "created_at": wsp.now(),
    }

    return "\n".join(combined), key


def create_blind_9_models_sweep(checkpoints_dir: Path | None = None,
                                mode: str = "focused",
                                seed: int = 42) -> tuple[str, dict]:
    """Shuffle all 9 trained CNN checkpoints (3 conditions x 3 seeds) into an
    anonymous 'Model 01' through 'Model 09' blind benchmark."""
    if checkpoints_dir is None:
        checkpoints_dir = WSP_ROOT / "experiment_1_read" / "checkpoints"

    conditions = ["none", "standard", "strong"]
    seeds = [0, 1, 2]

    models_info = []
    for cond in conditions:
        for s in seeds:
            p = checkpoints_dir / f"cnn_{cond}_seed{s}.pt"
            if not p.exists():
                raise FileNotFoundError(f"Missing checkpoint: {p}")
            models_info.append({
                "path": p,
                "condition": f"{cond}_augmentation" if cond != "none" else "no_augmentation",
                "raw_condition": cond,
                "train_seed": s,
            })

    rng = random.Random(seed)
    rng.shuffle(models_info)

    combined_sections = [
        "=" * 80,
        "# BLIND 9-MODEL BENCHMARK: IDENTIFY REGULARIZATION REGIME ACROSS 9 NETWORKS",
        f"# Total Models: {len(models_info)} (3 conditions x 3 seeds)",
        f"# Mode: {mode} (conv1 + classifier weights per model)",
        "=" * 80,
        "",
    ]

    key_manifest = {}
    for idx, item in enumerate(models_info, start=1):
        model_label = f"Model_{idx:02d}"
        state_dict = wsp.load_state_dict(item["path"])
        serialized = serialize_state_dict(state_dict, mode=mode)

        combined_sections.append("=" * 80)
        combined_sections.append(f"=== {model_label} ===")
        combined_sections.append("=" * 80)
        combined_sections.append(serialized)
        combined_sections.append("\n")

        key_manifest[model_label] = {
            "source_file": item["path"].name,
            "condition": item["condition"],
            "raw_condition": item["raw_condition"],
            "train_seed": item["train_seed"],
        }

    full_key = {
        "num_models": len(models_info),
        "conditions": ["no_augmentation", "standard_augmentation", "strong_augmentation"],
        "shuffle_seed": seed,
        "models": key_manifest,
        "created_at": wsp.now(),
    }

    return "\n".join(combined_sections), full_key


def main() -> None:
    parser = argparse.ArgumentParser(description="WSP Weight Serializer")
    parser.add_argument("--checkpoint", type=str, help="Path to single .pt checkpoint")
    parser.add_argument("--mode", choices=["focused", "full"], default="focused",
                        help="focused (conv1 + head) or full (all weights)")
    parser.add_argument("--precision", type=int, default=3, help="Decimal precision")
    parser.add_argument("--output", type=str, default=None, help="Output .txt file")
    
    # Blind comparison options
    parser.add_argument("--blind-compare", nargs=2, metavar=("CKPT_1", "CKPT_2"),
                        help="Two checkpoints to compare blindly")
    parser.add_argument("--labels", nargs=2, default=["no_augmentation", "strong_augmentation"],
                        help="Condition labels for the two checkpoints")
    parser.add_argument("--key-output", type=str, default=None,
                        help="Path to save blind comparison key (.json)")
    parser.add_argument("--seed", type=int, default=42, help="Seed for blind shuffle")

    args = parser.parse_args()

    if args.blind_compare:
        ckpt1 = Path(args.blind_compare[0])
        ckpt2 = Path(args.blind_compare[1])
        text, key = create_blind_ab_comparison(
            ckpt1, ckpt2,
            label_a=args.labels[0], label_b=args.labels[1],
            mode=args.mode, seed=args.seed
        )
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(text, encoding="utf-8")
            print(f"Saved blind A/B text to {out_p}")
        else:
            print(text[:2000] + "\n... [truncated] ...")

        if args.key_output:
            key_p = Path(args.key_output)
            wsp.write_json(key, key_p)
            print(f"Saved answer key to {key_p}")

    elif args.checkpoint:
        ckpt = Path(args.checkpoint)
        state_dict = wsp.load_state_dict(ckpt)
        text = serialize_state_dict(state_dict, mode=args.mode, precision=args.precision)
        if args.output:
            out_p = Path(args.output)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(text, encoding="utf-8")
            print(f"Saved serialized weights to {out_p}")
        else:
            print(text[:2000] + "\n... [truncated] ...")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
