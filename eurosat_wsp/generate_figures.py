"""Generate high-resolution publication-quality figures for the EuroSAT WSP GitHub repository.

Saves figures to weight_space_programming/assets/:
1. write_training_curves.png: Epoch-by-epoch comparison (Claude Init vs Kaiming Baseline)
2. read_seed_correlation_heatmap.png: 9x9 Pearson correlation matrix showing seed recovery
3. edit_resnet18_recovery.png: Benchmark of Original vs Corrupted vs Repaired ResNet-18
4. write_synthesized_filters.png: Visual gallery of the 16 handcrafted 3x3 RGB filters
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Paths
WSP_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WSP_ROOT.parent
ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# Set high-quality styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 15,
    "lines.linewidth": 2,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_training_curves():
    """Figure 1: 20-epoch training curves (Loss and Accuracy)."""
    # 1. Load Claude init history
    claude_metrics_p = WSP_ROOT / "experiment_3_write" / "baseline" / "llm_trained_metrics.json"
    with open(claude_metrics_p, "r", encoding="utf-8") as f:
        claude_data = json.load(f)
    claude_hist = claude_data["history"]

    # 2. Load Kaiming baseline history
    kaiming_metrics_p = WSP_ROOT / "experiment_1_read" / "metrics" / "cnn_standard_seed0.json"
    with open(kaiming_metrics_p, "r", encoding="utf-8") as f:
        kaiming_data = json.load(f)
    kaiming_hist = kaiming_data["history"]

    epochs = np.arange(1, 21)

    claude_tr_loss = [h["train_loss"] for h in claude_hist]
    claude_val_loss = [h["val_loss"] for h in claude_hist]
    claude_tr_acc = [h["train_acc"] * 100 for h in claude_hist]
    claude_val_acc = [h["val_acc"] * 100 for h in claude_hist]

    kaiming_tr_loss = kaiming_hist["train_loss"]
    kaiming_val_loss = kaiming_hist["val_loss"]
    kaiming_tr_acc = [a * 100 for a in kaiming_hist["train_acc"]]
    kaiming_val_acc = [a * 100 for a in kaiming_hist["val_acc"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Accuracy Plot ---
    ax1.plot(epochs, claude_val_acc, color="#1f77b4", marker="o", markersize=4, label="Claude Fable Init (Val)")
    ax1.plot(epochs, kaiming_val_acc, color="#ff7f0e", linestyle="--", marker="s", markersize=4, label="Kaiming Ref (Val)")
    ax1.plot(epochs, claude_tr_acc, color="#1f77b4", alpha=0.35, label="Claude Fable Init (Train)")
    ax1.plot(epochs, kaiming_tr_acc, color="#ff7f0e", linestyle=":", alpha=0.35, label="Kaiming Ref (Train)")

    ax1.axhline(claude_data["test_acc"] * 100, color="#1f77b4", linestyle="-.", alpha=0.7,
                label=f"Claude Final Test: {claude_data['test_acc']*100:.2f}%")
    ax1.axhline(kaiming_data["test"]["accuracy"] * 100, color="#ff7f0e", linestyle="-.", alpha=0.7,
                label=f"Kaiming Final Test: {kaiming_data['test']['accuracy']*100:.2f}%")

    ax1.set_title("Validation Accuracy Across 20 Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_xticks(range(1, 21, 2))
    ax1.set_ylim(50, 95)
    ax1.legend(loc="lower right", frameon=True)

    # --- Loss Plot ---
    ax2.plot(epochs, claude_val_loss, color="#1f77b4", marker="o", markersize=4, label="Claude Fable Init (Val Loss)")
    ax2.plot(epochs, kaiming_val_loss, color="#ff7f0e", linestyle="--", marker="s", markersize=4, label="Kaiming Ref (Val Loss)")
    ax2.plot(epochs, claude_tr_loss, color="#1f77b4", alpha=0.35, label="Claude Fable Init (Train Loss)")
    ax2.plot(epochs, kaiming_tr_loss, color="#ff7f0e", linestyle=":", alpha=0.35, label="Kaiming Ref (Train Loss)")

    ax2.set_title("Cross-Entropy Loss (Label Smoothing 0.1)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_xticks(range(1, 21, 2))
    ax2.legend(loc="upper right", frameon=True)

    fig.suptitle("Experiment 3 (WRITE): Training Dynamics with LLM-Initialized conv1 vs. Kaiming Normal", y=1.03)
    out_p = ASSETS_DIR / "write_training_curves.png"
    plt.savefig(out_p)
    plt.close()
    print(f"Saved: {out_p}")


def plot_seed_correlation_heatmap():
    """Figure 2: 9x9 Pearson correlation heatmap of conv1 weights."""
    ckpt_dir = WSP_ROOT / "experiment_1_read" / "checkpoints"
    key_p = WSP_ROOT / "prompts" / "key_read_exp1_all9_models.json"
    with open(key_p, "r", encoding="utf-8") as f:
        key_data = json.load(f)

    # Load 9 models in order Model_01 to Model_09
    model_labels = [f"Model_{i:02d}" for i in range(1, 10)]
    weights = []
    condition_tags = []

    for label in model_labels:
        src = key_data["models"][label]["source_file"]
        state = torch.load(ckpt_dir / src, map_location="cpu", weights_only=True)
        w = state["conv1.weight"].detach().cpu().numpy().flatten()
        weights.append(w)
        raw_cond = key_data["models"][label]["raw_condition"]
        seed = key_data["models"][label]["train_seed"]
        condition_tags.append(f"{label}\n({raw_cond[:3]}, s{seed})")

    weights = np.array(weights)
    corr_matrix = np.corrcoef(weights)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr_matrix, cmap="magma", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation (r)")

    ax.set_xticks(range(9))
    ax.set_yticks(range(9))
    ax.set_xticklabels(condition_tags, rotation=45, ha="right")
    ax.set_yticklabels(condition_tags)

    # Annotate correlation values
    for i in range(9):
        for j in range(9):
            val = corr_matrix[i, j]
            color = "black" if val > 0.6 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9, fontweight="bold")

    ax.set_title("Experiment 1 (READ): Pairwise Correlation Matrix Across 9 Blind Models\n"
                 "(Highlighting Latent Seed Recovery: {01,02,08}, {03,04,09}, {05,06,07})", pad=15)
    out_p = ASSETS_DIR / "read_seed_correlation_heatmap.png"
    plt.savefig(out_p)
    plt.close()
    print(f"Saved: {out_p}")


def plot_resnet18_recovery():
    """Figure 3: ResNet-18 EDIT Accuracy Recovery Benchmark."""
    edit_metrics_p = WSP_ROOT / "experiment_2_edit" / "metrics" / "llm_repair_a_result.json"
    with open(edit_metrics_p, "r", encoding="utf-8") as f:
        edit_data = json.load(f)

    orig_acc = edit_data["original_accuracy"] * 100
    corrupt_acc = edit_data["corrupted_accuracy"] * 100
    repaired_acc = edit_data["repaired_accuracy"] * 100
    recovery = edit_data["recovery_percentage"]

    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Original ResNet-18\n(Pristine Baseline)", 
                  "Corrupted Variant A\n(Gaussian Noise, SNR 1:35)", 
                  "Repaired via Fable 5.1\n(Debiased Wiener-MMSE)"]
    values = [orig_acc, corrupt_acc, repaired_acc]
    colors = ["#2ca02c", "#d62728", "#1f77b4"]

    bars = ax.bar(categories, values, color=colors, width=0.55, edgecolor="black", linewidth=1.2)

    # Annotations on top of bars
    for bar, val in zip(bars, values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.2, f"{val:.2f}%", 
                ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Arrow and Recovery indicator
    ax.annotate(f"Accuracy Recovery:\n+{recovery:.1f}% (Zero-Data)",
                xy=(2, repaired_acc / 2), xytext=(1.2, 70),
                arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=2, headwidth=8),
                fontsize=11, fontweight="bold", color="#1f77b4",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#e6f2ff", edgecolor="#1f77b4"))

    ax.set_ylim(0, 115)
    ax.set_ylabel("EuroSAT Test Accuracy (%)")
    ax.set_title("Experiment 2 (EDIT): Surgical Weight Repair of ResNet-18 (layer3.0.conv1)\n"
                 "Without Backpropagation or Training Samples", pad=12)

    out_p = ASSETS_DIR / "edit_resnet18_recovery.png"
    plt.savefig(out_p)
    plt.close()
    print(f"Saved: {out_p}")


def plot_synthesized_filters():
    """Figure 4: Visual gallery of the 16 synthesized 3x3 RGB filters."""
    weights_p = WSP_ROOT / "runs" / "write_run_001" / "parsed_output" / "first_conv_fable.pt"
    weights = torch.load(weights_p, map_location="cpu", weights_only=True).numpy()  # [16, 3, 3, 3]

    filter_names = [
        "F00: VARI Green-Excess", "F01: Green-Red Contrast", "F02: Lapl. of VARI", "F03: Dark-Green (Forest)",
        "F04: Water B-R Contrast", "F05: SeaLake (Blue Dom.)", "F06: River Ripple/Edge", "F07: Sobel-X Edge",
        "F08: Sobel-Y Edge", "F09: 45° Diagonal Edge", "F10: 135° Diagonal Edge", "F11: Crop Row (Horiz.)",
        "F12: Crop Row (Vert.)", "F13: Diagonal Furrow", "F14: Urban Spot/Corner", "F15: Built Structure"
    ]

    fig, axes = plt.subplots(4, 4, figsize=(11, 11))
    axes = axes.flatten()

    for i in range(16):
        # Shape: [3, 3, 3] -> transpose to [3, 3, 3] HWC
        f_rgb = weights[i].transpose(1, 2, 0)
        # Normalize to [0, 1] for visual display
        f_min, f_max = f_rgb.min(), f_rgb.max()
        f_norm = (f_rgb - f_min) / (f_max - f_min + 1e-12)

        axes[i].imshow(f_norm, interpolation="nearest")
        axes[i].set_title(filter_names[i], fontsize=9, fontweight="bold")
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    fig.suptitle("Experiment 3 (WRITE): The 16 Synthesized 3x3 RGB Satellite Land-Cover Filters\n"
                 "(Natural Language Physical Priors Compiled Directly into Weight Space)", y=0.98, fontsize=14)
    out_p = ASSETS_DIR / "write_synthesized_filters.png"
    plt.savefig(out_p)
    plt.close()
    print(f"Saved: {out_p}")


def main():
    print("Generating all 4 publication-quality figures into assets/...")
    plot_training_curves()
    plot_seed_correlation_heatmap()
    plot_resnet18_recovery()
    plot_synthesized_filters()
    print("All figures successfully generated in assets/!")


if __name__ == "__main__":
    main()
