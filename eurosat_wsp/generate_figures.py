"""Generate publication-quality figures for the EuroSAT WSP benchmark.

Saves to weight_space_programming/assets/:
  1. write_training_curves.png       — 20-epoch curves: Claude Init vs Kaiming
  2. read_seed_correlation_heatmap.png — 9×9 Pearson correlation matrix
  3. edit_resnet18_recovery.png       — Original / Corrupted / Repaired bar chart
  4. write_synthesized_filters.png    — 16 synthesised 3×3 RGB filter gallery
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
WSP_ROOT  = Path(__file__).resolve().parent
REPO_ROOT = WSP_ROOT.parent
ASSETS_DIR = REPO_ROOT / "assets"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style (applied once, shared by all figures) ─────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#E8E8E8",
    "grid.linewidth":    0.8,
    "lines.linewidth":   2.2,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
})

# Colour palette
C_CLAUDE  = "#2176AE"   # blue
C_KAIMING = "#E07B39"   # orange
C_GREEN   = "#2CA02C"
C_RED     = "#D62728"


# ── Figure 1: Training Curves ─────────────────────────────────────────────────
def plot_training_curves():
    """20-epoch val accuracy & loss: Claude Fable 5.1 init vs Kaiming baseline."""
    claude_p  = WSP_ROOT / "experiment_3_write" / "baseline" / "llm_trained_metrics.json"
    kaiming_p = WSP_ROOT / "experiment_1_read" / "metrics" / "cnn_standard_seed0.json"

    with open(claude_p,  encoding="utf-8") as f: claude_data  = json.load(f)
    with open(kaiming_p, encoding="utf-8") as f: kaiming_data = json.load(f)

    epochs = np.arange(1, 21)
    ch = claude_data["history"]
    kh = kaiming_data["history"]

    c_val_acc  = [h["val_acc"]  * 100 for h in ch]
    c_tr_acc   = [h["train_acc"]* 100 for h in ch]
    c_val_loss = [h["val_loss"]       for h in ch]
    c_tr_loss  = [h["train_loss"]     for h in ch]

    k_val_acc  = [a * 100 for a in kh["val_acc"]]
    k_tr_acc   = [a * 100 for a in kh["train_acc"]]
    k_val_loss = kh["val_loss"]
    k_tr_loss  = kh["train_loss"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    # Accuracy
    ax1.plot(epochs, c_val_acc,  color=C_CLAUDE,  label="Claude Fable 5.1 — Val")
    ax1.plot(epochs, c_tr_acc,   color=C_CLAUDE,  alpha=0.35, linestyle=":")
    ax1.plot(epochs, k_val_acc,  color=C_KAIMING, linestyle="--", label="Kaiming (Std Aug) — Val")
    ax1.plot(epochs, k_tr_acc,   color=C_KAIMING, alpha=0.35, linestyle=":")
    ax1.axhline(claude_data["test_acc"]*100,       color=C_CLAUDE,  linestyle="-.", alpha=0.7,
                label=f"Fable test: {claude_data['test_acc']*100:.2f}%")
    ax1.axhline(kaiming_data["test"]["accuracy"]*100, color=C_KAIMING, linestyle="-.", alpha=0.7,
                label=f"Kaiming test: {kaiming_data['test']['accuracy']*100:.2f}%")
    ax1.set_title("Validation Accuracy", fontweight="bold")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_xlim(1, 20); ax1.set_ylim(50, 97)
    ax1.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax1.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#ccc")

    # Loss
    ax2.plot(epochs, c_val_loss,  color=C_CLAUDE,  label="Claude Fable 5.1 — Val")
    ax2.plot(epochs, c_tr_loss,   color=C_CLAUDE,  alpha=0.35, linestyle=":")
    ax2.plot(epochs, k_val_loss,  color=C_KAIMING, linestyle="--", label="Kaiming (Std Aug) — Val")
    ax2.plot(epochs, k_tr_loss,   color=C_KAIMING, alpha=0.35, linestyle=":")
    ax2.set_title("Validation Loss", fontweight="bold")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss (CE + Label Smooth)")
    ax2.set_xlim(1, 20)
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9, edgecolor="#ccc")

    fig.suptitle(
        "Experiment 3 — WRITE: Training Dynamics\n"
        "LLM-Initialised conv1 (Claude Fable 5.1)  vs.  Kaiming-Normal Baseline",
        fontsize=13, fontweight="bold", y=1.02
    )
    out = ASSETS_DIR / "write_training_curves.png"
    plt.savefig(out); plt.close()
    print(f"Saved → {out}")


# ── Figure 2: Seed Correlation Heatmap ───────────────────────────────────────
def plot_seed_correlation_heatmap():
    """9×9 Pearson correlation matrix of conv1 weight vectors for 9 blind models."""
    ckpt_dir = WSP_ROOT / "experiment_1_read" / "checkpoints"
    key_p    = WSP_ROOT / "prompts" / "key_read_exp1_all9_models.json"
    with open(key_p, encoding="utf-8") as f:
        key_data = json.load(f)

    model_labels = [f"Model_{i:02d}" for i in range(1, 10)]
    weights, tick_labels = [], []

    for label in model_labels:
        src = key_data["models"][label]["source_file"]
        state = torch.load(ckpt_dir / src, map_location="cpu", weights_only=True)
        weights.append(state["conv1.weight"].detach().cpu().numpy().flatten())
        raw = key_data["models"][label]["raw_condition"]
        s   = key_data["models"][label]["train_seed"]
        tick_labels.append(f"{label}\n({raw[:3]}, s{s})")

    corr = np.corrcoef(np.array(weights))

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(corr, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson r", fontsize=11)

    ax.set_xticks(range(9)); ax.set_yticks(range(9))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tick_labels, fontsize=9)
    ax.spines[:].set_visible(False)

    for i in range(9):
        for j in range(9):
            v = corr[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.55 else "#222",
                    fontsize=8.5, fontweight="bold")

    ax.set_title(
        "Experiment 1 — READ: Pairwise conv1 Weight Correlation (9 Blind Models)\n"
        "Latent seed clusters: {01,02,08}  |  {03,04,09}  |  {05,06,07}",
        fontsize=12, fontweight="bold", pad=14
    )
    out = ASSETS_DIR / "read_seed_correlation_heatmap.png"
    plt.savefig(out, bbox_inches="tight"); plt.close()
    print(f"Saved → {out}")


# ── Figure 3: EDIT Recovery Bar Chart ────────────────────────────────────────
def plot_resnet18_recovery():
    """Original / Corrupted / Repaired ResNet-18 accuracy comparison."""
    p = WSP_ROOT / "experiment_2_edit" / "metrics" / "llm_repair_a_result.json"
    with open(p, encoding="utf-8") as f:
        d = json.load(f)

    orig_acc    = d["original_accuracy"]  * 100
    corrupt_acc = d["corrupted_accuracy"] * 100
    repair_acc  = d["repaired_accuracy"]  * 100
    recovery    = d["recovery_percentage"]

    fig, ax = plt.subplots(figsize=(8, 5.5), constrained_layout=True)

    bars = ax.bar(
        ["Original ResNet-18\n(Pristine Baseline)",
         "Corrupted Variant A\n(Gaussian Noise, SNR 1:35)",
         "Repaired by Fable 5.1\n(Debiased Wiener-MMSE)"],
        [orig_acc, corrupt_acc, repair_acc],
        color=[C_GREEN, C_RED, C_CLAUDE],
        width=0.52, edgecolor="white", linewidth=0
    )

    for bar, val in zip(bars, [orig_acc, corrupt_acc, repair_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.2f}%", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color="#222")

    ax.annotate(
        f"+{recovery:.1f}% Recovery\n(zero data, no backprop)",
        xy=(2, repair_acc * 0.55),
        xytext=(1.35, 72),
        arrowprops=dict(arrowstyle="->", color=C_CLAUDE, lw=1.8),
        fontsize=10, fontweight="bold", color=C_CLAUDE,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#e8f4ff", edgecolor=C_CLAUDE, alpha=0.9)
    )

    ax.set_ylim(0, 115)
    ax.set_ylabel("EuroSAT Test Accuracy (%)")
    ax.yaxis.grid(True, color="#E8E8E8", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(
        "Experiment 2 — EDIT: Surgical Weight Repair of ResNet-18\n"
        "layer3.0.conv1 — no training data, no gradient descent",
        fontsize=13, fontweight="bold", pad=12
    )

    out = ASSETS_DIR / "edit_resnet18_recovery.png"
    plt.savefig(out); plt.close()
    print(f"Saved → {out}")


# ── Figure 4: Synthesised Filters ────────────────────────────────────────────
def plot_synthesized_filters():
    """4×4 gallery of the 16 LLM-synthesised 3×3 RGB conv1 filters."""
    weights_p = WSP_ROOT / "runs" / "write_run_001" / "parsed_output" / "first_conv_fable.pt"
    weights   = torch.load(weights_p, map_location="cpu", weights_only=True).numpy()  # (16,3,3,3)

    names = [
        "F00  VARI Green-Excess",   "F01  Green-Red Contrast",
        "F02  Laplacian of VARI",   "F03  Dark-Green (Forest)",
        "F04  Water B-R Contrast",  "F05  SeaLake (Blue Dom.)",
        "F06  River Ripple/Edge",   "F07  Sobel-X Edge",
        "F08  Sobel-Y Edge",        "F09  45° Diagonal Edge",
        "F10  135° Diagonal Edge",  "F11  Crop Row (Horiz.)",
        "F12  Crop Row (Vert.)",    "F13  Diagonal Furrow",
        "F14  Urban Spot/Corner",   "F15  Built Structure",
    ]

    fig, axes = plt.subplots(4, 4, figsize=(10, 10), constrained_layout=True)
    fig.patch.set_facecolor("white")

    for i, ax in enumerate(axes.flatten()):
        f = weights[i].transpose(1, 2, 0)          # HWC
        f_norm = (f - f.min()) / (f.max() - f.min() + 1e-12)
        ax.imshow(f_norm, interpolation="nearest")
        ax.set_title(names[i], fontsize=8.5, fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc"); spine.set_linewidth(0.8)

    fig.suptitle(
        "Experiment 3 — WRITE: 16 LLM-Synthesised 3×3 RGB Satellite Filters\n"
        "Physical priors compiled directly into weight space by Claude Fable 5.1",
        fontsize=13, fontweight="bold", y=1.02
    )
    out = ASSETS_DIR / "write_synthesized_filters.png"
    plt.savefig(out); plt.close()
    print(f"Saved → {out}")


def main():
    print("Generating 4 publication-quality figures …")
    plot_training_curves()
    plot_seed_correlation_heatmap()
    plot_resnet18_recovery()
    plot_synthesized_filters()
    print("Done — all figures saved to assets/")


if __name__ == "__main__":
    main()
