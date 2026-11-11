"""Generate Figure 5: Per-Epoch Weight Initialization Impact Plot.

Shows how different initialisation regimes affect learning dynamics across all 20
training epochs.  Three Kaiming-Normal baselines (none / standard / strong
augmentation, each averaged over seeds 0-1-2 with ±1-std shading) are overlaid
with the single Claude Fable 5.1 LLM-initialised run.

Saves to:  weight_space_programming/assets/write_init_impact_per_epoch.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
WSP_ROOT   = Path(__file__).resolve().parent
REPO_ROOT  = WSP_ROOT.parent
ASSETS_DIR = REPO_ROOT / "assets"
METRICS_DIR = WSP_ROOT / "experiment_1_read" / "metrics"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)

# ── Styling ────────────────────────────────────────────────────────────────────
plt.style.use(
    "seaborn-v0_8-whitegrid"
    if "seaborn-v0_8-whitegrid" in plt.style.available
    else "default"
)
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   12,
    "figure.titlesize": 15,
    "lines.linewidth":  2.2,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

# ── Colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "none":     "#e67e22",   # orange
    "standard": "#2ecc71",   # green
    "strong":   "#9b59b6",   # purple
    "claude":   "#2980b9",   # blue
}


def _load_kaiming_history(condition: str) -> dict[str, list[float]]:
    """Load and stack val_acc / train_acc for seeds 0-1-2 of a given condition."""
    seeds_val, seeds_train = [], []
    seeds_val_loss, seeds_train_loss = [], []
    for seed in (0, 1, 2):
        p = METRICS_DIR / f"cnn_{condition}_seed{seed}.json"
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        h = data["history"]
        seeds_val.append(h["val_acc"])
        seeds_train.append(h["train_acc"])
        seeds_val_loss.append(h["val_loss"])
        seeds_train_loss.append(h["train_loss"])
    return {
        "val_acc":    np.array(seeds_val),    # (3, 20)
        "train_acc":  np.array(seeds_train),
        "val_loss":   np.array(seeds_val_loss),
        "train_loss": np.array(seeds_train_loss),
    }


def _load_claude_history() -> dict[str, list[float]]:
    """Load Claude Fable 5.1 LLM-initialised training history."""
    p = WSP_ROOT / "experiment_3_write" / "baseline" / "llm_trained_metrics.json"
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    hist = data["history"]
    return {
        "val_acc":   np.array([h["val_acc"]   for h in hist]),
        "train_acc": np.array([h["train_acc"] for h in hist]),
        "val_loss":  np.array([h["val_loss"]  for h in hist]),
        "train_loss":np.array([h["train_loss"]for h in hist]),
        "test_acc":  data["test_acc"],
    }


def _band(ax, epochs, arr2d, color, label, alpha_fill=0.18, linestyle="-"):
    """Plot mean ± 1-std band for a (seeds × epochs) array."""
    mean = arr2d.mean(axis=0) * 100
    std  = arr2d.std(axis=0) * 100
    line, = ax.plot(epochs, mean, color=color, linestyle=linestyle, label=label)
    ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=alpha_fill)
    return line


def generate_init_impact_figure():
    epochs = np.arange(1, 21)

    # Load data
    kaiming = {c: _load_kaiming_history(c) for c in ("none", "standard", "strong")}
    claude  = _load_claude_history()

    # ── Figure: 1 row × 2 columns ──────────────────────────────────────────────
    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(14, 5.5))

    condition_labels = {
        "none":     "Kaiming (No Aug.)",
        "standard": "Kaiming (Std. Aug.)",
        "strong":   "Kaiming (Strong Aug.)",
    }

    # --- Accuracy panel --------------------------------------------------------
    for cond, label in condition_labels.items():
        _band(ax_acc, epochs, kaiming[cond]["val_acc"],
              PALETTE[cond], label)

    # Claude — single run, so no std band; use dashed line + marker
    ax_acc.plot(
        epochs, claude["val_acc"] * 100,
        color=PALETTE["claude"], linestyle="--", linewidth=2.5,
        marker="D", markersize=4, label="LLM Init (Claude Fable 5.1)"
    )
    # Mark final test accuracy
    ax_acc.axhline(
        claude["test_acc"] * 100,
        color=PALETTE["claude"], linestyle=":", linewidth=1.5, alpha=0.7,
        label=f"Fable Test Acc: {claude['test_acc']*100:.2f}%"
    )

    ax_acc.set_title("Validation Accuracy vs. Epoch\n(Initialisation Regime Comparison)")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Val Accuracy (%)")
    ax_acc.set_xticks(range(1, 21, 2))
    ax_acc.set_ylim(55, 97)
    ax_acc.legend(loc="lower right", frameon=True, fontsize=10)

    # Early-epoch zoom inset (epochs 1-5) to highlight initialisation impact
    axins = ax_acc.inset_axes([0.07, 0.42, 0.42, 0.38])
    for cond, label in condition_labels.items():
        d = kaiming[cond]["val_acc"]
        mean5 = d[:, :5].mean(axis=0) * 100
        std5  = d[:, :5].std(axis=0) * 100
        axins.plot(epochs[:5], mean5, color=PALETTE[cond], linewidth=1.6)
        axins.fill_between(epochs[:5], mean5-std5, mean5+std5,
                           color=PALETTE[cond], alpha=0.18)
    axins.plot(epochs[:5], claude["val_acc"][:5]*100,
               color=PALETTE["claude"], linestyle="--", linewidth=1.8,
               marker="D", markersize=4)
    axins.set_title("Epochs 1-5 (init. divergence)", fontsize=8)
    axins.set_xticks([1, 2, 3, 4, 5])
    axins.tick_params(labelsize=8)
    ax_acc.indicate_inset_zoom(axins, edgecolor="grey")

    # --- Loss panel ------------------------------------------------------------
    for cond, label in condition_labels.items():
        _band(ax_loss, epochs, kaiming[cond]["val_loss"],
              PALETTE[cond], label)

    ax_loss.plot(
        epochs, claude["val_loss"],
        color=PALETTE["claude"], linestyle="--", linewidth=2.5,
        marker="D", markersize=4, label="LLM Init (Claude Fable 5.1)"
    )

    ax_loss.set_title("Validation Loss vs. Epoch\n(Initialisation Regime Comparison)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Val Loss (CE + Label Smooth)")
    ax_loss.set_xticks(range(1, 21, 2))
    ax_loss.legend(loc="upper right", frameon=True, fontsize=10)

    # ── Global annotation ──────────────────────────────────────────────────────
    fig.suptitle(
        "Weight Initialisation Impact on EuroSAT Training Dynamics\n"
        "Kaiming-Normal (3 conditions × 3 seeds, mean ± 1 std)  vs.  "
        "LLM-Written conv1 Weights (Claude Fable 5.1)",
        y=1.03, fontsize=13, fontweight="bold"
    )

    # ── Extra summary table as a text box ──────────────────────────────────────
    summary_lines = [
        "Final Test Accuracy Summary",
        "─" * 36,
        f"{'Kaiming None   (mean±std)':28s}  {kaiming['none']['val_acc'][:,-1].mean()*100:5.2f}% ± {kaiming['none']['val_acc'][:,-1].std()*100:.2f}%",
        f"{'Kaiming Std    (mean±std)':28s}  {kaiming['standard']['val_acc'][:,-1].mean()*100:5.2f}% ± {kaiming['standard']['val_acc'][:,-1].std()*100:.2f}%",
        f"{'Kaiming Strong (mean±std)':28s}  {kaiming['strong']['val_acc'][:,-1].mean()*100:5.2f}% ± {kaiming['strong']['val_acc'][:,-1].std()*100:.2f}%",
        f"{'LLM Init (Claude Fable 5.1)':28s}  {claude['test_acc']*100:5.2f}%  (test)",
    ]
    fig.text(0.5, -0.05, "\n".join(summary_lines),
             ha="center", va="top", fontsize=9.5,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4ff", edgecolor="#aab"))

    out_p = ASSETS_DIR / "write_init_impact_per_epoch.png"
    plt.savefig(out_p)
    plt.close()
    print(f"Saved → {out_p}")
    return out_p


if __name__ == "__main__":
    generate_init_impact_figure()
