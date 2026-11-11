"""Figure 5: Per-Epoch Initialisation Impact — No-Aug Kaiming vs Claude Fable 5.1.

Clean two-panel plot:
  Left  — Validation Accuracy per epoch
  Right — Validation Loss per epoch

Kaiming No-Augmentation baseline: mean ± 1 std over seeds 0, 1, 2.
Claude Fable 5.1 LLM-initialised run: single run, dashed line.

Saves to:  weight_space_programming/assets/write_init_impact_per_epoch.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
WSP_ROOT    = Path(__file__).resolve().parent
ASSETS_DIR  = WSP_ROOT.parent / "assets"
METRICS_DIR = WSP_ROOT / "experiment_1_read" / "metrics"
ASSETS_DIR.mkdir(parents=True, exist_ok=True)


def _load_kaiming_no_aug() -> dict[str, np.ndarray]:
    """Load val_acc / val_loss for the no-augmentation Kaiming baseline (seeds 0-2)."""
    val_acc_seeds, val_loss_seeds = [], []
    for seed in (0, 1, 2):
        p = METRICS_DIR / f"cnn_none_seed{seed}.json"
        with open(p, encoding="utf-8") as f:
            h = json.load(f)["history"]
        val_acc_seeds.append(h["val_acc"])
        val_loss_seeds.append(h["val_loss"])
    return {
        "val_acc":  np.array(val_acc_seeds),   # shape (3, 20)
        "val_loss": np.array(val_loss_seeds),
    }


def _load_claude() -> dict[str, np.ndarray | float]:
    """Load Claude Fable 5.1 LLM-initialised training history."""
    p = WSP_ROOT / "experiment_3_write" / "baseline" / "llm_trained_metrics.json"
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    hist = data["history"]
    return {
        "val_acc":  np.array([h["val_acc"]  for h in hist]),
        "val_loss": np.array([h["val_loss"] for h in hist]),
        "test_acc": data["test_acc"],
    }


def generate():
    epochs  = np.arange(1, 21)
    kaiming = _load_kaiming_no_aug()
    claude  = _load_claude()

    # Computed statistics
    k_acc_mean = kaiming["val_acc"].mean(0) * 100
    k_acc_std  = kaiming["val_acc"].std(0)  * 100
    k_loss_mean = kaiming["val_loss"].mean(0)
    k_loss_std  = kaiming["val_loss"].std(0)

    c_acc  = claude["val_acc"]  * 100
    c_loss = claude["val_loss"]

    # ── Colours ────────────────────────────────────────────────────────────────
    C_KAIMING = "#E07B39"   # warm orange
    C_CLAUDE  = "#2176AE"   # clean blue

    # ── Style ──────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.color":       "#E8E8E8",
        "grid.linewidth":   0.8,
        "lines.linewidth":  2.2,
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "savefig.dpi":      300,
        "savefig.bbox":     "tight",
    })

    fig, (ax_acc, ax_loss) = plt.subplots(
        1, 2, figsize=(13, 5), constrained_layout=True
    )

    # ── Left panel: Validation Accuracy ───────────────────────────────────────
    ax_acc.fill_between(
        epochs, k_acc_mean - k_acc_std, k_acc_mean + k_acc_std,
        color=C_KAIMING, alpha=0.18, label="_nolegend_"
    )
    ax_acc.plot(
        epochs, k_acc_mean,
        color=C_KAIMING, linewidth=2.4,
        label=f"Kaiming Normal — No Aug  (mean ± 1σ, 3 seeds)"
    )
    ax_acc.plot(
        epochs, c_acc,
        color=C_CLAUDE, linewidth=2.4, linestyle="--",
        marker="o", markersize=4, markevery=2,
        label=f"LLM Init — Claude Fable 5.1  (test: {claude['test_acc']*100:.2f}%)"
    )

    ax_acc.set_title("Validation Accuracy", fontsize=13, fontweight="bold", pad=10)
    ax_acc.set_xlabel("Epoch", fontsize=11)
    ax_acc.set_ylabel("Val Accuracy (%)", fontsize=11)
    ax_acc.set_xlim(1, 20)
    ax_acc.set_ylim(55, 97)
    ax_acc.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax_acc.legend(loc="lower right", frameon=True, fontsize=10,
                  framealpha=0.9, edgecolor="#cccccc")

    # ── Right panel: Validation Loss ──────────────────────────────────────────
    ax_loss.fill_between(
        epochs, k_loss_mean - k_loss_std, k_loss_mean + k_loss_std,
        color=C_KAIMING, alpha=0.18, label="_nolegend_"
    )
    ax_loss.plot(
        epochs, k_loss_mean,
        color=C_KAIMING, linewidth=2.4,
        label="Kaiming Normal — No Aug  (mean ± 1σ, 3 seeds)"
    )
    ax_loss.plot(
        epochs, c_loss,
        color=C_CLAUDE, linewidth=2.4, linestyle="--",
        marker="o", markersize=4, markevery=2,
        label="LLM Init — Claude Fable 5.1"
    )

    ax_loss.set_title("Validation Loss", fontsize=13, fontweight="bold", pad=10)
    ax_loss.set_xlabel("Epoch", fontsize=11)
    ax_loss.set_ylabel("Val Loss (CE + Label Smooth)", fontsize=11)
    ax_loss.set_xlim(1, 20)
    ax_loss.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax_loss.legend(loc="upper right", frameon=True, fontsize=10,
                   framealpha=0.9, edgecolor="#cccccc")

    # ── Super-title ────────────────────────────────────────────────────────────
    fig.suptitle(
        "Weight Initialisation Impact on EuroSAT Training Dynamics\n"
        "Kaiming-Normal No-Augmentation  vs.  LLM-Synthesized conv1 (Claude Fable 5.1)",
        fontsize=13, fontweight="bold", y=1.02
    )

    out_p = ASSETS_DIR / "write_init_impact_per_epoch.png"
    plt.savefig(out_p)
    plt.close()
    print(f"Saved → {out_p}")


if __name__ == "__main__":
    generate()
