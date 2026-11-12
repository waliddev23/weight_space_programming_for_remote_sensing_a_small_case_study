"""Figures for the EuroSAT weight-space programming write-up.

    python eurosat_wsp/generate_figures.py            # all figures
    python eurosat_wsp/generate_figures.py fig3 fig4  # a subset

Every figure is rebuilt from result files on disk (no hard-coded numbers) and
saved to assets/ as PNG (for Medium) and SVG (for GitHub / print). Figures carry
no titles: the claim belongs in the caption next to the image (see README).

Inputs
  fig1, fig2  experiment_1_read/checkpoints + prompts/key_read_exp1_all9_models.json
  fig3        experiment_2_edit/metrics/edit_baselines.json  (run_edit_baselines.py)
  fig4        experiment_3_write/trained_runs/*.json          (wsp.py eval-write --train)
  fig5        runs/write_run_001/parsed_output/first_conv_fable.pt
  fig6        experiment_5_write_code/private, experiment_6_write_multispectral/private, W1 result
  fig7        experiment_6_write_multispectral/init_compare/results.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

WSP_ROOT = Path(__file__).resolve().parent
if str(WSP_ROOT) not in sys.path:
    sys.path.insert(0, str(WSP_ROOT))

import common as wsp
import edit_repairs
import read_features as rf

ASSETS = WSP_ROOT.parent / "assets"
WRITE_RUN = WSP_ROOT / "runs" / "write_run_001"

# ── Palette (validated: dataviz reference palette, light surface) ────────────
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"          # series 1: the LLM-written / proposed method
ORANGE = "#eb6834"        # series 2: Kaiming baseline
CONDITION = {             # ordinal ramp, one hue, light -> dark = more augmentation
    "no_augmentation": "#86b6ef",
    "standard_augmentation": "#2a78d6",
    "strong_augmentation": "#104281",
}
CONDITION_LABEL = {"no_augmentation": "none", "standard_augmentation": "standard",
                   "strong_augmentation": "strong"}
CONDITION_LEGEND = {"no_augmentation": "no augmentation", "standard_augmentation": "standard augmentation",
                    "strong_augmentation": "strong augmentation"}
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#f4f8fd", "#86b6ef", "#2a78d6", "#104281"])
DIV = LinearSegmentedColormap.from_list("blue_red", ["#2a78d6", "#f0efec", "#e34948"])

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "lines.linewidth": 1.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "svg.fonttype": "none",
})


def save(fig, name: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 250}), ("svg", {})):
        fig.savefig(ASSETS / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.08, **kw)
    plt.close(fig)
    print(f"saved assets/{name}.png/.svg")


def panel_label(ax, text: str) -> None:
    ax.set_title(text, loc="left", color=INK, fontsize=9, pad=8)


# ── READ: load the nine models in seed-major order ───────────────────────────
def load_read_models():
    key = wsp.read_json(WSP_ROOT / "prompts" / "key_read_exp1_all9_models.json")["models"]
    order = list(CONDITION)
    items = sorted(key.items(), key=lambda kv: (kv[1]["train_seed"], order.index(kv[1]["condition"])))
    models = []
    for model_id, meta in items:
        sd = wsp.load_state_dict(WSP_ROOT / "experiment_1_read" / "checkpoints" / meta["source_file"])
        models.append({"id": model_id, "seed": meta["train_seed"], "condition": meta["condition"],
                       "state": sd, **rf.fingerprint(sd)})
    return models


def fig1_read_seed_structure(models) -> None:
    corr = rf.correlation_matrix([m["state"]["conv1.weight"] for m in models])
    labels = [f"M{m['id'][-2:]} · {CONDITION_LABEL[m['condition']]}" for m in models]

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(np.clip(corr, 0, 1), cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(9), labels, rotation=50, ha="right", rotation_mode="anchor")
    ax.set_yticks(range(9), labels)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for b in (2.5, 5.5):                      # surface gaps between seed blocks
        ax.axhline(b, color=SURFACE, lw=3)
        ax.axvline(b, color=SURFACE, lw=3)
    for i in range(9):                        # label only the within-seed pairs
        for j in range(9):
            if i != j and i // 3 == j // 3:
                ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white")
    for g in range(3):
        ax.text(g * 3 + 1, -0.85, f"seed {models[g * 3]['seed']}", ha="center", va="bottom",
                color=INK_2, fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, ticks=[0, 0.5, 1])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8, labelcolor=INK_2)
    cb.set_label("Pearson r of conv1 weights", color=INK_2)
    save(fig, "fig1_read_seed_structure")


def fig2_read_fingerprints(models) -> None:
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    for seed in sorted({m["seed"] for m in models}):
        trio = [m for m in models if m["seed"] == seed]
        xs = [m["luminance_fraction"] for m in trio]
        ys = [m["head_effective_rank"] for m in trio]
        ax.plot(xs, ys, color=AXIS, lw=1.0, zorder=1)
        # label the strong-augmentation end, alternating sides so neighbours never touch
        side = -1 if seed % 2 == 0 and seed != max(m["seed"] for m in models) else 1
        ax.annotate(f"seed {seed}", (xs[-1], ys[-1]), xytext=(9 * side, 0), textcoords="offset points",
                    ha="right" if side < 0 else "left", va="center", fontsize=8, color=MUTED)
    for cond, color in CONDITION.items():
        pts = [m for m in models if m["condition"] == cond]
        ax.scatter([m["luminance_fraction"] for m in pts], [m["head_effective_rank"] for m in pts],
                   s=46, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                   label=CONDITION_LEGEND[cond])
    ax.set_xlabel("conv1 luminance fraction  (higher = more colour-invariant)")
    ax.set_ylabel("classifier effective rank")
    ax.set_ylim(9.1, 9.85)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2,
              columnspacing=1.2)
    save(fig, "fig2_read_fingerprints")


# ── EDIT ─────────────────────────────────────────────────────────────────────
EDIT_ORDER = ["corrupted", "zero_rows", "norm_match", "clip_range", "global_svd",
              "mean_filter", "conditional_mean"]
VARIANT_TITLE = {"A": "A  noise on 32 filters",
                 "B": "B  90% pruned in 96 filters",
                 "C": "C  96 filters removed"}


def fig3_edit_methods() -> None:
    res = wsp.read_json(WSP_ROOT / "experiment_2_edit" / "metrics" / "edit_baselines.json")
    orig = res["original_test_accuracy"] * 100
    variants = list(res["variants"])
    fig, axes = plt.subplots(1, len(variants), figsize=(7.4, 3.0), sharey=True)
    y = np.arange(len(EDIT_ORDER))[::-1]
    for ax, v in zip(np.atleast_1d(axes), variants):
        rows = res["variants"][v]["methods"]
        acc = [rows[m]["test_accuracy"] * 100 for m in EDIT_ORDER]
        corrupted = res["variants"][v]["corrupted_test_accuracy"] * 100
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.axvline(orig, color=INK_2, lw=0.9)
        if v == variants[0]:
            ax.text(orig - 1.5, y[1] + 0.5, f"original{chr(10)}{orig:.1f}", ha="right", va="center",
                    fontsize=7.5, color=INK_2, linespacing=1.1)
        ax.hlines(y, corrupted, acc, color=GRID, lw=2.0, zorder=1)
        colors = [BLUE if m == "conditional_mean" else MUTED for m in EDIT_ORDER]
        ax.scatter(acc, y, s=34, color=colors, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        best = acc[EDIT_ORDER.index("conditional_mean")]
        ax.annotate(f"{best:.1f}", (best, y[EDIT_ORDER.index("conditional_mean")]),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    fontsize=8, color=INK)
        ax.set_xlim(40, 101)
        ax.set_xticks([40, 60, 80, 100])
        ax.set_xlabel("test accuracy (%)")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        panel_label(ax, VARIANT_TITLE.get(v, v))
    axes[0].set_yticks(y, [edit_repairs.LABELS[m] for m in EDIT_ORDER])
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_edit_methods")


# ── WRITE ────────────────────────────────────────────────────────────────────
def load_trained(prefix: str):
    runs = {}
    for p in sorted((WSP_ROOT / "experiment_3_write" / "trained_runs").glob(f"{prefix}*_seed*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        runs[d["seed"]] = d
    return runs


def fig4_write_training() -> None:
    fable = load_trained("first_conv_fable")
    kaiming = load_trained("first_conv_kaiming")
    seeds = sorted(set(fable) & set(kaiming))
    if not seeds:
        raise SystemExit("no paired runs in experiment_3_write/trained_runs; run eval-write --train first")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 2.9), width_ratios=[2.3, 1])
    ax1.grid(True, axis="y")
    ax1.set_axisbelow(True)
    for runs, color, name in ((kaiming, ORANGE, "Kaiming init"), (fable, BLUE, "LLM-written conv1")):
        curves = np.array([[h["val_acc"] * 100 for h in runs[s]["history"]] for s in seeds])
        ep = np.arange(1, curves.shape[1] + 1)
        ax1.fill_between(ep, curves.min(0), curves.max(0), color=color, alpha=0.15, lw=0)
        ax1.plot(ep, curves.mean(0), color=color, label=name)
    ax1.set_xlim(1, ep[-1])
    ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("validation accuracy (%)")
    ax1.legend(loc="lower right")
    panel_label(ax1, f"Validation accuracy, mean and range over {len(seeds)} seeds")

    ax2.grid(True, axis="y")
    ax2.set_axisbelow(True)
    for s in seeds:
        k, f = kaiming[s]["test_acc"] * 100, fable[s]["test_acc"] * 100
        ax2.plot([0, 1], [k, f], color=AXIS, lw=1.0, zorder=1)
        ax2.annotate(f"seed {s}", (0, k), xytext=(-8, 0), textcoords="offset points",
                     ha="right", va="center", fontsize=7.5, color=MUTED)
    for x, runs, color in ((0, kaiming, ORANGE), (1, fable, BLUE)):
        ax2.scatter([x] * len(seeds), [runs[s]["test_acc"] * 100 for s in seeds], s=38,
                    color=color, edgecolor=SURFACE, linewidth=1.3, zorder=3)
    ax2.set_xticks([0, 1], ["Kaiming", "LLM-written"])
    ax2.set_xlim(-0.75, 1.35)
    lo = min(r["test_acc"] for runs in (fable, kaiming) for r in runs.values()) * 100
    hi = max(r["test_acc"] for runs in (fable, kaiming) for r in runs.values()) * 100
    ax2.set_ylim(np.floor((lo - 0.25) * 2) / 2, np.ceil((hi + 0.25) * 2) / 2)
    ax2.set_ylabel("test accuracy (%)")
    ax2.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.5))
    panel_label(ax2, "Test accuracy per seed")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_write_training")


FILTER_NAMES = [
    "VARI, smoothed", "green − red", "VARI edges", "dark green",
    "blue − red", "dark blue (water)", "blue texture", "vertical edge",
    "horizontal edge", "45° edge", "135° edge", "horizontal rows",
    "vertical rows", "diagonal rows", "bright spot", "corner",
]


def fig5_write_filters() -> None:
    W = torch.load(WRITE_RUN / "parsed_output" / "first_conv_fable.pt",
                   map_location="cpu", weights_only=True).numpy()
    vmax = float(np.abs(W).max())
    fig = plt.figure(figsize=(7.4, 4.6))
    outer = fig.add_gridspec(4, 4, wspace=0.18, hspace=0.62, bottom=0.12)
    for i in range(16):
        inner = outer[i // 4, i % 4].subgridspec(1, 3, wspace=0.08)
        for c, ch in enumerate("RGB"):
            ax = fig.add_subplot(inner[0, c])
            im = ax.imshow(W[i, c], cmap=DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i < 4:
                ax.text(1, -0.75, ch, ha="center", va="bottom", fontsize=7.5, color=MUTED)
            if c == 1:
                ax.text(1, 2.9, f"{i:02d}  {FILTER_NAMES[i]}", ha="center", va="top",
                        fontsize=7.8, color=INK_2)
    cax = fig.add_axes([0.35, 0.03, 0.3, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.4, 0, 0.4])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, labelsize=8, labelcolor=INK_2)
    cb.set_label("weight value", color=INK_2, fontsize=8)
    save(fig, "fig5_write_filters")


AQUA = "#1baf7a"          # series 3


def _rounds(folder: str) -> list[float]:
    files = sorted((WSP_ROOT / folder / "private").glob("round_*.json"))
    return [json.loads(f.read_text(encoding="utf-8"))["test"]["test_accuracy"] * 100 for f in files]


def fig6_write_no_training() -> None:
    rgb = _rounds("experiment_5_write_code")
    ms = _rounds("experiment_6_write_multispectral")
    w1 = json.loads((WSP_ROOT / "experiment_4_write_full" / "runs" / "w1_attempt_1" / "result.json")
                    .read_text(encoding="utf-8"))["test_accuracy"] * 100
    base_rgb = json.loads((WSP_ROOT / "experiment_4_write_full" / "baselines" / "baselines.json")
                          .read_text(encoding="utf-8"))["centroid"]["test_accuracy"] * 100
    base_ms = json.loads((WSP_ROOT / "experiment_6_write_multispectral" / "private" / "baselines.json")
                         .read_text(encoding="utf-8"))["centroid"] * 100

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    x = np.arange(len(rgb))
    for vals, base, color, name, base_name in ((rgb, base_rgb, ORANGE, "Fable 5.2, RGB (W2)", "mean-colour classifier"),
                                               (ms, base_ms, BLUE, "Fable 5.2, 13 bands (W3)",
                                                "mean-spectrum classifier")):
        ax.axhline(base, color=color, lw=0.9, alpha=0.6, zorder=1)
        ax.text(len(vals) - 1 + 0.35, base, base_name, va="center", fontsize=7.5, color=INK_2,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
        ax.plot(x, vals, color=color, marker="o", ms=6, mec=SURFACE, mew=1.3, zorder=3, label=name)
        ax.annotate(f"{vals[-1]:.1f}", (x[-1], vals[-1]), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=INK)
    ax.scatter([0], [w1], marker="D", s=30, color=ORANGE, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.annotate(f"W1: RGB, one layer, typed by hand ({w1:.1f})", (0, w1), xytext=(8, 7), textcoords="offset points",
                va="center", fontsize=7.5, color=INK_2)
    ax.set_xticks(x, ["0\nno feedback", "1", "2", "3"])
    ax.set_xlim(-0.3, len(rgb) - 1 + 1.75)
    ax.set_ylim(25, 65)
    ax.set_xlabel("feedback rounds")
    ax.set_ylabel("test accuracy without training (%)")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.36))
    save(fig, "fig6_write_no_training")


def fig7_write_init() -> None:
    res = json.loads((WSP_ROOT / "experiment_6_write_multispectral" / "init_compare" / "results.json")
                     .read_text(encoding="utf-8"))
    before = json.loads((WSP_ROOT / "experiment_6_write_multispectral" / "private" / "round_3.json")
                        .read_text(encoding="utf-8"))["test"]["test_accuracy"] * 100
    budgets = [0.01, 0.1, 1.0]
    budget_labels = ["1%\n(163)", "10%\n(1,620)", "100%\n(16,199)"]
    inits = [("kaiming", ORANGE, "Kaiming initialisation"),
             ("fable_as_written", AQUA, "Fable 5.2 network, as written"),
             ("fable_rescaled", BLUE, "Fable 5.2 network, rescaled")]

    def vals(b, key):
        rows = sorted((r for r in res.values() if r["budget"] == b and r["init"] == key), key=lambda r: r["seed"])
        return np.array([r["test_accuracy"] * 100 for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 3.5), width_ratios=[1.45, 1])

    # (a) test accuracy after training: mean ± s.d. over seeds, seeds as dots
    ax1.grid(True, axis="y")
    ax1.set_axisbelow(True)
    ax1.axhline(before, color=INK_2, lw=0.8, zorder=1)
    ax1.text(2.45, before - 0.6, f"Fable 5.2 network\nbefore training: {before:.1f}%", fontsize=7.5,
             color=INK_2, va="top", ha="right", linespacing=1.15)
    off = {0: -0.24, 1: 0.0, 2: 0.24}
    for j, (key, color, name) in enumerate(inits):
        for i, b in enumerate(budgets):
            v = vals(b, key)
            xpos = i + off[j]
            ax1.scatter(xpos - 0.09 + np.linspace(-0.03, 0.03, len(v)), v, s=9, color=color, alpha=0.45, lw=0, zorder=2)
            ax1.errorbar(xpos, v.mean(), yerr=v.std(ddof=1), fmt="o", ms=5.5, color=color, mec=SURFACE, mew=1.0,
                         elinewidth=1.2, capsize=2.5, zorder=3, label=name if i == 0 else None)
    ax1.set_xticks(range(3), budget_labels)
    ax1.set_xlim(-0.5, 2.5)
    ax1.set_ylim(44, 92)
    ax1.set_xlabel("training labels used (images)")
    ax1.set_ylabel("test accuracy after training (%)")
    panel_label(ax1, f"(a) Accuracy, mean ± s.d. over {len(vals(1.0, 'kaiming'))} seeds")

    # (b) paired gain over Kaiming, same seed and same training images
    ax2.grid(True, axis="y")
    ax2.set_axisbelow(True)
    ax2.axhline(0, color=INK_2, lw=0.9, zorder=1)
    for j, (key, color, name) in enumerate(inits[1:]):
        for i, b in enumerate(budgets):
            d = vals(b, key) - vals(b, "kaiming")
            xpos = i + (j - 0.5) * 0.32
            ax2.scatter(xpos + np.linspace(-0.06, 0.06, len(d)), d, s=9, color=color, alpha=0.5, lw=0, zorder=2)
            ax2.hlines(d.mean(), xpos - 0.11, xpos + 0.11, color=color, lw=2.2, zorder=3)
            if key == "fable_rescaled":
                ax2.annotate(f"{d.mean():+.1f}", (xpos, d.max()), xytext=(0, 5), textcoords="offset points",
                             ha="center", fontsize=7.5, color=INK)
    ax2.set_xticks(range(3), budget_labels)
    ax2.set_xlim(-0.5, 2.5)
    ax2.set_ylabel("Fable minus Kaiming (points)")
    ax2.set_xlabel("training labels used (images)")
    panel_label(ax2, "(b) Gain over Kaiming, per seed")

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(w_pad=2.2)
    save(fig, "fig7_write_init")


FIGURES = {
    "fig6": fig6_write_no_training,
    "fig7": fig7_write_init,
    "fig1": lambda: fig1_read_seed_structure(load_read_models()),
    "fig2": lambda: fig2_read_fingerprints(load_read_models()),
    "fig3": fig3_edit_methods,
    "fig4": fig4_write_training,
    "fig5": fig5_write_filters,
}


def main(names: list[str]) -> None:
    for name in names or FIGURES:
        FIGURES[name]()


if __name__ == "__main__":
    main(sys.argv[1:])
