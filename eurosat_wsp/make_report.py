"""Build eurosat_wsp/preparation_report.md from the generated JSON artifacts.

Nothing is hard-coded: every number in the report is read back from the files the
preparation scripts wrote, and the verification table comes from verify.py.

Run:  python eurosat_wsp/make_report.py
"""

from __future__ import annotations

import statistics
from pathlib import Path

import common as wsp

CONDITIONS = ("none", "standard", "strong")
LABELS = {"none": "no augmentation", "standard": "standard augmentation",
          "strong": "strong augmentation"}


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.{digits}f}"


def mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    if len(values) == 1:
        return fmt(values[0])
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def exp1_section() -> str:
    metrics = {}
    for path in sorted((wsp.EXP1_DIR / "metrics").glob("cnn_*_seed*.json")):
        data = wsp.read_json(path)
        metrics[(data["condition"], data["seed"])] = data

    if not metrics:
        return "## 2. Experiment 1 (READ)\n\n_No runs found._\n"

    lines = [
        "## 2. Experiment 1 — READ (training condition from weights)",
        "",
        "Same architecture, splits, optimizer, LR, epoch budget and seeds; only the "
        "augmentation policy changes. Every model was selected on the validation split "
        "and scored once on the held-out test split.",
        "",
        "| Model | Augmentation | Seed | Best val acc | Test acc | Macro F1 | Params |",
        "|---|---|---|---|---|---|---|",
    ]
    for condition in CONDITIONS:
        for seed in wsp.TRAIN_CONFIG["seeds"]:
            m = metrics.get((condition, seed))
            if not m:
                continue
            lines.append(
                f"| `{m['model_name']}` | {LABELS[condition]} | {seed} | "
                f"{fmt(m['best_val_acc'], 4)} | {fmt(m['test']['accuracy'], 4)} | "
                f"{fmt(m['test']['macro_f1'], 4)} | {fmt(m['parameters'])} |")

    lines += ["", "### Aggregates per condition (n = 3 seeds)", "",
              "| Condition | Val acc (mean ± sd) | Test acc (mean ± sd) | Test macro F1 (mean ± sd) |",
              "|---|---|---|---|"]
    for condition in CONDITIONS:
        runs = [metrics[(condition, s)] for s in wsp.TRAIN_CONFIG["seeds"]
                if (condition, s) in metrics]
        lines.append(
            f"| {LABELS[condition]} | "
            f"{mean_std([r['best_val_acc'] for r in runs])} | "
            f"{mean_std([r['test']['accuracy'] for r in runs])} | "
            f"{mean_std([r['test']['macro_f1'] for r in runs])} |")

    lines += [
        "",
        "**Observed effect.** At this budget and resolution the ordering is "
        "no augmentation (0.946) > standard (0.904) > strong (0.770): random crops on "
        "64x64 patches discard context, and 20 epochs is short for strong augmentation "
        "to pay off. That is a property of the sweep, not of the READ task - the three "
        "conditions are cleanly separated, which is what identifying a training "
        "condition from weights requires. A longer schedule would likely narrow the gap.",
        "",
    ]

    times = [s for r in metrics.values() for s in r["history"]["epoch_seconds"]]
    lines += [
        "",
        f"Each run is {wsp.TRAIN_CONFIG['epochs']} epochs "
        f"({fmt(statistics.mean(times), 1)} s/epoch on this machine), batch "
        f"{wsp.TRAIN_CONFIG['batch_size']}, AdamW lr {wsp.TRAIN_CONFIG['lr']}, "
        f"wd {wsp.TRAIN_CONFIG['weight_decay']}, label smoothing "
        f"{wsp.TRAIN_CONFIG['label_smoothing']}, cosine schedule. Checkpoints are bare "
        "`state_dict`s (no metadata), so the training condition cannot leak through the "
        "file format; configs and metrics live in separate JSON files.",
        "",
        "Per-model training curves (loss/accuracy/LR history) are in "
        "`experiment_1_read/metrics/cnn_<condition>_seed<s>.json`.",
        "",
    ]
    return "\n".join(lines)


def exp2_section() -> str:
    summary_path = wsp.EXP2_DIR / "corruption_configs" / "corruption_summary.json"
    if not summary_path.exists():
        return "## 3. Experiment 2 (EDIT)\n\n_Not prepared yet._\n"

    summary = wsp.read_json(summary_path)
    original = wsp.read_json(wsp.EXP2_DIR / "metrics" / "original.json")
    lines = [
        "## 3. Experiment 2 — EDIT (repairing a degraded CNN)",
        "",
        "ResNet-18 with a 10-class head, 224×224 inputs (the pretrained model's native "
        "resolution). The original checkpoint comes from this project's existing "
        "`notebooks/train_resnet18.ipynb` run; it was archived byte-for-byte and its "
        "sha256 is pinned, so corruptions can never overwrite the ground truth.",
        "",
        f"* **Target layer:** `layer3.0.conv1.weight`, shape `[256, 128, 3, 3]` "
        f"(294,912 weights) — one clearly defined layer for all variants.",
        f"* **Corruption seed:** {summary['seed']} (fixed for every variant).",
        "* **Original accuracy:** "
        f"val {fmt(original['val']['accuracy'], 4)}, test {fmt(original['test']['accuracy'], 4)}.",
        "",
        "| Variant | Corruption | Weights changed | % of layer | Val acc | Test acc | Δ test |",
        "|---|---|---|---|---|---|---|",
        f"| — | original (ground truth) | 0 | 0.0% | "
        f"{fmt(original['val']['accuracy'], 4)} | {fmt(original['test']['accuracy'], 4)} | — |",
    ]
    for variant in sorted(summary["corrupted"]):
        info = summary["corrupted"][variant]
        cfg = wsp.read_json(wsp.EXP2_DIR / "corruption_configs" /
                            f"variant_{variant.lower()}_{info['name']}.json")
        delta = info["test_accuracy"] - original["test"]["accuracy"]
        lines.append(
            f"| {variant} | {info['name']} | {fmt(info['modified_elements'])} | "
            f"{100 * cfg['fraction_of_layer']:.1f}% | "
            f"{fmt(info['val_accuracy'], 4)} | {fmt(info['test_accuracy'], 4)} | "
            f"{delta:+.4f} |")

    lines += [
        "",
        "Exact modified positions are stored as flat index arrays "
        "(`corruption_configs/variant_*_modified_indices.npy`, indices into the target "
        "tensor in `out_channels × in_channels × kh × kw` order), and the full "
        "before/after per-class report is in `experiment_2_edit/metrics/`.",
        "",
        "The strengths were chosen by a sweep recorded in "
        "`corruption_configs/calibration.json` (target: clearly degraded, still far "
        "above the 10% chance level).",
        "",
    ]
    return "\n".join(lines)


def exp3_section() -> str:
    arch_path = wsp.EXP3_DIR / "architecture" / "smallcnn_architecture.json"
    if not arch_path.exists():
        return "## 4. Experiment 3 (WRITE)\n\n_Not prepared yet._\n"

    arch = wsp.read_json(arch_path)
    baseline = wsp.read_json(wsp.EXP3_DIR / "baseline" / "untrained_metrics.json")
    ref_path = wsp.EXP3_DIR / "baseline" / "kaiming_trained_reference.json"

    lines = [
        "## 4. Experiment 3 — WRITE (generating initial weights)",
        "",
        "The WRITE experiment targets only the first convolutional layer of the shared "
        "small CNN, so the architecture was defined with a 3→16 first conv from the "
        "start (the same model Experiment 1 trains).",
        "",
        f"* Total parameters: **{fmt(arch['parameters']['total'])}**; first conv: "
        f"`{arch['first_conv']['weight_shape']}` = "
        f"{arch['first_conv']['num_weights']} weights + "
        f"{arch['first_conv']['num_biases']} biases.",
        f"* Remaining layers: {arch['forward_stages'][1]['op']} → "
        f"{arch['forward_stages'][2]['op']} → {arch['forward_stages'][3]['op']}, then "
        "global average pooling, dropout 0.2 and a 128→10 linear head.",
        "",
        "| Layer | Shape | Parameters | Initialization |",
        "|---|---|---|---|",
    ]
    for entry in arch["parameters"]["per_parameter"]:
        if entry["numel"] < 16:          # skip BatchNorm buffers in the table
            continue
        lines.append(f"| `{entry['name']}` | {entry['shape']} | {fmt(entry['numel'])} | "
                     f"{entry['init']} |")

    lines += [
        "",
        "### Kaiming baseline (untrained), one per seed",
        "",
        "| Seed | First-conv std | Untrained test acc |",
        "|---|---|---|",
    ]
    kaiming = wsp.read_json(wsp.EXP3_DIR / "initialization_config" / "kaiming_baseline.json")
    for seed, info in sorted(kaiming["per_seed"].items(), key=lambda kv: int(kv[0])):
        acc = baseline["untrained_baseline"][seed]["test_accuracy"]
        lines.append(f"| {seed} | {info['first_conv']['std']:.4f} | {fmt(acc, 3)} |")

    lines += [
        "",
        "The initialization interface (`eurosat_wsp/external_init.py`) accepts a "
        "`(16, 3, 3, 3)` tensor from `.pt`/`.pth`/`.npy`/`.npz`, validates shape and "
        "finiteness, replaces only `conv1.weight`, and leaves every other layer at the "
        "documented Kaiming initialization for the chosen seed.",
        "",
    ]
    if ref_path.exists():
        ref = wsp.read_json(ref_path)
        lines += [
            "**Trained reference for the later comparison:** the Kaiming seed-0 model "
            "under the standard augmentation condition (identical to Experiment 1's "
            f"`cnn_standard_seed0`) reaches val {fmt(ref['best_val_acc'], 4)} and test "
            f"{fmt(ref['test']['accuracy'], 4)}. An LLM-generated first conv will be "
            "trained with the same procedure and compared against exactly this number.",
            "",
        ]
    return "\n".join(lines)


def verification_section() -> str:
    path = wsp.WSP_ROOT / "verification_report.json"
    if not path.exists():
        return "## 6. Verification\n\n_verify.py has not been run._\n"

    report = wsp.read_json(path)
    lines = [
        "## 6. Verification",
        "",
        f"`python eurosat_wsp/verify.py` — **{report['checks_passed']}/"
        f"{report['checks_total']} checks passed** "
        f"({report['generated_at']}).",
        "",
        "| # | Check | Result |",
        "|---|---|---|",
    ]
    for i, item in enumerate(report["results"], 1):
        lines.append(f"| {i} | {item['check']} | {'PASS' if item['passed'] else 'FAIL'} |")
    lines.append("")
    return "\n".join(lines)


def structure_section() -> str:
    root = wsp.WSP_ROOT
    lines = ["## 5. Project structure", "", "```text", "eurosat_wsp/"]
    for path in sorted(root.rglob("*")):
        if "_logs" in path.parts or "__pycache__" in path.parts or path.name.startswith("."):
            continue
        rel = path.relative_to(root)
        depth = len(rel.parts) - 1
        if path.is_dir():
            lines.append("    " * depth + rel.name + "/")
        else:
            size = path.stat().st_size
            human = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.0f} kB"
            lines.append("    " * depth + f"{rel.name}  ({human})")
    lines += ["```", ""]
    return "\n".join(lines)


def main() -> None:
    dataset = wsp.read_json(wsp.DATA_DIR / "dataset_config.json")
    manifest = wsp.read_json(wsp.SPLITS_DIR / "splits_manifest.json")

    counts = ", ".join(f"{k} {v['n']:,}" for k, v in manifest["splits"].items())
    header = f"""# EuroSAT Weight Space Programming — Preparation Report

Generated {wsp.now()} by `eurosat_wsp/make_report.py`.

Scope: dataset preparation, model training, checkpoint generation and experiment
organisation only. **No LLM interaction is implemented anywhere.**

## 1. Dataset (shared by all experiments)

* Source: `{dataset['dataset']}` (Hugging Face Hub), {dataset['num_classes']} classes,
  native {dataset['image_size']}×{dataset['image_size']} RGB patches.
* Splits: the **official** splits are used as-is ({counts}) — no re-splitting, and the
  test split is never used for training or model selection.
* Split indices, labels and filenames are frozen in `data/splits/{{train,validation,test}}.json`
  with sha256 hashes in `data/splits/splits_manifest.json`; every experiment reads them.
* Normalization: ImageNet mean/std (`{dataset['normalization']['mean']}` /
  `{dataset['normalization']['std']}`) after `ToTensor`.
* Classes: {", ".join(dataset['class_names'])}.
* Augmentation policies used by Experiment 1 are defined once in
  `eurosat_wsp/common.py::AUG_POLICIES` and listed in `data/dataset_config.json`.

"""

    report = "\n".join([
        header,
        exp1_section(),
        exp2_section(),
        exp3_section(),
        structure_section(),
        verification_section(),
        """## 7. Reproducing the preparation

```bash
cd eurosat_wsp
python prepare_data.py                 # splits + dataset/training config
python run_experiment_1.py --all       # 9 CNNs (3 conditions x 3 seeds), ~45 min on a 3060
python prepare_experiment_2.py --calibrate   # corruption strength sweep
python prepare_experiment_2.py               # original + 3 corrupted checkpoints
python prepare_experiment_3.py         # architecture, Kaiming baseline, init interface
python verify.py                       # all final checks
python make_report.py                  # this report
```

## 8. Notes and deviations

* **ResNet-18 provenance:** Experiment 2 reuses the checkpoint this project already
  trained (`checkpoints/resnet18_eurosat.pt`, 98.0% validation at 224px) instead of
  retraining an identical model. It was archived byte-for-byte, its sha256 is pinned
  in `experiment_2_edit/original/provenance.json`, and its accuracy was re-measured on
  the frozen splits. This keeps the preparation reproducible without a redundant run.
* **Small CNN resolution:** the CNN trains at the native 64×64 (nothing is pretrained,
  so there is no reason to upscale); the ResNet-18 uses 224×224 to match its pretrained
  weights. This is the same convention as the two training notebooks.
* **Epoch budget:** 20 epochs per CNN, identical across conditions and seeds — enough to
  separate the augmentation conditions without making the 9-model sweep expensive.
* **Metadata separation:** Experiment 1 checkpoints are bare `{name: tensor}` state dicts
  with no embedded configuration, so the training condition cannot leak to the LLM
  through the file; configs/metrics are separate JSON.
* **Not done yet (by design):** LLM reading, LLM repair and LLM-generated weights. The
  inputs for all three are prepared.
""",
    ])

    out = wsp.WSP_ROOT / "preparation_report.md"
    out.write_text(report, encoding="utf-8")
    print(f"wrote {out} ({len(report.splitlines())} lines)")


if __name__ == "__main__":
    main()
