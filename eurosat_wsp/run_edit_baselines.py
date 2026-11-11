"""Evaluate every tensor-only repair method on EDIT variants A, B and C.

Usage (from the repository root or anywhere):
    python eurosat_wsp/run_edit_baselines.py                 # all variants, all methods
    python eurosat_wsp/run_edit_baselines.py --variants A    # one variant

Also evaluates the repair script submitted in a run directory for variant A
(``--submitted``), so the headline number is reproduced by the same code path.

Writes experiment_2_edit/metrics/edit_baselines.json. The original checkpoint and
the modified-index files are never read; reference accuracies come from
corruption_configs/corruption_summary.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

WSP_ROOT = Path(__file__).resolve().parent
if str(WSP_ROOT) not in sys.path:
    sys.path.insert(0, str(WSP_ROOT))

import common as wsp
import edit_repairs

TARGET = "layer3.0.conv1.weight"
VARIANT_FILES = {
    "A": "resnet18_corrupted_a_gaussian_noise.pt",
    "B": "resnet18_corrupted_b_pruning.pt",
    "C": "resnet18_corrupted_c_filter_removal.pt",
}
DEFAULT_SUBMITTED = WSP_ROOT / "runs" / "edit_run_001" / "parsed_output" / "repair_script.py"


def load_repair_script(path: Path):
    spec = importlib.util.spec_from_file_location("submitted_repair", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.repair_layer3_conv1


def build_test_loader():
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
                             transforms.Normalize(wsp.MEAN, wsp.STD)])
    raw = wsp.load_raw()
    ds = wsp.ArrayDataset(*wsp.preload_split(raw["test"]), tf)
    return wsp.make_loader(ds, batch_size=128, shuffle=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+", default=list(VARIANT_FILES), choices=list(VARIANT_FILES))
    ap.add_argument("--methods", nargs="+", default=list(edit_repairs.METHODS),
                    choices=list(edit_repairs.METHODS))
    ap.add_argument("--submitted", type=Path, default=DEFAULT_SUBMITTED,
                    help="repair script evaluated on variant A as 'submitted'")
    ap.add_argument("--out", type=Path,
                    default=WSP_ROOT / "experiment_2_edit" / "metrics" / "edit_baselines.json")
    args = ap.parse_args()

    device = wsp.get_device()
    use_amp = device.type == "cuda"
    loader = build_test_loader()
    criterion = nn.CrossEntropyLoss()
    summary = wsp.read_json(WSP_ROOT / "experiment_2_edit" / "corruption_configs" / "corruption_summary.json")
    orig_acc = summary["original"]["test_accuracy"]

    model = resnet18(num_classes=10).to(device).eval()
    results = {"original_test_accuracy": orig_acc, "variants": {}}

    for v in args.variants:
        sd = wsp.load_state_dict(WSP_ROOT / "experiment_2_edit" / "corrupted" / VARIANT_FILES[v])
        corrupted = sd[TARGET].clone()
        corrupted_acc = summary["corrupted"][v]["test_accuracy"]
        methods = {name: edit_repairs.METHODS[name] for name in args.methods}
        if v == "A" and args.submitted and args.submitted.exists():
            methods["submitted"] = load_repair_script(args.submitted)

        rows = {}
        for name, fn in methods.items():
            t0 = time.time()
            repaired = fn(corrupted.clone())
            if repaired.shape != corrupted.shape or not torch.isfinite(repaired).all():
                raise ValueError(f"{name} returned an invalid tensor on variant {v}")
            sd[TARGET] = repaired
            model.load_state_dict(sd)
            _, acc, _, _ = wsp.evaluate(model, loader, criterion, device, use_amp)
            recovery = (acc - corrupted_acc) / (orig_acc - corrupted_acc)
            rows[name] = {"test_accuracy": float(acc), "recovery": float(recovery)}
            print(f"[{v}] {name:18s} acc {acc * 100:6.2f}%  recovery {recovery * 100:+7.1f}%  "
                  f"({time.time() - t0:.0f}s)", flush=True)

        results["variants"][v] = {"name": summary["corrupted"][v]["name"],
                                  "corrupted_test_accuracy": corrupted_acc, "methods": rows}

    results["evaluated_at"] = wsp.now()
    wsp.write_json(results, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
