"""Final checks for the EuroSAT WSP preparation.

Verifies, with no training involved:
  1. every checkpoint loads into its architecture,
  2. every model outputs 10 class logits,
  3. the original and corrupted ResNet-18 checkpoints differ,
  4. the original ResNet-18 checkpoint is untouched (sha256 vs provenance),
  5. the first conv layer is 16x3x3x3 / 432 weights / 16 biases and the
     initialization interface accepts external weights,
  6. all experiments used the same splits and the same training configuration
     (only the augmentation policy differs).

Writes eurosat_wsp/verification_report.json and exits non-zero on any failure.

Run:  python eurosat_wsp/verify.py
"""

from __future__ import annotations

import json
import sys

import torch

import common as wsp
from prepare_experiment_2 import IMG_SIZE, TARGET_LAYER, build_resnet18

RESULTS: list[dict] = []


def check(name: str, ok: bool, detail) -> bool:
    RESULTS.append({"check": name, "passed": bool(ok), "detail": detail})
    print(("  PASS  " if ok else "  FAIL  ") + name + ("" if ok else f"  -> {detail}"))
    return bool(ok)


def tensor_diff(a: dict, b: dict) -> dict:
    common = [k for k in a if k in b and a[k].shape == b[k].shape]
    changed_tensors, changed_elements = [], 0
    for k in common:
        n = int((a[k] != b[k]).sum())
        if n:
            changed_tensors.append(k)
            changed_elements += n
    same = all(torch.equal(a[k], b[k]) for k in common)
    return {"tensors_compared": len(common), "changed_tensors": changed_tensors,
            "changed_elements": changed_elements, "bitwise_identical": bool(same)}


def main() -> None:
    print("== 6. splits and shared training configuration")
    manifest = wsp.read_json(wsp.SPLITS_DIR / "splits_manifest.json")
    for split, info in manifest["splits"].items():
        path = wsp.SPLITS_DIR / f"{split}.json"
        payload = wsp.read_json(path)
        ok = (wsp.sha256_file(path) == info["sha256"]
              and len(payload["indices"]) == info["n"]
              and sorted(set(payload["labels"])) == list(range(wsp.NUM_CLASSES)))
        check(f"split '{split}' matches manifest, {info['n']} samples, 10 classes",
              ok, payload.get("split"))

    configs = [wsp.read_json(p) for p in
               sorted((wsp.EXP1_DIR / "configs").glob("cnn_*_seed*.json"))]
    if configs:
        base_training = {k: v for k, v in configs[0]["training"].items()
                         if k not in {"epoch_seconds", "device"}}
        identical = all(
            {k: v for k, v in c["training"].items()
             if k not in {"epoch_seconds", "device"}} == base_training
            for c in configs)
        check(f"all {len(configs)} Experiment 1 configs share optimizer/LR/epochs/batch",
              identical, base_training)
        policies = sorted({c["augmentation"]["policy"] for c in configs})
        check("Experiment 1 differs only in augmentation policy",
              policies == ["none", "standard", "strong"], policies)
        shared_data = all(c["data"]["dataset"] == wsp.HF_DATASET for c in configs)
        check("Experiment 1 configs all reference the same dataset/splits",
              shared_data, wsp.HF_DATASET)

    print("\n== 1-2. Experiment 1 checkpoints load and emit 10 classes")
    ckpts = sorted((wsp.EXP1_DIR / "checkpoints").glob("cnn_*.pt"))
    check("nine Experiment 1 checkpoints present", len(ckpts) == 9,
          [p.name for p in ckpts])
    for path in ckpts:
        raw = torch.load(path, map_location="cpu", weights_only=True)
        is_bare = isinstance(raw, dict) and "model" not in raw
        model = wsp.build_smallcnn(seed=None)
        try:
            model.load_state_dict(raw, strict=True)
            model.eval()
            with torch.no_grad():
                logits = model(torch.rand(2, 3, wsp.IMG_SIZE, wsp.IMG_SIZE))
            ok = (is_bare and tuple(logits.shape) == (2, wsp.NUM_CLASSES)
                  and wsp.count_parameters(model) == 98954)
            detail = {"bare_state_dict": is_bare, "logits": list(logits.shape),
                      "parameters": wsp.count_parameters(model)}
        except Exception as exc:                      # noqa: BLE001 - reported below
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        check(f"{path.name}: loads, bare state_dict, 10 logits", ok, detail)

    print("\n== 2-4. Experiment 2: original vs corrupted ResNet-18")
    original_dir = wsp.EXP2_DIR / "original"
    provenance_path = original_dir / "provenance.json"
    original_path = original_dir / "resnet18_original_state_dict.pt"
    original = wsp.load_state_dict(original_path)
    original_ok = True
    model = build_resnet18()
    try:
        model.load_state_dict(original, strict=True)
        model.eval()
        with torch.no_grad():
            logits = model(torch.rand(2, 3, IMG_SIZE, IMG_SIZE))
        original_ok = tuple(logits.shape) == (2, wsp.NUM_CLASSES)
        detail = {"logits": list(logits.shape), "tensors": len(original)}
    except Exception as exc:                          # noqa: BLE001
        original_ok, detail = False, f"{type(exc).__name__}: {exc}"
    check("original ResNet-18 loads and emits 10 logits", original_ok, detail)

    provenance = wsp.read_json(provenance_path)
    source_hash = wsp.sha256_file(wsp.PROJECT_ROOT / provenance["source_checkpoint"])
    check("source checkpoint untouched (sha256 unchanged)",
          source_hash == provenance["source_sha256"],
          {"expected": provenance["source_sha256"], "actual": source_hash})
    check("archived original identical to the training artifact",
          wsp.sha256_file(original_dir / "resnet18_original.pt") == provenance["source_sha256"],
          provenance["archived_copy"])
    check("original state_dict matches its recorded hash",
          wsp.sha256_file(original_path) == provenance["bare_state_dict_sha256"],
          provenance["bare_state_dict"])
    check(f"corruption target layer is {TARGET_LAYER} with shape [256, 128, 3, 3]",
          tuple(original[TARGET_LAYER].shape) == (256, 128, 3, 3),
          list(original[TARGET_LAYER].shape))

    corrupted = sorted((wsp.EXP2_DIR / "corrupted").glob("*.pt"))
    check("three corrupted checkpoints present", len(corrupted) == 3,
          [p.name for p in corrupted])
    for path in corrupted:
        state = wsp.load_state_dict(path)
        diff = tensor_diff(original, state)
        model = build_resnet18()
        try:
            model.load_state_dict(state, strict=True)
            model.eval()
            with torch.no_grad():
                logits = model(torch.rand(2, 3, IMG_SIZE, IMG_SIZE))
            ok = (not diff["bitwise_identical"] and diff["changed_elements"] > 0
                  and tuple(logits.shape) == (2, wsp.NUM_CLASSES)
                  and set(state) == set(original))
        except Exception as exc:                      # noqa: BLE001
            ok, diff = False, {**diff, "error": f"{type(exc).__name__}: {exc}"}
        check(f"{path.name}: differs from original, loads, 10 logits", ok, diff)

    print("\n== 5. Experiment 3: first conv layer and init interface")
    arch = wsp.read_json(wsp.EXP3_DIR / "architecture" / "smallcnn_architecture.json")
    check("architecture total is 98,954 parameters",
          arch["parameters"]["total"] == 98954, arch["parameters"]["total"])
    conv = arch["first_conv"]
    check("first conv is 3 -> 16, kernel 3x3, 432 weights + 16 biases",
          conv["weight_shape"] == [16, 3, 3, 3] and conv["num_weights"] == 432
          and conv["num_biases"] == 16, conv)

    model = wsp.build_smallcnn(seed=0)
    check("model.conv1.weight is (16, 3, 3, 3)",
          tuple(model.conv1.weight.shape) == (16, 3, 3, 3)
          and model.conv1.weight.numel() == 432
          and model.conv1.bias.numel() == 16,
          list(model.conv1.weight.shape))

    external = torch.randn(16, 3, 3, 3) * 0.1
    wsp.apply_external_first_conv(model, external)
    accepted = torch.equal(model.conv1.weight.detach().cpu(), external)
    with torch.no_grad():
        logits = model(torch.rand(2, 3, wsp.IMG_SIZE, wsp.IMG_SIZE))
    check("interface accepts external (16,3,3,3) weights and the model still runs",
          accepted and tuple(logits.shape) == (2, wsp.NUM_CLASSES),
          {"weights_replaced": bool(accepted), "logits": list(logits.shape)})

    rejected = False
    try:
        wsp.apply_external_first_conv(model, torch.randn(16, 3, 2, 2))
    except ValueError:
        rejected = True
    check("interface rejects a wrong first-conv shape", rejected, "no ValueError raised")

    state_dict_input = {"conv1.weight": torch.randn(16, 3, 3, 3)}
    model2 = wsp.build_smallcnn(seed=1)
    wsp.apply_external_first_conv(model2, state_dict_input)
    check("interface accepts a state_dict containing the first-conv tensor",
          torch.equal(model2.conv1.weight.detach().cpu(), state_dict_input["conv1.weight"]),
          list(state_dict_input))

    baselines = sorted((wsp.EXP3_DIR / "baseline").glob("smallcnn_kaiming_seed*.pt"))
    check("three Kaiming baselines present", len(baselines) == 3,
          [p.name for p in baselines])
    for path in baselines:
        state = wsp.load_state_dict(path)
        m = wsp.build_smallcnn(seed=None)
        try:
            m.load_state_dict(state, strict=True)
            ok = tuple(state["conv1.weight"].shape) == (16, 3, 3, 3)
            detail = {"first_conv": list(state["conv1.weight"].shape)}
        except Exception as exc:                      # noqa: BLE001
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        check(f"{path.name}: loads into SmallCNN", ok, detail)

    failures = [r for r in RESULTS if not r["passed"]]
    payload = {
        "generated_at": wsp.now(),
        "device": str(wsp.get_device()),
        "checks_total": len(RESULTS),
        "checks_passed": len(RESULTS) - len(failures),
        "checks_failed": len(failures),
        "all_passed": not failures,
        "results": RESULTS,
    }
    wsp.write_json(payload, wsp.WSP_ROOT / "verification_report.json")
    print(f"\n{payload['checks_passed']}/{payload['checks_total']} checks passed")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
