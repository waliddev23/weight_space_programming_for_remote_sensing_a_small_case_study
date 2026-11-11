"""Evaluation & Benchmarking Engine for EuroSAT Weight-Space Programming.

Evaluates Claude's answers and generated weights against the frozen baselines:
1. READ: Validates predictions against blind key, reports accuracy & confidence.
2. WRITE: 
   - Zero-shot evaluation on EuroSAT test set vs. untrained Kaiming (~10%).
   - Full 20-epoch training run vs. trained Kaiming reference (90.8%).
3. EDIT:
   - Executes Python repair function on corrupted ResNet-18.
   - Evaluates test accuracy and computes Recovery % vs ground truth (98.2%).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

# Ensure eurosat_wsp root is in sys.path
WSP_ROOT = Path(__file__).resolve().parent
if str(WSP_ROOT) not in sys.path:
    sys.path.insert(0, str(WSP_ROOT))

import common as wsp
import wsp_parser


def evaluate_read(response_path: Path, key_path: Path) -> dict:
    """Evaluate Claude's READ prediction against the ground-truth blind key (A/B or 9-model)."""
    response_text = Path(response_path).read_text(encoding="utf-8")
    key = wsp.read_json(key_path)

    parsed = wsp_parser.parse_read_response(response_text)

    # 1. 9-Model Sweep Evaluation
    if "models" in key:
        print("\n" + "=" * 60)
        print("=== READ EXPERIMENT: 9-MODEL SWEEP EVALUATION ===")
        print("=" * 60)
        
        preds_9 = parsed["models_9"]
        detailed = {}
        correct_count = 0
        total_models = len(key["models"])

        for model_id, meta in sorted(key["models"].items()):
            truth = meta["condition"]
            pred = preds_9.get(model_id)
            is_correct = (pred == truth)
            if is_correct:
                correct_count += 1
            detailed[model_id] = {
                "source_file": meta["source_file"],
                "ground_truth": truth,
                "predicted": pred,
                "correct": is_correct,
            }
            status_tag = "✓ CORRECT" if is_correct else "✗ WRONG"
            print(f"{model_id} ({meta['source_file']}): Truth [{truth}] vs Pred [{pred}] -> {status_tag}")

        accuracy = correct_count / total_models if total_models > 0 else 0.0
        print("-" * 60)
        print(f"Sweep Accuracy: {correct_count}/{total_models} ({accuracy * 100:.1f}%)")
        print(f"Chance Level Baseline: 33.3% (Random guessing: 1 in 1,680 exact partitions)")

        result = {
            "mode": "9_model_sweep",
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_models": total_models,
            "details": detailed,
            "confidence": parsed["confidence"],
            "evaluated_at": wsp.now(),
        }

        out_file = WSP_ROOT / "experiment_1_read" / "metrics" / "llm_read_9models_result.json"
        wsp.write_json(result, out_file)
        print(f"Logged result to {out_file}\n")
        return result

    # 2. A/B Comparison Evaluation
    true_a = key["model_a"]["condition"]
    true_b = key["model_b"]["condition"]
    
    pred_a = parsed["pred_model_a"]
    pred_b = parsed["pred_model_b"]

    correct_a = (pred_a == true_a) if pred_a else False
    correct_b = (pred_b == true_b) if pred_b else False
    all_correct = bool(correct_a and correct_b)

    result = {
        "mode": "ab_comparison",
        "all_correct": all_correct,
        "model_a": {"predicted": pred_a, "ground_truth": true_a, "correct": correct_a},
        "model_b": {"predicted": pred_b, "ground_truth": true_b, "correct": correct_b},
        "confidence": parsed["confidence"],
        "evaluated_at": wsp.now(),
    }

    print("\n" + "=" * 60)
    print("=== READ EXPERIMENT: A/B EVALUATION ===")
    print("=" * 60)
    print(f"Overall Result: {'✓ PASS (Correct Identification)' if all_correct else '✗ FAIL'}")
    print(f"Model A: Predicted [{pred_a}] | Ground Truth [{true_a}] -> {'CORRECT' if correct_a else 'WRONG'}")
    print(f"Model B: Predicted [{pred_b}] | Ground Truth [{true_b}] -> {'CORRECT' if correct_b else 'WRONG'}")
    if parsed["confidence"] is not None:
        print(f"Model Confidence: {parsed['confidence']:.1f}%")

    out_file = WSP_ROOT / "experiment_1_read" / "metrics" / "llm_read_result.json"
    wsp.write_json(result, out_file)
    print(f"Logged result to {out_file}\n")
    return result


def evaluate_write_zero_shot(weights_path: Path, seed: int = 0) -> dict:
    """Evaluate untrained zero-shot accuracy of Claude's synthesized conv1 weights."""
    print("\n" + "=" * 60)
    print("=== WRITE EXPERIMENT: ZERO-SHOT EVALUATION ===")
    print("=" * 60)

    device = wsp.get_device()
    weights = wsp.load_external_first_conv(weights_path)
    model = wsp.build_smallcnn(seed=seed)
    wsp.apply_external_first_conv(model, weights)
    model.to(device).eval()

    raw = wsp.load_raw()
    test_data = wsp.preload_split(raw["test"])
    test_ds = wsp.ArrayDataset(*test_data, wsp.build_eval_transform())
    test_loader = wsp.make_loader(test_ds, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    loss, acc, y_pred, y_true = wsp.evaluate(model, test_loader, criterion, device, device.type == "cuda")
    metrics = wsp.classification_metrics(y_true, y_pred)

    # Reference untrained Kaiming baselines
    untrained = wsp.read_json(WSP_ROOT / "experiment_3_write" / "baseline" / "untrained_metrics.json")
    kaiming_acc = float(untrained["untrained_baseline"][str(seed)]["test_accuracy"])

    print(f"Claude Synthesized Weights: Test Loss = {loss:.4f} | Test Acc = {acc * 100:.2f}%")
    print(f"Kaiming Baseline (Seed {seed}): Test Acc = {kaiming_acc * 100:.2f}%")
    diff = acc - kaiming_acc
    print(f"Zero-Shot Delta vs Kaiming: {diff * 100:+.2f}%")

    res = {
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "macro_f1": float(metrics["macro_f1"]),
        "kaiming_baseline_acc": kaiming_acc,
        "delta": float(diff),
        "evaluated_at": wsp.now(),
    }
    out_file = WSP_ROOT / "experiment_3_write" / "baseline" / "llm_zero_shot_metrics.json"
    wsp.write_json(res, out_file)
    print(f"Saved zero-shot metrics to {out_file}\n")
    return res


def evaluate_write_train(weights_path: Path, seed: int = 0, epochs: int = 20) -> dict:
    """Train SmallCNN initialized with Claude's conv1 weights for 20 epochs."""
    print("\n" + "=" * 60)
    print(f"=== WRITE EXPERIMENT: 20-EPOCH TRAINING SCHEDULE ===")
    print("=" * 60)

    device = wsp.get_device()
    use_amp = (device.type == "cuda")
    # Match run_experiment_1.py exactly so the Kaiming references are a fair control.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    weights = wsp.load_external_first_conv(weights_path)
    model = wsp.build_smallcnn(seed=seed)
    wsp.apply_external_first_conv(model, weights)
    model.to(device)

    raw = wsp.load_raw()
    data = wsp.preload_all(raw)

    train_ds = wsp.ArrayDataset(*data["train"], wsp.build_train_transform("standard"))
    val_ds = wsp.ArrayDataset(*data["validation"], wsp.build_eval_transform())
    test_ds = wsp.ArrayDataset(*data["test"], wsp.build_eval_transform())

    train_loader = wsp.make_loader(train_ds, batch_size=wsp.TRAIN_CONFIG["batch_size"],
                                   shuffle=True, seed=seed)
    val_loader = wsp.make_loader(val_ds, batch_size=256, shuffle=False)
    test_loader = wsp.make_loader(test_ds, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=wsp.TRAIN_CONFIG["label_smoothing"])
    eval_criterion = nn.CrossEntropyLoss()  # val/test loss without smoothing, as in Experiment 1
    optimizer = wsp.make_optimizer(model, lr=wsp.TRAIN_CONFIG["lr"],
                                  weight_decay=wsp.TRAIN_CONFIG["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    best_val_acc = -1.0
    best_state = None
    history = []

    print(f"Starting training for {epochs} epochs on {device} (AMP={use_amp})...")
    for epoch in range(1, epochs + 1):
        t0 = wsp.time.time()
        tr_loss, tr_acc = wsp.train_one_epoch(model, train_loader, criterion, optimizer,
                                             scaler, device, use_amp)
        val_loss, val_acc, _, _ = wsp.evaluate(model, val_loader, eval_criterion, device, use_amp)
        scheduler.step()
        dt = wsp.time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = wsp.state_dict_cpu(model)

        print(f"Epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.3f} | {dt:.1f}s")

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
        })

    # Evaluate best model on test split
    model.load_state_dict(best_state)
    test_loss, test_acc, y_pred, y_true = wsp.evaluate(model, test_loader, eval_criterion, device, use_amp)
    test_metrics = wsp.classification_metrics(y_true, y_pred)

    # Reference: Experiment 1 Kaiming model with the same seed, standard augmentation
    ref = wsp.read_json(WSP_ROOT / "experiment_1_read" / "metrics" / f"cnn_standard_seed{seed}.json")
    ref_val = float(ref["best_val_acc"])
    ref_test = float(ref["test"]["accuracy"])

    print("\n--- Training Complete ---")
    print(f"Claude Init: Best Val Acc = {best_val_acc * 100:.2f}% | Test Acc = {test_acc * 100:.2f}%")
    print(f"Kaiming Ref: Best Val Acc = {ref_val * 100:.2f}% | Test Acc = {ref_test * 100:.2f}%")
    print(f"Δ Test Acc: {(test_acc - ref_test) * 100:+.2f}%")

    out_payload = {
        "weights_source": str(weights_path),
        "seed": seed,
        "epochs": epochs,
        "per_class": test_metrics.get("per_class"),
        "best_val_acc": float(best_val_acc),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "macro_f1": float(test_metrics["macro_f1"]),
        "kaiming_ref_val": ref_val,
        "kaiming_ref_test": ref_test,
        "history": history,
        "evaluated_at": wsp.now(),
    }
    save_path = (WSP_ROOT / "experiment_3_write" / "trained_runs"
                 / f"{Path(weights_path).stem}_seed{seed}.json")
    wsp.write_json(out_payload, save_path)
    print(f"Saved training history to {save_path}\n")
    return out_payload


def evaluate_edit(repair_script_path: Path, variant: str = "A") -> dict:
    """Execute Claude's Python repair function on the corrupted ResNet-18."""
    print("\n" + "=" * 60)
    print(f"=== EDIT EXPERIMENT: RESNET-18 REPAIR (VARIANT {variant.upper()}) ===")
    print("=" * 60)

    # Load dynamic repair function
    spec = importlib.util.spec_from_file_location("repair_module", repair_script_path)
    repair_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(repair_module)

    if not hasattr(repair_module, "repair_layer3_conv1"):
        raise AttributeError(f"{repair_script_path} does not define 'repair_layer3_conv1'")

    corrupted_ckpt_path = (WSP_ROOT / "experiment_2_edit" / "corrupted" /
                          f"resnet18_corrupted_{variant.lower()}_gaussian_noise.pt")
    if not corrupted_ckpt_path.exists():
        # Match variant by pattern
        for f in (WSP_ROOT / "experiment_2_edit" / "corrupted").glob(f"resnet18_corrupted_{variant.lower()}_*.pt"):
            corrupted_ckpt_path = f
            break

    state_dict = wsp.load_state_dict(corrupted_ckpt_path)
    target_key = "layer3.0.conv1.weight"
    corrupted_tensor = state_dict[target_key].clone()

    print(f"Calling repair function on {target_key} (shape {list(corrupted_tensor.shape)})...")
    repaired_tensor = repair_module.repair_layer3_conv1(corrupted_tensor)

    # Validate repaired tensor
    if tuple(repaired_tensor.shape) != tuple(corrupted_tensor.shape):
        raise ValueError(f"Repair changed tensor shape from {corrupted_tensor.shape} to {repaired_tensor.shape}")
    if not torch.isfinite(repaired_tensor).all():
        raise ValueError("Repaired tensor contains NaN or Inf values!")

    state_dict[target_key] = repaired_tensor

    # Load torchvision ResNet18 and evaluate on 224x224
    from torchvision.models import resnet18
    model = resnet18(num_classes=10)
    model.load_state_dict(state_dict)

    device = wsp.get_device()
    model.to(device).eval()

    # ResNet-18 uses 224x224 resolution
    transform_224 = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(wsp.MEAN, wsp.STD),
    ])

    raw = wsp.load_raw()
    test_ds = wsp.ArrayDataset(*wsp.preload_split(raw["test"]), transform_224)
    test_loader = wsp.make_loader(test_ds, batch_size=128, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    loss, acc, y_pred, y_true = wsp.evaluate(model, test_loader, criterion, device, device.type == "cuda")

    # Read summary baseline numbers
    summary = wsp.read_json(WSP_ROOT / "experiment_2_edit" / "corruption_configs" / "corruption_summary.json")
    orig_acc = summary["original"]["test_accuracy"]
    corrupted_acc = summary["corrupted"][variant.upper()]["test_accuracy"]

    # Recovery formula: (Acc_repaired - Acc_corrupted) / (Acc_original - Acc_corrupted)
    recovery = (acc - corrupted_acc) / (orig_acc - corrupted_acc)

    print("\n--- Repair Results ---")
    print(f"Original Accuracy:  {orig_acc * 100:.2f}%")
    print(f"Corrupted Accuracy: {corrupted_acc * 100:.2f}%")
    print(f"Repaired Accuracy:  {acc * 100:.2f}%")
    print(f"Recovery Metric:    {recovery * 100:+.2f}%")

    res = {
        "variant": variant.upper(),
        "repaired_accuracy": float(acc),
        "original_accuracy": orig_acc,
        "corrupted_accuracy": corrupted_acc,
        "recovery_percentage": float(recovery * 100),
        "evaluated_at": wsp.now(),
    }
    out_file = WSP_ROOT / "experiment_2_edit" / "metrics" / f"llm_repair_{variant.lower()}_result.json"
    wsp.write_json(res, out_file)
    print(f"Saved repair metrics to {out_file}\n")
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="WSP Evaluation Harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # READ
    p_read = subparsers.add_parser("read", help="Evaluate READ prediction against key")
    p_read.add_argument("--response", type=str, required=True, help="Claude's response text file")
    p_read.add_argument("--key", type=str, required=True, help="Path to blind key JSON")

    # WRITE
    p_write = subparsers.add_parser("write", help="Evaluate WRITE synthesized weights")
    p_write.add_argument("--weights", type=str, required=True, help="Path to .pt conv1 weights")
    p_write.add_argument("--seed", type=int, default=0, help="Seed for remaining layers")
    p_write.add_argument("--train", action="store_true", help="Run 20-epoch training schedule")
    p_write.add_argument("--epochs", type=int, default=20, help="Epoch budget")

    # EDIT
    p_edit = subparsers.add_parser("edit", help="Evaluate EDIT repair script on ResNet-18")
    p_edit.add_argument("--repair-script", type=str, required=True, help="Path to Python repair script")
    p_edit.add_argument("--variant", choices=["A", "B", "C"], default="A", help="Corruption variant")

    args = parser.parse_args()

    if args.command == "read":
        evaluate_read(Path(args.response), Path(args.key))
    elif args.command == "write":
        if args.train:
            evaluate_write_train(Path(args.weights), seed=args.seed, epochs=args.epochs)
        else:
            evaluate_write_zero_shot(Path(args.weights), seed=args.seed)
    elif args.command == "edit":
        evaluate_edit(Path(args.repair_script), variant=args.variant)


if __name__ == "__main__":
    main()
