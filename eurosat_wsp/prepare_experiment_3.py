"""Experiment 3 (WRITE): architecture spec, Kaiming baseline, init interface.

Prepares the inputs for generating the *first* convolutional layer (3 -> 16, 3x3,
432 weights + 16 biases) and comparing externally generated weights against the
standard Kaiming initialization.

Writes:
    experiment_3_write/architecture/smallcnn_architecture.json
    experiment_3_write/architecture/first_conv_spec.json
    experiment_3_write/initialization_config/kaiming_baseline.json
    experiment_3_write/initialization_config/interface.json
    experiment_3_write/baseline/smallcnn_kaiming_seed{0,1,2}.pt
    experiment_3_write/baseline/first_conv_kaiming_seed{0,1,2}.pt
    experiment_3_write/baseline/untrained_metrics.json
    experiment_3_write/baseline/kaiming_trained_reference.json

No LLM-generated weights are produced here.

Run:  python eurosat_wsp/prepare_experiment_3.py
"""

from __future__ import annotations

import torch
import torch.nn as nn

import common as wsp

ARCH_DIR = wsp.EXP3_DIR / "architecture"
INIT_DIR = wsp.EXP3_DIR / "initialization_config"
BASE_DIR = wsp.EXP3_DIR / "baseline"

INIT_RULES = {
    "conv1.weight": "kaiming_normal_(mode='fan_out', nonlinearity='relu')",
    "conv1.bias": "zeros",
    "conv weights (features.*.0.weight)": "kaiming_normal_(mode='fan_out', nonlinearity='relu')",
    "conv biases": "none (features convs use bias=False)",
    "batchnorm weight": "ones",
    "batchnorm bias": "zeros",
    "classifier weight": "kaiming_uniform_(a=sqrt(5))  [PyTorch Linear default]",
    "classifier bias": "uniform(-1/sqrt(fan_in), 1/sqrt(fan_in))  [PyTorch Linear default]",
}


def parameter_table(model: nn.Module) -> list[dict]:
    table = []
    for name, param in model.state_dict().items():
        entry = {
            "name": name,
            "shape": list(param.shape),
            "numel": int(param.numel()),
            "dtype": str(param.dtype),
        }
        if name.endswith(".weight") and param.ndim == 4:
            entry["init"] = "kaiming_normal (fan_out, relu)"
        elif name.startswith("conv1."):
            entry["init"] = INIT_RULES[name]
        elif name.endswith("running_mean") or name.endswith("running_var"):
            entry["init"] = "n/a (BatchNorm buffers: mean 0, var 1)"
        elif name.endswith("num_batches_tracked"):
            entry["init"] = "n/a (BatchNorm buffer: 0)"
        elif ".1." in name and name.endswith(".weight"):
            entry["init"] = "ones (BatchNorm)"
        elif ".1." in name and name.endswith(".bias"):
            entry["init"] = "zeros (BatchNorm)"
        elif name.startswith("classifier.2.weight"):
            entry["init"] = INIT_RULES["classifier weight"]
        elif name.startswith("classifier.2.bias"):
            entry["init"] = INIT_RULES["classifier bias"]
        else:
            entry["init"] = "unknown"
        table.append(entry)
    return table


def module_architecture(model: nn.Module) -> list[dict]:
    """Ordered description of the forward pass (spatial sizes at 64x64 input)."""
    stages = [
        {"stage": "conv1", "op": "Conv2d(3, 16, 3, stride=1, padding=1, bias=True)",
         "activation": "none (BatchNorm/ReLU are applied inside stage 1)",
         "output": "16 x 64 x 64"},
        {"stage": "features.0", "op": "Conv2d(16, 32, 3, padding=1, bias=False)+BN+ReLU+MaxPool2d(2)",
         "output": "32 x 32 x 32"},
        {"stage": "features.1", "op": "Conv2d(32, 64, 3, padding=1, bias=False)+BN+ReLU+MaxPool2d(2)",
         "output": "64 x 16 x 16"},
        {"stage": "features.2", "op": "Conv2d(64, 128, 3, padding=1, bias=False)+BN+ReLU+MaxPool2d(2)",
         "output": "128 x 8 x 8"},
        {"stage": "pool", "op": "AdaptiveAvgPool2d(1)", "output": "128 x 1 x 1"},
        {"stage": "classifier", "op": "Flatten + Dropout(0.2) + Linear(128, 10)",
         "output": "10 logits"},
    ]
    return stages


def build_model(seed: int) -> wsp.SmallCNN:
    # Documented, reproducible Kaiming init (common.SmallCNN.reset_parameters).
    return wsp.build_smallcnn(seed=seed)


def main() -> None:
    device = wsp.get_device()
    print(f"device: {device}", flush=True)

    model = build_model(seed=0)
    total = wsp.count_parameters(model)

    architecture = {
        "name": "SmallCNN",
        "experiment": "experiment_3_write (also used unchanged by experiment_1_read)",
        "source": "eurosat_wsp/common.py::SmallCNN",
        "input": {"shape": [3, wsp.IMG_SIZE, wsp.IMG_SIZE],
                  "dtype": "float32", "normalization": "ImageNet mean/std",
                  "mean": list(wsp.MEAN), "std": list(wsp.STD)},
        "channels": list(wsp.CHANNELS),
        "dropout": wsp.TRAIN_CONFIG["dropout"],
        "num_classes": wsp.NUM_CLASSES,
        "class_names": wsp.CLASS_NAMES,
        "output": {"shape": [wsp.NUM_CLASSES], "meaning": "unnormalized class logits"},
        "first_conv": {
            "module": "conv1",
            "state_dict_key": "conv1.weight",
            "weight_shape": [wsp.FIRST_CONV_OUT, 3, 3, 3],
            "num_weights": wsp.FIRST_CONV_OUT * 3 * 3 * 3,
            "bias_shape": [wsp.FIRST_CONV_OUT],
            "num_biases": wsp.FIRST_CONV_OUT,
            "bias_state_dict_key": "conv1.bias",
            "layout": "out_channels x in_channels x kernel_h x kernel_w",
        },
        "forward_stages": module_architecture(model),
        "parameters": {
            "total": total,
            "total_excluding_buffers": int(sum(p.numel() for p in model.parameters())),
            "per_parameter": parameter_table(model),
        },
        "initialization_rules": INIT_RULES,
        "created_at": wsp.now(),
    }
    wsp.write_json(architecture, ARCH_DIR / "smallcnn_architecture.json")

    first_conv = model.conv1.weight.detach()
    wsp.write_json({
        "module": "conv1",
        "purpose": "the single layer targeted by the WRITE experiment",
        "expected_shape": [wsp.FIRST_CONV_OUT, 3, 3, 3],
        "num_weights": int(first_conv.numel()),
        "bias_shape": [wsp.FIRST_CONV_OUT],
        "num_biases": int(model.conv1.bias.numel()),
        "dtype": "float32",
        "value_range_note": "no constraint; weights are used as-is by Conv2d",
        "how_to_supply": [
            "a .pt/.pth file containing either the (16,3,3,3) tensor, a state_dict "
            "with exactly one (16,3,3,3) tensor, or a bare tensor saved via torch.save",
            "a .npy file with a (16,3,3,3) float array (or .npz with one such array)",
        ],
        "interface": "eurosat_wsp/external_init.py",
        "validation": "shape must be (16,3,3,3) and all values finite; "
                      "the 16 biases stay at their initialized zeros unless supplied",
        "kaiming_baseline_stats_seed0": wsp.tensor_stats(first_conv),
        "created_at": wsp.now(),
    }, ARCH_DIR / "first_conv_spec.json")

    # ---- Kaiming baselines, one per seed used by Experiment 1
    untrained = {}
    init_configs = {}
    for seed in wsp.TRAIN_CONFIG["seeds"]:
        model = build_model(seed).to(device)
        state = wsp.state_dict_cpu(model)

        ckpt = BASE_DIR / f"smallcnn_kaiming_seed{seed}.pt"
        wsp.save_state_dict(state, ckpt)
        conv_path = BASE_DIR / f"first_conv_kaiming_seed{seed}.pt"
        torch.save(state["conv1.weight"].clone(), conv_path)

        model.eval()
        loader = wsp.make_loader(
            wsp.ArrayDataset(*wsp.preload_split(wsp.load_raw()["test"]),
                             wsp.build_eval_transform()),
            batch_size=256, shuffle=False)
        loss, acc, pred, true = wsp.evaluate(
            model, loader, nn.CrossEntropyLoss(), device, device.type == "cuda")
        metrics = wsp.classification_metrics(true, pred)
        untrained[str(seed)] = {
            "seed": seed,
            "test_loss": loss,
            "test_accuracy": acc,
            "macro_f1": metrics["macro_f1"],
            "note": "untrained network: accuracy is ~chance (1/10)",
        }
        print(f"seed {seed}: untrained test acc {acc:.3f} | "
              f"init {wsp.tensor_stats(state['conv1.weight'])['std']:.4f} std", flush=True)

        init_configs[str(seed)] = {
            "seed": seed,
            "first_conv": wsp.tensor_stats(state["conv1.weight"]),
            "first_conv_bias": wsp.tensor_stats(state["conv1.bias"]),
            "full_state_dict": f"experiment_3_write/baseline/smallcnn_kaiming_seed{seed}.pt",
            "first_conv_tensor": f"experiment_3_write/baseline/first_conv_kaiming_seed{seed}.pt",
            "state_dict_sha256": wsp.sha256_file(ckpt),
            "first_conv_sha256": wsp.sha256_file(conv_path),
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    wsp.write_json({
        "name": "kaiming_baseline",
        "description": "standard Kaiming-initialized SmallCNN, one per Experiment 1 seed",
        "rules": INIT_RULES,
        "rng": "torch.manual_seed(seed) before construction; reset_parameters() is called "
               "explicitly so the init is identical across processes",
        "seeds": wsp.TRAIN_CONFIG["seeds"],
        "per_seed": init_configs,
        "created_at": wsp.now(),
    }, INIT_DIR / "kaiming_baseline.json")

    wsp.write_json({
        "script": "eurosat_wsp/external_init.py",
        "target_layer": "conv1.weight",
        "expected_shape": [wsp.FIRST_CONV_OUT, 3, 3, 3],
        "num_weights": wsp.FIRST_CONV_OUT * 3 * 3 * 3,
        "accepted_files": [".pt", ".pth", ".npy", ".npz"],
        "usage": [
            "python eurosat_wsp/external_init.py --weights path/to/first_conv.pt "
            "--check-only",
            "python eurosat_wsp/external_init.py --weights path/to/first_conv.pt "
            "--seed 0 --save experiment_3_write/baseline/smallcnn_external_seed0.pt",
        ],
        "python_api": [
            "wsp.load_external_first_conv(path) -> Tensor(16,3,3,3)",
            "wsp.apply_external_first_conv(model, tensor_or_state_dict)",
            "external_init.build_with_external_init(path, seed, save=...) -> SmallCNN",
        ],
        "remaining_layers": "receive the documented Kaiming initialization for `seed`; "
                            "only conv1.weight is replaced",
        "comparison_protocol": "train the resulting model with the Experiment 1 "
                               "'standard' augmentation condition and compare against "
                               "experiment_1_read/metrics/cnn_standard_seed<seed>.json",
        "created_at": wsp.now(),
    }, INIT_DIR / "interface.json")

    wsp.write_json({
        "untrained_baseline": untrained,
        "trained_reference": {
            "source": "experiment_1_read/metrics/cnn_standard_seed0.json",
            "note": "the Kaiming seed-0 baseline trained with the standard augmentation "
                    "condition is exactly Experiment 1's cnn_standard_seed0 model",
        },
        "created_at": wsp.now(),
    }, BASE_DIR / "untrained_metrics.json")

    ref_path = wsp.EXP1_DIR / "metrics" / "cnn_standard_seed0.json"
    if ref_path.exists():
        ref = wsp.read_json(ref_path)
        wsp.write_json({
            "reference_model": "cnn_standard_seed0",
            "init": "Kaiming baseline (seed 0)",
            "condition": "standard augmentation",
            "training": wsp.TRAIN_CONFIG,
            "best_val_acc": ref["best_val_acc"],
            "best_epoch": ref["best_epoch"],
            "val": ref["val"],
            "test": ref["test"],
            "checkpoint": ref["checkpoint"],
            "note": "same initialization and training procedure the LLM-generated "
                    "first conv will later be compared against",
            "created_at": wsp.now(),
        }, BASE_DIR / "kaiming_trained_reference.json")
        print(f"wrote kaiming_trained_reference (test {ref['test']['accuracy']:.3f})")
    else:
        print("WARNING: experiment_1 metrics missing; trained reference not written")

    print("wrote experiment_3_write/ artifacts")


if __name__ == "__main__":
    main()
