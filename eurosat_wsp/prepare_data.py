"""Experiment preparation, step 1: freeze the dataset and the splits.

Uses the official `blanchon/EuroSAT_RGB` splits (train 16200 / validation 5400 /
test 5400) as-is - no re-splitting, so every experiment reads exactly the same
images and the test split stays untouched by model selection.

Writes:
    data/dataset_config.json
    data/training_config.json
    data/splits/{train,validation,test}.json      (indices, labels, filenames)
    data/splits/splits_manifest.json              (counts, class balance, hashes)

Run:  python eurosat_wsp/prepare_data.py
"""

from __future__ import annotations

import numpy as np

import common as wsp


def main() -> None:
    raw = wsp.load_raw()

    names = raw["train"].features["label"].names
    assert names == wsp.CLASS_NAMES, f"unexpected class order: {names}"

    manifest_splits = {}
    for split in ("train", "validation", "test"):
        hf = raw[split]
        labels = np.asarray(hf["label"], dtype=np.int64)
        filenames = list(hf["filename"])
        indices = list(range(len(labels)))

        payload = {
            "source": wsp.HF_DATASET,
            "split": split,
            "n": len(indices),
            "index_space": "positional index into the official HF split, 0..n-1",
            "indices": indices,
            "labels": labels.tolist(),
            "filenames": filenames,
        }
        path = wsp.write_json(payload, wsp.SPLITS_DIR / f"{split}.json")

        counts = {name: int((labels == i).sum()) for i, name in enumerate(wsp.CLASS_NAMES)}
        manifest_splits[split] = {
            "n": len(indices),
            "sha256": wsp.sha256_file(path),
            "class_counts": counts,
            "first_label": int(labels[0]),
            "last_label": int(labels[-1]),
        }
        print(f"{split:10s} n={len(indices):6d}  sha256={manifest_splits[split]['sha256'][:16]}...")

    wsp.write_json(
        {
            "dataset": wsp.HF_DATASET,
            "source": "Hugging Face Hub (official EuroSAT_RGB release)",
            "official_splits": True,
            "num_classes": wsp.NUM_CLASSES,
            "class_names": wsp.CLASS_NAMES,
            "class_to_index": wsp.CLASS_TO_INDEX,
            "label_column": "label",
            "image_size": wsp.IMG_SIZE,
            "channels": 3,
            "dtype_on_disk": "uint8 RGB, native 64x64 (no resizing for the small CNN)",
            "normalization": {
                "mean": list(wsp.MEAN),
                "std": list(wsp.STD),
                "note": "ImageNet statistics, shared by the CNN and the ResNet-18",
            },
            "augmentation_policies": {
                policy: wsp.describe_policy(policy) for policy in wsp.AUG_POLICIES
            },
            "eval_transform": "ToTensor + Normalize (no resize, no crop, no flip)",
            "splits": {
                "train": "official train (16,200) - the only split used for gradients",
                "validation": "official validation (5,400) - model selection only",
                "test": "official test (5,400) - held out, used once per final model",
            },
            "split_files": {
                split: f"data/splits/{split}.json"
                for split in ("train", "validation", "test")
            },
            "created_at": wsp.now(),
        },
        wsp.DATA_DIR / "dataset_config.json",
    )

    wsp.write_json(
        {
            "shared_by": "experiment_1_read (all conditions and seeds)",
            **wsp.TRAIN_CONFIG,
            "loss": "CrossEntropyLoss(label_smoothing=0.1)",
            "optimizer_param_groups": "decay on conv/linear weights, no decay on BN/bias",
            "seed_controls": (
                "the same seed is used for weight init, DataLoader shuffle order and "
                "augmentation RNG, so conditions differ only in the augmentation policy"
            ),
            "checkpoint_format": (
                "torch.save(state_dict) on CPU - a bare {name: tensor} mapping with no "
                "metadata; training metadata lives in configs/ and metrics/"
            ),
            "num_workers": 0,
            "created_at": wsp.now(),
        },
        wsp.DATA_DIR / "training_config.json",
    )

    wsp.write_json(
        {
            "dataset": wsp.HF_DATASET,
            "splits": manifest_splits,
            "test_isolation": (
                "the test split is never used for training, augmentation statistics, "
                "early stopping or checkpoint selection"
            ),
            "created_at": wsp.now(),
        },
        wsp.SPLITS_DIR / "splits_manifest.json",
    )
    print("wrote", wsp.DATA_DIR / "dataset_config.json")


if __name__ == "__main__":
    main()
