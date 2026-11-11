# EuroSAT Weight Space Programming — Preparation Report

Generated 2026-11-05T16:24:22 by `eurosat_wsp/make_report.py`.

Scope: dataset preparation, model training, checkpoint generation and experiment
organisation only. **No LLM interaction is implemented anywhere.**

## 1. Dataset (shared by all experiments)

* Source: `blanchon/EuroSAT_RGB` (Hugging Face Hub), 10 classes,
  native 64×64 RGB patches.
* Splits: the **official** splits are used as-is (train 16,200, validation 5,400, test 5,400) — no re-splitting, and the
  test split is never used for training or model selection.
* Split indices, labels and filenames are frozen in `data/splits/{train,validation,test}.json`
  with sha256 hashes in `data/splits/splits_manifest.json`; every experiment reads them.
* Normalization: ImageNet mean/std (`[0.485, 0.456, 0.406]` /
  `[0.229, 0.224, 0.225]`) after `ToTensor`.
* Classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, SeaLake.
* Augmentation policies used by Experiment 1 are defined once in
  `eurosat_wsp/common.py::AUG_POLICIES` and listed in `data/dataset_config.json`.


## 2. Experiment 1 — READ (training condition from weights)

Same architecture, splits, optimizer, LR, epoch budget and seeds; only the augmentation policy changes. Every model was selected on the validation split and scored once on the held-out test split.

| Model | Augmentation | Seed | Best val acc | Test acc | Macro F1 | Params |
|---|---|---|---|---|---|---|
| `cnn_none_seed0` | no augmentation | 0 | 0.9437 | 0.9444 | 0.9432 | 98,954 |
| `cnn_none_seed1` | no augmentation | 1 | 0.9417 | 0.9469 | 0.9459 | 98,954 |
| `cnn_none_seed2` | no augmentation | 2 | 0.9457 | 0.9465 | 0.9455 | 98,954 |
| `cnn_standard_seed0` | standard augmentation | 0 | 0.9039 | 0.9081 | 0.9070 | 98,954 |
| `cnn_standard_seed1` | standard augmentation | 1 | 0.9013 | 0.9039 | 0.9028 | 98,954 |
| `cnn_standard_seed2` | standard augmentation | 2 | 0.8954 | 0.8985 | 0.8974 | 98,954 |
| `cnn_strong_seed0` | strong augmentation | 0 | 0.7550 | 0.7730 | 0.7596 | 98,954 |
| `cnn_strong_seed1` | strong augmentation | 1 | 0.7420 | 0.7620 | 0.7505 | 98,954 |
| `cnn_strong_seed2` | strong augmentation | 2 | 0.7515 | 0.7735 | 0.7635 | 98,954 |

### Aggregates per condition (n = 3 seeds)

| Condition | Val acc (mean ± sd) | Test acc (mean ± sd) | Test macro F1 (mean ± sd) |
|---|---|---|---|
| no augmentation | 0.9437 ± 0.0020 | 0.9459 ± 0.0013 | 0.9449 ± 0.0014 |
| standard augmentation | 0.9002 ± 0.0044 | 0.9035 ± 0.0048 | 0.9024 ± 0.0048 |
| strong augmentation | 0.7495 ± 0.0067 | 0.7695 ± 0.0065 | 0.7579 ± 0.0067 |

**Observed effect.** At this budget and resolution the ordering is no augmentation (0.946) > standard (0.904) > strong (0.770): random crops on 64x64 patches discard context, and 20 epochs is short for strong augmentation to pay off. That is a property of the sweep, not of the READ task - the three conditions are cleanly separated, which is what identifying a training condition from weights requires. A longer schedule would likely narrow the gap.


Each run is 20 epochs (24.0 s/epoch on this machine), batch 64, AdamW lr 0.001, wd 0.0005, label smoothing 0.1, cosine schedule. Checkpoints are bare `state_dict`s (no metadata), so the training condition cannot leak through the file format; configs and metrics live in separate JSON files.

Per-model training curves (loss/accuracy/LR history) are in `experiment_1_read/metrics/cnn_<condition>_seed<s>.json`.

## 3. Experiment 2 — EDIT (repairing a degraded CNN)

ResNet-18 with a 10-class head, 224×224 inputs (the pretrained model's native resolution). The original checkpoint comes from this project's existing `notebooks/train_resnet18.ipynb` run; it was archived byte-for-byte and its sha256 is pinned, so corruptions can never overwrite the ground truth.

* **Target layer:** `layer3.0.conv1.weight`, shape `[256, 128, 3, 3]` (294,912 weights) — one clearly defined layer for all variants.
* **Corruption seed:** 12345 (fixed for every variant).
* **Original accuracy:** val 0.9802, test 0.9819.

| Variant | Corruption | Weights changed | % of layer | Val acc | Test acc | Δ test |
|---|---|---|---|---|---|---|
| — | original (ground truth) | 0 | 0.0% | 0.9802 | 0.9819 | — |
| A | gaussian_noise | 36,864 | 12.5% | 0.5841 | 0.5767 | -0.4052 |
| B | pruning | 99,552 | 33.8% | 0.6441 | 0.6454 | -0.3365 |
| C | filter_removal | 110,592 | 37.5% | 0.4796 | 0.4867 | -0.4952 |

Exact modified positions are stored as flat index arrays (`corruption_configs/variant_*_modified_indices.npy`, indices into the target tensor in `out_channels × in_channels × kh × kw` order), and the full before/after per-class report is in `experiment_2_edit/metrics/`.

The strengths were chosen by a sweep recorded in `corruption_configs/calibration.json` (target: clearly degraded, still far above the 10% chance level).

## 4. Experiment 3 — WRITE (generating initial weights)

The WRITE experiment targets only the first convolutional layer of the shared small CNN, so the architecture was defined with a 3→16 first conv from the start (the same model Experiment 1 trains).

* Total parameters: **98,954**; first conv: `[16, 3, 3, 3]` = 432 weights + 16 biases.
* Remaining layers: Conv2d(16, 32, 3, padding=1, bias=False)+BN+ReLU+MaxPool2d(2) → Conv2d(32, 64, 3, padding=1, bias=False)+BN+ReLU+MaxPool2d(2) → Conv2d(64, 128, 3, padding=1, bias=False)+BN+ReLU+MaxPool2d(2), then global average pooling, dropout 0.2 and a 128→10 linear head.

| Layer | Shape | Parameters | Initialization |
|---|---|---|---|
| `conv1.weight` | [16, 3, 3, 3] | 432 | kaiming_normal (fan_out, relu) |
| `conv1.bias` | [16] | 16 | zeros |
| `features.0.0.weight` | [32, 16, 3, 3] | 4,608 | kaiming_normal (fan_out, relu) |
| `features.0.1.weight` | [32] | 32 | ones (BatchNorm) |
| `features.0.1.bias` | [32] | 32 | zeros (BatchNorm) |
| `features.0.1.running_mean` | [32] | 32 | n/a (BatchNorm buffers: mean 0, var 1) |
| `features.0.1.running_var` | [32] | 32 | n/a (BatchNorm buffers: mean 0, var 1) |
| `features.1.0.weight` | [64, 32, 3, 3] | 18,432 | kaiming_normal (fan_out, relu) |
| `features.1.1.weight` | [64] | 64 | ones (BatchNorm) |
| `features.1.1.bias` | [64] | 64 | zeros (BatchNorm) |
| `features.1.1.running_mean` | [64] | 64 | n/a (BatchNorm buffers: mean 0, var 1) |
| `features.1.1.running_var` | [64] | 64 | n/a (BatchNorm buffers: mean 0, var 1) |
| `features.2.0.weight` | [128, 64, 3, 3] | 73,728 | kaiming_normal (fan_out, relu) |
| `features.2.1.weight` | [128] | 128 | ones (BatchNorm) |
| `features.2.1.bias` | [128] | 128 | zeros (BatchNorm) |
| `features.2.1.running_mean` | [128] | 128 | n/a (BatchNorm buffers: mean 0, var 1) |
| `features.2.1.running_var` | [128] | 128 | n/a (BatchNorm buffers: mean 0, var 1) |
| `classifier.2.weight` | [10, 128] | 1,280 | kaiming_uniform_(a=sqrt(5))  [PyTorch Linear default] |

### Kaiming baseline (untrained), one per seed

| Seed | First-conv std | Untrained test acc |
|---|---|---|
| 0 | 0.1187 | 0.086 |
| 1 | 0.1179 | 0.113 |
| 2 | 0.1162 | 0.043 |

The initialization interface (`eurosat_wsp/external_init.py`) accepts a `(16, 3, 3, 3)` tensor from `.pt`/`.pth`/`.npy`/`.npz`, validates shape and finiteness, replaces only `conv1.weight`, and leaves every other layer at the documented Kaiming initialization for the chosen seed.

**Trained reference for the later comparison:** the Kaiming seed-0 model under the standard augmentation condition (identical to Experiment 1's `cnn_standard_seed0`) reaches val 0.9039 and test 0.9081. An LLM-generated first conv will be trained with the same procedure and compared against exactly this number.

## 5. Project structure

```text
eurosat_wsp/
common.py  (18 kB)
data/
    dataset_config.json  (2 kB)
    splits/
        splits_manifest.json  (2 kB)
        test.json  (249 kB)
        train.json  (756 kB)
        validation.json  (250 kB)
    training_config.json  (1 kB)
experiment_1_read/
    checkpoints/
        cnn_none_seed0.pt  (405 kB)
        cnn_none_seed1.pt  (405 kB)
        cnn_none_seed2.pt  (405 kB)
        cnn_standard_seed0.pt  (405 kB)
        cnn_standard_seed1.pt  (405 kB)
        cnn_standard_seed2.pt  (405 kB)
        cnn_strong_seed0.pt  (405 kB)
        cnn_strong_seed1.pt  (405 kB)
        cnn_strong_seed2.pt  (405 kB)
    configs/
        cnn_none_seed0.json  (2 kB)
        cnn_none_seed1.json  (2 kB)
        cnn_none_seed2.json  (2 kB)
        cnn_standard_seed0.json  (2 kB)
        cnn_standard_seed1.json  (2 kB)
        cnn_standard_seed2.json  (2 kB)
        cnn_strong_seed0.json  (2 kB)
        cnn_strong_seed1.json  (2 kB)
        cnn_strong_seed2.json  (2 kB)
    metrics/
        cnn_none_seed0.json  (11 kB)
        cnn_none_seed1.json  (11 kB)
        cnn_none_seed2.json  (11 kB)
        cnn_standard_seed0.json  (11 kB)
        cnn_standard_seed1.json  (11 kB)
        cnn_standard_seed2.json  (11 kB)
        cnn_strong_seed0.json  (11 kB)
        cnn_strong_seed1.json  (11 kB)
        cnn_strong_seed2.json  (11 kB)
experiment_2_edit/
    corrupted/
        resnet18_corrupted_a_gaussian_noise.pt  (44.8 MB)
        resnet18_corrupted_b_pruning.pt  (44.8 MB)
        resnet18_corrupted_c_filter_removal.pt  (44.8 MB)
    corruption_configs/
        calibration.json  (8 kB)
        chosen_strengths.json  (1 kB)
        corruption_summary.json  (1 kB)
        variant_a_gaussian_noise.json  (16 kB)
        variant_a_modified_indices.npy  (295 kB)
        variant_b_modified_indices.npy  (797 kB)
        variant_b_pruning.json  (17 kB)
        variant_c_filter_removal.json  (17 kB)
        variant_c_modified_indices.npy  (885 kB)
    metrics/
        corrupted_a.json  (7 kB)
        corrupted_b.json  (7 kB)
        corrupted_c.json  (7 kB)
        original.json  (7 kB)
    original/
        provenance.json  (1 kB)
        resnet18_original.pt  (44.8 MB)
        resnet18_original_state_dict.pt  (44.8 MB)
experiment_3_write/
    architecture/
        first_conv_spec.json  (1 kB)
        smallcnn_architecture.json  (8 kB)
    baseline/
        first_conv_kaiming_seed0.pt  (3 kB)
        first_conv_kaiming_seed1.pt  (3 kB)
        first_conv_kaiming_seed2.pt  (3 kB)
        kaiming_trained_reference.json  (8 kB)
        smallcnn_kaiming_seed0.pt  (405 kB)
        smallcnn_kaiming_seed1.pt  (405 kB)
        smallcnn_kaiming_seed2.pt  (405 kB)
        untrained_metrics.json  (1 kB)
    initialization_config/
        interface.json  (1 kB)
        kaiming_baseline.json  (4 kB)
external_init.py  (4 kB)
make_report.py  (15 kB)
preparation_report.md  (15 kB)
prepare_data.py  (5 kB)
prepare_experiment_2.py  (18 kB)
prepare_experiment_3.py  (12 kB)
run_experiment_1.py  (11 kB)
verification_report.json  (9 kB)
verify.py  (10 kB)
```

## 6. Verification

`python eurosat_wsp/verify.py` — **35/35 checks passed** (2026-11-05T16:23:23).

| # | Check | Result |
|---|---|---|
| 1 | split 'train' matches manifest, 16200 samples, 10 classes | PASS |
| 2 | split 'validation' matches manifest, 5400 samples, 10 classes | PASS |
| 3 | split 'test' matches manifest, 5400 samples, 10 classes | PASS |
| 4 | all 9 Experiment 1 configs share optimizer/LR/epochs/batch | PASS |
| 5 | Experiment 1 differs only in augmentation policy | PASS |
| 6 | Experiment 1 configs all reference the same dataset/splits | PASS |
| 7 | nine Experiment 1 checkpoints present | PASS |
| 8 | cnn_none_seed0.pt: loads, bare state_dict, 10 logits | PASS |
| 9 | cnn_none_seed1.pt: loads, bare state_dict, 10 logits | PASS |
| 10 | cnn_none_seed2.pt: loads, bare state_dict, 10 logits | PASS |
| 11 | cnn_standard_seed0.pt: loads, bare state_dict, 10 logits | PASS |
| 12 | cnn_standard_seed1.pt: loads, bare state_dict, 10 logits | PASS |
| 13 | cnn_standard_seed2.pt: loads, bare state_dict, 10 logits | PASS |
| 14 | cnn_strong_seed0.pt: loads, bare state_dict, 10 logits | PASS |
| 15 | cnn_strong_seed1.pt: loads, bare state_dict, 10 logits | PASS |
| 16 | cnn_strong_seed2.pt: loads, bare state_dict, 10 logits | PASS |
| 17 | original ResNet-18 loads and emits 10 logits | PASS |
| 18 | source checkpoint untouched (sha256 unchanged) | PASS |
| 19 | archived original identical to the training artifact | PASS |
| 20 | original state_dict matches its recorded hash | PASS |
| 21 | corruption target layer is layer3.0.conv1.weight with shape [256, 128, 3, 3] | PASS |
| 22 | three corrupted checkpoints present | PASS |
| 23 | resnet18_corrupted_a_gaussian_noise.pt: differs from original, loads, 10 logits | PASS |
| 24 | resnet18_corrupted_b_pruning.pt: differs from original, loads, 10 logits | PASS |
| 25 | resnet18_corrupted_c_filter_removal.pt: differs from original, loads, 10 logits | PASS |
| 26 | architecture total is 98,954 parameters | PASS |
| 27 | first conv is 3 -> 16, kernel 3x3, 432 weights + 16 biases | PASS |
| 28 | model.conv1.weight is (16, 3, 3, 3) | PASS |
| 29 | interface accepts external (16,3,3,3) weights and the model still runs | PASS |
| 30 | interface rejects a wrong first-conv shape | PASS |
| 31 | interface accepts a state_dict containing the first-conv tensor | PASS |
| 32 | three Kaiming baselines present | PASS |
| 33 | smallcnn_kaiming_seed0.pt: loads into SmallCNN | PASS |
| 34 | smallcnn_kaiming_seed1.pt: loads into SmallCNN | PASS |
| 35 | smallcnn_kaiming_seed2.pt: loads into SmallCNN | PASS |

## 7. Reproducing the preparation

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
