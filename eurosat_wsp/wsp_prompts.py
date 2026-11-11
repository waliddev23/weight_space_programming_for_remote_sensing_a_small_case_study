"""Prompt Generator Engine for EuroSAT Weight-Space Programming.

Constructs ready-to-use prompts for Claude (1M context) across:
1. READ (Weight Introspection: Focused & Full 1M)
2. WRITE (Zero-Shot Synthesis of conv1 weights)
3. EDIT (Surgical Weight Repair of corrupted ResNet-18)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure eurosat_wsp root is in sys.path
WSP_ROOT = Path(__file__).resolve().parent
if str(WSP_ROOT) not in sys.path:
    sys.path.insert(0, str(WSP_ROOT))

import common as wsp
from wsp_serializer import create_blind_ab_comparison, create_blind_9_models_sweep


PROMPTS_DIR = WSP_ROOT / "prompts"


def generate_read_9_models_prompt(serialized_sweep: str) -> str:
    """Generate prompt for classifying all 9 trained CNN models into their 3 conditions."""
    prompt = f"""You are an expert in neural network mechanistic interpretability and weight-space analysis.

# Context: EuroSAT Satellite Land Cover Classification Benchmark
Nine identical convolutional neural networks (SmallCNN) were trained from scratch on the EuroSAT_RGB dataset (10 classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, SeaLake) at native 64x64 resolution.

All 9 models share the EXACT SAME:
- Architecture (SmallCNN, 98,954 parameters: 4 conv stages + global average pooling + linear head)
- Dataset train/val/test splits (16,200 train / 5,400 val / 5,400 test)
- Optimizer (AdamW, lr=1e-3, weight_decay=5e-4, 20 epochs, cosine schedule)

# The Three Regularization Conditions (3 models per condition):
The 9 models represent exactly three experimental conditions, each trained across 3 random seeds:
1. "no_augmentation" (3 models): Trained strictly on raw unaugmented patches. Memorizes native spatial coordinates and exact pixel RGB values. Test accuracy: ~94.6%.
2. "standard_augmentation" (3 models): Moderate regularization (RandomResizedCrop scale 0.7-1.0, horizontal & vertical flips, moderate color jitter). Test accuracy: ~90.4%.
3. "strong_augmentation" (3 models): Heavy regularization (RandomResizedCrop scale 0.45-1.0, horizontal/vertical flips, 90-degree rotations, heavy color jitter, random grayscale p=0.15, Gaussian blur p=0.30). Test accuracy: ~77.0%.

# Your Task
Below are the serialized weights (conv1: 16x3x3x3 + classifier: 10x128) for nine models labeled anonymously as:
**Model_01** through **Model_09** (shuffled in random order).

For each of the 9 models, classify its training condition into one of:
- `no_augmentation`
- `standard_augmentation`
- `strong_augmentation`

Exactly 3 models belong to each condition.

# Analytical Guidelines:
1. **Spatial Isotropy & Rotational Symmetry:**
   - In `conv1.weight`, examine whether the 3x3 spatial filters show 90-degree rotational symmetry or diagonal/isotropic spreading (indicative of `strong_augmentation`), or sharp axis-aligned vertical/horizontal edge detectors (`no_augmentation`).
2. **Spectral / Channel Divergence:**
   - Examine the variance between R, G, and B channel kernels. Heavy color jitter and grayscale suppress color divergence between channels.
3. **Weight Norms & Classifier Head:**
   - Regularization alters the spectral norm and Frobenius norm of feature and classifier weights.

# Output Format Required
Please structure your response with:
1. **Per-Model Predictions:**
   - Model_01: [no_augmentation | standard_augmentation | strong_augmentation] (Confidence: X%)
   - Model_02: [no_augmentation | standard_augmentation | strong_augmentation] (Confidence: X%)
   ...
   - Model_09: [no_augmentation | standard_augmentation | strong_augmentation] (Confidence: X%)

2. **Group Summary:**
   - Models in no_augmentation: [Model_XX, Model_YY, Model_ZZ]
   - Models in standard_augmentation: [Model_XX, Model_YY, Model_ZZ]
   - Models in strong_augmentation: [Model_XX, Model_YY, Model_ZZ]

3. **Mechanistic Reasoning:**
   Detailed breakdown citing specific filter numbers, channel values, and geometric features that separate the three regimes.

---
# SERIALIZED WEIGHTS OF THE 9 MODELS

{serialized_sweep}
"""
    return prompt


def generate_read_prompt(serialized_comparison: str, mode: str = "focused") -> str:
    """Generate the full prompt for the READ experiment."""
    prompt = f"""You are an expert in neural network mechanistic interpretability and weight-space analysis.

# Context: EuroSAT Satellite Land Cover Classification
Two identical convolutional neural networks (SmallCNN) were trained from scratch on the EuroSAT_RGB dataset (10 classes: Annual Crop, Forest, Herbaceous Vegetation, Highway, Industrial Buildings, Pasture, Permanent Crop, Residential Buildings, River, SeaLake) at native 64x64 resolution.

Both models share the EXACT SAME:
- Architecture (SmallCNN, 98,954 parameters)
- Dataset train/val/test splits (16,200 train / 5,400 val / 5,400 test)
- Optimizer (AdamW, lr=1e-3, weight_decay=5e-4, 20 epochs, cosine schedule)
- Random seed

# The Hidden Difference: Data Augmentation Regularization
The ONLY difference between the two models during training was their data augmentation policy:
1. "no_augmentation": Trained strictly on raw unaugmented pixel patches. The model memorized native pixel coordinates, sharp horizontal/vertical field boundaries, and absolute RGB color intensities. Achieves ~94.6% test accuracy.
2. "strong_augmentation": Trained with heavy regularizing augmentations: random 90-degree rotations, random horizontal & vertical flips, aggressive color jitter, random grayscale (p=0.15), and Gaussian blur. Achieves ~77.0% test accuracy.

# Your Task
Below are the serialized weights for **MODEL A** and **MODEL B** (serialized at 3-decimal precision).
One is "no_augmentation" and the other is "strong_augmentation".

Analyze the weights and answer:
1. Which model is "no_augmentation" and which model is "strong_augmentation"?
2. What specific mechanistic evidence in the weight tensors supports your conclusion?
   - **Spatial Symmetry / Isotropy:** Do the 3x3 conv filters in conv1 show rotationally symmetric/isotropic patterns (consistent with 90° rotation augmentation) or sharp, axis-aligned, directional edge detectors (consistent with no augmentation)?
   - **Color Sensitivity:** Do the RGB channels show balanced/invariant weights (consistent with color jitter/grayscale) or distinct spectral-contrast signatures?
   - **Norms / Weight Distribution:** Are the filter norms or classifier weights different?

# Output Format Required
Please structure your answer as follows:
- **Prediction:**
  - Model A: [no_augmentation | strong_augmentation]
  - Model B: [no_augmentation | strong_augmentation]
- **Confidence:** [0% - 100%]
- **Key Mechanistic Evidence:**
  - [Detailed step-by-step reasoning referencing specific filter/channel indices]

---
# SERIALIZED WEIGHTS

{serialized_comparison}
"""
    return prompt


def generate_write_prompt() -> str:
    """Generate the prompt for synthesizing conv1 weights."""
    classes_str = ", ".join(f"{i}: {name}" for i, name in enumerate(wsp.CLASS_NAMES))
    prompt = f"""You are an expert deep learning researcher specializing in satellite computer vision and weight-space programming.

# Task: Synthesize Functional Weights for a Vision Model
You are tasked with handcrafting the raw floating-point weights for the first convolutional layer (`conv1.weight`) of a convolutional neural network designed for EuroSAT satellite land cover classification.

# Architecture & Target Specification
- **Target Tensor:** `conv1.weight`
- **Shape:** `[16, 3, 3, 3]` (16 filters, 3 input channels [R, G, B], 3x3 spatial kernel = 432 weights)
- **Input Channels:** Channel 0 = Red, Channel 1 = Green, Channel 2 = Blue (normalized with ImageNet mean/std).
- **Target Classes (10 categories):**
  {classes_str}
- **Statistical Guidelines (matching Kaiming Normal):**
  - Expected mean: ~0.0
  - Expected standard deviation: ~0.118 (fan_out = 16 * 9 = 144, std = sqrt(2 / 144) ≈ 0.118)
  - Values typically between -0.40 and +0.40.

# Satellite Remote Sensing Physical Priors
EuroSAT contains top-down aerial land cover scenes. To extract functional features zero-shot, your 16 filters should implement distinct, interpretable feature extractors:
1. **Vegetation Discriminators (Filters 0–3):** Contrast Green against Red and Blue (approximating visible vegetation indices like VARI: 2G - R - B) to detect Forest, Pasture, and Crops.
2. **Water Bodies & Moisture Detectors (Filters 4–6):** Capitalize on water absorbing Red light while reflecting Blue/Green.
3. **Infrastructure & Road Edge Detectors (Filters 7–10):** High-frequency spatial gradients (Sobel / Laplacian horizontal, vertical, and diagonal edge detectors) acting equally across RGB to detect Highways and Industrial Buildings.
4. **Agricultural Field Textures (Filters 11–13):** Directional Gabor-like periodic filters to detect agricultural rows and field boundaries.
5. **Urban High-Density / Residential Patterns (Filters 14–15):** Corner and high-contrast spot detectors.

# Output Format Required
You MUST return your response as a valid, parsable JSON object in a ```json code block containing the exact shape `[16, 3, 3, 3]`:

```json
{{
  "conv1_weight": [
    // Filter 0 (shape 3x3x3)
    [
      // Chan 0 (Red) 3x3
      [0.05, -0.12, 0.05],
      [-0.10, 0.25, -0.10],
      [0.05, -0.12, 0.05]
    ],
    [
      // Chan 1 (Green) 3x3
      ...
    ],
    [
      // Chan 2 (Blue) 3x3
      ...
    ]
    // ... Repeat for all 16 filters
  ],
  "design_rationales": {{
    "filter_0": "Description of what Filter 0 detects...",
    ...
  }}
}}
```
Ensure all 16 filters are fully defined (16 filters x 3 channels x 3 rows x 3 columns = 432 floats). Do not use ellipsis `...` in the JSON array.
"""
    return prompt


def generate_edit_prompt(variant: str = "A") -> str:
    """Generate prompt for repairing degraded ResNet-18 weights via Python script."""
    summary_path = WSP_ROOT / "experiment_2_edit" / "corruption_configs" / "corruption_summary.json"
    summary = wsp.read_json(summary_path)

    v_info = summary["corrupted"][variant.upper()]
    orig_info = summary["original"]

    prompt = f"""You are an expert in weight-space programming and neural network repair.

# Task: Surgical Weight Repair Without Backpropagation
A ResNet-18 model trained on EuroSAT (224x224 RGB, 10 classes) has achieved:
- Original Test Accuracy: {orig_info['test_accuracy'] * 100:.2f}%
- Original Val Accuracy:  {orig_info['val_accuracy'] * 100:.2f}%

# The Corruption
A single critical bottleneck layer, `layer3.0.conv1.weight` (shape `[256, 128, 3, 3] = 294,912` weights), was degraded using **{v_info['name']}**:
- Degraded Test Accuracy: {v_info['test_accuracy'] * 100:.2f}% (Drop: -{(orig_info['test_accuracy'] - v_info['test_accuracy'])*100:.2f}%)
- Number of modified elements: {v_info['modified_elements']:,} (out of 294,912)

# Your Goal
Following Takehi et al. (MIT CSAIL, Nov 2026, Experiments 6 & 8), write a Python repair function that takes the corrupted tensor and repairs it using algebraic or statistical techniques to restore as much of the original classification accuracy as possible without retraining.

Techniques you can use:
1. **Low-Rank SVD / PCA Filtering:** The weight tensor reshaped as [256, 128 * 9] = [256, 1152] has low intrinsic dimensionality; noise introduces spurious singular values.
2. **Outlier Detection & Denoising:** Corrupted weights exhibit anomalous deviations from the layer's expected Gaussian distribution.
3. **Channel-wise Statistics Matching:** Adjusting channel variances back to empirical layer norms.

# Output Format Required
Provide a self-contained Python function in a ```python code block:

```python
import torch

def repair_layer3_conv1(corrupted_tensor: torch.Tensor) -> torch.Tensor:
    \"\"\"
    Args:
        corrupted_tensor: torch.Tensor of shape [256, 128, 3, 3] (float32)
    Returns:
        repaired_tensor: torch.Tensor of shape [256, 128, 3, 3] (float32)
    \"\"\"
    # Your algorithmic repair logic here
    repaired = corrupted_tensor.clone()
    ...
    return repaired
```
Explain the mathematical rationale behind your repair strategy.
"""
    return prompt


def generate_all_prompts(seed: int = 42) -> None:
    PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 0. READ 9-Model Sweep (The comprehensive 9-CNN matrix)
    text_9, key_9 = create_blind_9_models_sweep(mode="focused", seed=seed)
    prompt_read_9 = generate_read_9_models_prompt(text_9)
    p_9 = PROMPTS_DIR / "prompt_read_exp1_all9_models.txt"
    p_9.write_text(prompt_read_9, encoding="utf-8")
    wsp.write_json(key_9, PROMPTS_DIR / "key_read_exp1_all9_models.json")
    print(f"Generated: {p_9} (~{len(prompt_read_9.split())} words, 9 models)")

    # 1. READ Prompts (Focused & Full A/B Pairs)
    ckpt_none = WSP_ROOT / "experiment_1_read" / "checkpoints" / "cnn_none_seed0.pt"
    ckpt_strong = WSP_ROOT / "experiment_1_read" / "checkpoints" / "cnn_strong_seed0.pt"

    # Focused
    text_foc, key_foc = create_blind_ab_comparison(
        ckpt_none, ckpt_strong,
        label_a="no_augmentation", label_b="strong_augmentation",
        mode="focused", seed=seed
    )
    prompt_read_foc = generate_read_prompt(text_foc, mode="focused")
    p_foc = PROMPTS_DIR / "prompt_read_exp1_focused.txt"
    p_foc.write_text(prompt_read_foc, encoding="utf-8")
    wsp.write_json(key_foc, PROMPTS_DIR / "key_read_exp1_focused.json")
    print(f"Generated: {p_foc} (~{len(prompt_read_foc.split())} words)")

    # Full (1M context)
    text_full, key_full = create_blind_ab_comparison(
        ckpt_none, ckpt_strong,
        label_a="no_augmentation", label_b="strong_augmentation",
        mode="full", seed=seed
    )
    prompt_read_full = generate_read_prompt(text_full, mode="full")
    p_full = PROMPTS_DIR / "prompt_read_exp1_full_1m.txt"
    p_full.write_text(prompt_read_full, encoding="utf-8")
    wsp.write_json(key_full, PROMPTS_DIR / "key_read_exp1_full_1m.json")
    print(f"Generated: {p_full} (~{len(prompt_read_full.split())} words)")

    # 2. WRITE Prompt
    prompt_write = generate_write_prompt()
    p_write = PROMPTS_DIR / "prompt_write_exp3_conv1.txt"
    p_write.write_text(prompt_write, encoding="utf-8")
    print(f"Generated: {p_write}")

    # 3. EDIT Prompt (Variant A)
    prompt_edit_a = generate_edit_prompt(variant="A")
    p_edit_a = PROMPTS_DIR / "prompt_edit_exp2_variant_a.txt"
    p_edit_a.write_text(prompt_edit_a, encoding="utf-8")
    print(f"Generated: {p_edit_a}")


def main() -> None:
    parser = argparse.ArgumentParser(description="WSP Prompt Generator")
    parser.add_argument("--generate-all", action="store_true",
                        help="Generate all prompt files in eurosat_wsp/prompts/")
    parser.add_argument("--seed", type=int, default=42, help="Seed for blind A/B shuffle")
    args = parser.parse_args()

    if args.generate_all:
        generate_all_prompts(seed=args.seed)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
