"""Parser Engine for EuroSAT Weight-Space Programming.

Parses, sanitizes, and validates Claude's generated responses for:
1. WRITE: Extracts 432-weight conv1 tensor from JSON, checks shape (16,3,3,3) & finiteness.
2. READ: Extracts Model A/B predictions, confidence level, and key mechanistic arguments.
3. EDIT: Extracts Python repair function from markdown code blocks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure eurosat_wsp root is in sys.path
WSP_ROOT = Path(__file__).resolve().parent
if str(WSP_ROOT) not in sys.path:
    sys.path.insert(0, str(WSP_ROOT))

import common as wsp


def extract_code_block(text: str, language: str = "json") -> str:
    """Extract code from ```<language> ... ``` code block, or return text if not found."""
    pattern = rf"```{language}\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Try generic code block ``` ... ```
    generic_match = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()
    return text.strip()


def parse_write_response(response_text: str, save_path: Path | None = None) -> torch.Tensor:
    """Parse conv1 weights from Claude's response.
    
    Expected format in response:
    {
      "conv1_weight": [[[[...], ...], ...], ...],  # shape (16, 3, 3, 3)
      "design_rationales": { ... }
    }
    """
    cleaned_json_str = extract_code_block(response_text, language="json")
    
    try:
        data = json.loads(cleaned_json_str)
    except json.JSONDecodeError as e:
        # Fallback: search for first '{' and last '}'
        start = cleaned_json_str.find("{")
        end = cleaned_json_str.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(cleaned_json_str[start:end + 1])
        else:
            raise ValueError(f"Failed to parse valid JSON from response: {e}")

    if "conv1_weight" in data:
        raw_weights = data["conv1_weight"]
    elif isinstance(data, list):
        raw_weights = data
    else:
        raise KeyError(f"Expected key 'conv1_weight' in parsed JSON, got keys: {list(data.keys())}")

    tensor = torch.tensor(raw_weights, dtype=torch.float32)

    # Validation
    expected_shape = (16, 3, 3, 3)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Invalid tensor shape: expected {expected_shape}, got {tuple(tensor.shape)}")

    if not torch.isfinite(tensor).all():
        raise ValueError("Parsed weights contain NaN or Inf values!")

    stats = wsp.tensor_stats(tensor)
    print("Successfully parsed conv1.weight!")
    print(f"Shape: {stats['shape']} | Numel: {stats['numel']}")
    print(f"Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}, min={stats['min']:.4f}, max={stats['max']:.4f}")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, save_path)
        print(f"Saved tensor to {save_path}")

    return tensor


def parse_read_response(response_text: str) -> dict:
    """Extract prediction and confidence from Claude's READ response (A/B or 9-model sweep)."""
    result = {
        "pred_model_a": None,
        "pred_model_b": None,
        "models_9": {},
        "confidence": None,
        "raw_response": response_text,
    }

    # 1. Check for 9-model sweep predictions: Model_01: [condition]
    pattern_9 = r"Model_?(\d{1,2})\s*:\s*\[?(no_augmentation|standard_augmentation|strong_augmentation)\]?"
    matches_9 = re.findall(pattern_9, response_text, re.IGNORECASE)
    if matches_9:
        for idx_str, cond in matches_9:
            key = f"Model_{int(idx_str):02d}"
            result["models_9"][key] = cond.lower()

    # 2. Check for A/B predictions
    match_a = re.search(r"Model\s*A\s*:\s*\[?(no_augmentation|standard_augmentation|strong_augmentation)\]?", response_text, re.IGNORECASE)
    if match_a:
        result["pred_model_a"] = match_a.group(1).lower()

    match_b = re.search(r"Model\s*B\s*:\s*\[?(no_augmentation|standard_augmentation|strong_augmentation)\]?", response_text, re.IGNORECASE)
    if match_b:
        result["pred_model_b"] = match_b.group(1).lower()

    # Find Confidence
    match_conf = re.search(r"Confidence\s*:\s*\[?(\d+(?:\.\d+)?)\s*%?\]?", response_text, re.IGNORECASE)
    if match_conf:
        result["confidence"] = float(match_conf.group(1))

    return result


def parse_edit_response(response_text: str, save_path: Path | None = None) -> str:
    """Extract Python repair code block from Claude's EDIT response."""
    code = extract_code_block(response_text, language="python")
    if "def repair_layer3_conv1" not in code:
        # Check if generic code block contains it
        match = re.search(r"(def repair_layer3_conv1.*?)(?:```|$)", response_text, re.DOTALL)
        if match:
            code = match.group(1).strip()
        else:
            raise ValueError("Could not find 'repair_layer3_conv1' function definition in response.")

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(code, encoding="utf-8")
        print(f"Saved repair script to {save_path}")

    return code


def main() -> None:
    parser = argparse.ArgumentParser(description="WSP Response Parser")
    parser.add_argument("--mode", choices=["write", "read", "edit"], required=True)
    parser.add_argument("--input", type=str, required=True, help="Path to Claude's response text file")
    parser.add_argument("--save", type=str, default=None, help="Where to save parsed artifact (.pt or .py)")
    args = parser.parse_args()

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8")

    if args.mode == "write":
        parse_write_response(text, save_path=Path(args.save) if args.save else None)
    elif args.mode == "read":
        parsed = parse_read_response(text)
        print("Parsed READ Response:")
        print(json.dumps({k: v for k, v in parsed.items() if k != "raw_response"}, indent=2))
    elif args.mode == "edit":
        code = parse_edit_response(text, save_path=Path(args.save) if args.save else None)
        print(f"Extracted repair code ({len(code.splitlines())} lines).")


if __name__ == "__main__":
    main()
