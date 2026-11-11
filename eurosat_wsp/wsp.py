"""Main CLI entrypoint for EuroSAT Weight-Space Programming (WSP).

Unified interface to:
1. prepare: Generate all formatted prompts and blind A/B keys
2. parse: Parse LLM responses for READ, WRITE, or EDIT
3. eval-read: Evaluate LLM's introspection against blind keys
4. eval-write: Benchmark synthesized weights (zero-shot and 20-epoch training)
5. eval-edit: Benchmark surgical repair function on ResNet-18
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure eurosat_wsp root is in sys.path
WSP_ROOT = Path(__file__).resolve().parent
if str(WSP_ROOT) not in sys.path:
    sys.path.insert(0, str(WSP_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import wsp_eval
import wsp_parser
import wsp_prompts
import wsp_serializer


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="wsp",
        description="Unified CLI for EuroSAT Weight-Space Programming Harness"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. PREPARE
    p_prep = subparsers.add_parser("prepare", help="Generate all prompt files & keys")
    p_prep.add_argument("--seed", type=int, default=42, help="Seed for blind A/B shuffle")

    # 2. PARSE
    p_parse = subparsers.add_parser("parse", help="Parse Claude's response text into tensors/code")
    p_parse.add_argument("--mode", choices=["write", "read", "edit"], required=True)
    p_parse.add_argument("--input", type=str, required=True, help="Input response text file")
    p_parse.add_argument("--save", type=str, default=None, help="Output destination (.pt or .py)")

    # 3. EVAL-READ
    p_eval_r = subparsers.add_parser("eval-read", help="Evaluate READ response against key")
    p_eval_r.add_argument("--response", type=str, required=True, help="Claude's response text file")
    p_eval_r.add_argument("--key", type=str, default="eurosat_wsp/prompts/key_read_exp1_focused.json",
                          help="Path to blind key JSON")

    # 4. EVAL-WRITE
    p_eval_w = subparsers.add_parser("eval-write", help="Evaluate synthesized weights")
    p_eval_w.add_argument("--weights", type=str, required=True, help="Path to .pt conv1 tensor")
    p_eval_w.add_argument("--seed", type=int, default=0, help="Seed for other layers")
    p_eval_w.add_argument("--train", action="store_true", help="Run 20-epoch training schedule")
    p_eval_w.add_argument("--epochs", type=int, default=20, help="Number of training epochs")

    # 5. EVAL-EDIT
    p_eval_e = subparsers.add_parser("eval-edit", help="Evaluate surgical repair script")
    p_eval_e.add_argument("--repair-script", type=str, required=True, help="Path to repair .py script")
    p_eval_e.add_argument("--variant", choices=["A", "B", "C"], default="A", help="Corruption variant")

    args = parser.parse_args()

    if args.command == "prepare":
        wsp_prompts.generate_all_prompts(seed=args.seed)
    elif args.command == "parse":
        input_p = Path(args.input)
        text = input_p.read_text(encoding="utf-8")
        if args.mode == "write":
            wsp_parser.parse_write_response(text, save_path=Path(args.save) if args.save else None)
        elif args.mode == "read":
            res = wsp_parser.parse_read_response(text)
            print(res)
        elif args.mode == "edit":
            wsp_parser.parse_edit_response(text, save_path=Path(args.save) if args.save else None)
    elif args.command == "eval-read":
        wsp_eval.evaluate_read(Path(args.response), Path(args.key))
    elif args.command == "eval-write":
        if args.train:
            wsp_eval.evaluate_write_train(Path(args.weights), seed=args.seed, epochs=args.epochs)
        else:
            wsp_eval.evaluate_write_zero_shot(Path(args.weights), seed=args.seed)
    elif args.command == "eval-edit":
        wsp_eval.evaluate_edit(Path(args.repair_script), variant=args.variant)


if __name__ == "__main__":
    main()
