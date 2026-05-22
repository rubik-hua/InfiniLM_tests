#!/usr/bin/env python3
"""
Batch inference runner for InfiniLM.
Iterates over models and runs test_infer.py for each one.
"""

import argparse
import os
import subprocess
import sys

# ============================================================
# Default model names (only folder names, will be joined with --model-dir)
# ============================================================

DEFAULT_MODEL_NAMES = [
    "Baichuan2-7B-Chat",
    "chatglm3-6b",
    "DeepSeek-R1-Distill-Qwen-7B",
    "GLM-4-9B-0414",
    "internlm3-8b-instruct",
    "Meta-Llama-3.1-8B-Instruct",
    "Meta-Llama-3-8B-Instruct",
    "MiniCPM-SALA",
    "MiniCPM-V-2.6",
    "Mistral-7B-Instruct-v0.1",
    "Mistral-7B-Instruct-v0.2",
    "Qwen2.5-0.5B-Instruct",
]

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch inference runner for InfiniLM"
    )

    # Model settings
    parser.add_argument("--model-dir", type=str, default="/data/rubik/models/",
                        help="Root directory of model folders "
                             "(default: /data/rubik/models/)")
    parser.add_argument("--models", type=str, nargs="*", default=None,
                        help="Override model list with specific folder names "
                             "(e.g. --models Baichuan2-7B-Chat Meta-Llama-3-8B-Instruct). "
                             "If omitted, DEFAULT_MODEL_NAMES is used.")

    # Inference settings
    parser.add_argument("--device", type=str, default="nvidia",
                        help="Device type (e.g. nvidia, moore)")
    parser.add_argument("--tp", type=int, default=2,
                        help="Tensor parallelism degree")
    parser.add_argument("--batch-size", type=int, default=3,
                        help="Batch size for inference")
    parser.add_argument("--prompt", type=str, default="山东最高的山是？",
                        help="Prompt text for inference")
    parser.add_argument("--enable-paged-attn", action="store_true",
                        help="Enable paged attention")
    parser.add_argument("--script", type=str, default="examples/test_infer.py",
                        help="Path to the inference test script")

    args = parser.parse_args()

    # Build model name list
    model_names = args.models if args.models else DEFAULT_MODEL_NAMES

    # Build full model paths from model_dir + model_name
    model_dir = args.model_dir.rstrip("/")
    model_paths = [os.path.join(model_dir, name) for name in model_names]

    # Print configuration
    print(f"Configuration:")
    print(f"  Model dir:       {model_dir}")
    print(f"  Device:          {args.device}")
    print(f"  TP:              {args.tp}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Prompt:          {args.prompt}")
    print(f"  Paged Attention: {args.enable_paged_attn}")
    print(f"  Script:          {args.script}")
    print(f"  Models ({len(model_paths)}):")
    for p in model_paths:
        print(f"    - {p}")
    print()

    # Run inference for each model
    total = len(model_paths)

    for idx, model_path in enumerate(model_paths, 1):
        name = os.path.basename(model_path.rstrip("/"))
        print("=" * 60)
        print(f"  [{idx}/{total}] Inference model: {name}")
        print("=" * 60)

        cmd = [
            sys.executable, args.script,
            "--device", args.device,
            "--model", model_path,
            "--tp", str(args.tp),
            "--batch-size", str(args.batch_size),
            "--prompt", args.prompt,
        ]

        if args.enable_paged_attn:
            cmd.append("--enable-paged-attn")

        print(f"  Cmd: {' '.join(cmd)}")
        print()

        subprocess.run(cmd)


if __name__ == "__main__":
    main()

