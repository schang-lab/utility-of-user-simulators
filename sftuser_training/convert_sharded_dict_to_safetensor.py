#!/usr/bin/env python
"""
Convert FSDP sharded checkpoints (.distcp) to HuggingFace format for vLLM inference.

Usage:
    python convert_checkpoint_to_hf.py \
        --fsdp_checkpoint_path ./userlm_checkpoints/epoch-1-meta-llama--Meta-Llama-3-8B \
        --output_path ./converted_model \
        --model_name meta-llama/Meta-Llama-3-8B \
        --dtype bfloat16

Note: By default, preserves bfloat16 precision from training (saves disk space and memory).
      Use --dtype float32 if you need full precision, but this will be much larger.
"""

import argparse
import sys
import torch
from pathlib import Path

# Add llama-cookbook to path
cookbook_path = Path(__file__).parent / "llama-cookbook" / "src"
sys.path.insert(0, str(cookbook_path))

from llama_cookbook.inference.model_utils import load_llama_from_config
from llama_cookbook.model_checkpointing.checkpoint_handler import load_sharded_model_single_gpu
from transformers import AutoTokenizer, AutoConfig


def convert_fsdp_to_hf(
    fsdp_checkpoint_path: str,
    output_path: str,
    model_name: str,
    tokenizer_path: str,
    dtype: str = "bfloat16",
):
    """
    Convert FSDP sharded checkpoint to HuggingFace format.

    Args:
        fsdp_checkpoint_path: Path to the directory containing __1_0.distcp, __2_0.distcp, etc.
        output_path: Where to save the converted HuggingFace model
        model_name: Original HuggingFace model name (e.g., 'meta-llama/Meta-Llama-3-8B')
        dtype: Model precision - 'bfloat16' (default), 'float16', or 'float32'
    """
    fsdp_path = Path(fsdp_checkpoint_path)
    output_path = Path(output_path)

    # Map dtype string to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }

    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}. Choose from {list(dtype_map.keys())}")

    torch_dtype = dtype_map[dtype]

    # Validate input
    if not fsdp_path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {fsdp_path}")

    # Check for .distcp files
    distcp_files = list(fsdp_path.glob("*.distcp"))
    if not distcp_files:
        raise FileNotFoundError(
            f"No .distcp files found in {fsdp_path}. "
            f"Make sure you're pointing to the checkpoint directory."
        )

    print(f"Found {len(distcp_files)} .distcp checkpoint files")
    print(f"Loading model definition from: {model_name}")
    print(f"Target dtype: {dtype}")

    # Load the base model architecture
    model = load_llama_from_config(model_name)

    # Convert model to target dtype BEFORE loading checkpoint
    # This ensures checkpoint weights are loaded in the correct precision
    model = model.to(torch_dtype)
    print(f"✓ Model architecture loaded in {dtype}")

    # Load the sharded checkpoint weights
    print(f"Loading sharded checkpoint from: {fsdp_path}")
    model = load_sharded_model_single_gpu(model, str(fsdp_path))
    print("✓ Checkpoint weights loaded")

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Load and save tokenizer
    print(f"Loading tokenizer from: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_path)
    print("✓ Tokenizer saved")

    # Load and save config
    config = AutoConfig.from_pretrained(model_name)
    config.save_pretrained(output_path)
    print("✓ Config saved")

    # Save model in HuggingFace format
    print(f"Saving model to: {output_path}")
    model.save_pretrained(output_path)
    print(f"✓ Model saved in HuggingFace format ({dtype})")

    print("\n" + "="*60)
    print("✓ Conversion complete!")
    print(f"Model saved to: {output_path}")
    print(f"Precision: {dtype}")
    print("\nYou can now load it with vLLM using:")
    print(f"  vllm serve {output_path} --dtype {dtype}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSDP sharded checkpoints to HuggingFace format"
    )
    parser.add_argument(
        "--fsdp_checkpoint_path",
        type=str,
        required=True,
        help="Path to FSDP checkpoint directory (containing .distcp files)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save converted HuggingFace model",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Original HuggingFace model name (e.g., 'meta-llama/Meta-Llama-3-8B')",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer directory",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"],
        help="Model precision (default: bfloat16, matches training precision)",
    )

    args = parser.parse_args()

    convert_fsdp_to_hf(
        fsdp_checkpoint_path=args.fsdp_checkpoint_path,
        output_path=args.output_path,
        model_name=args.model_name,
        tokenizer_path=args.tokenizer_path,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
