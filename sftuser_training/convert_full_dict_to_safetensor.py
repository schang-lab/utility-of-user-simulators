"""
Convert a .pt checkpoint (with DTensors, world_size=1) to HuggingFace safetensors format.

Usage:
    python convert_full_dict_to_safetensor.py \
        --pt-path /path/to/model.pt \
        --model-name meta-llama/Llama-3.1-8B-Instruct \
        --output-dir /path/to/output
"""

import argparse
import os
import shutil
import torch
import torch.distributed._tensor as dt
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def dtensor_to_tensor(state_dict: dict) -> dict:
    """Convert DTensors to regular Tensors."""
    converted = {}
    for key, value in state_dict.items():
        if isinstance(value, dt.DTensor):
            converted[key] = value.full_tensor()
        elif isinstance(value, torch.Tensor):
            converted[key] = value
        else:
            print(f"Skipping non-tensor key: {key} (type={type(value)})")
    return converted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt-path", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--model-name", required=True, help="HuggingFace model name (e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--tokenizer-path", default=None, help="Path to tokenizer (defaults to --model-name if not set)")
    parser.add_argument("--output-dir", required=True, help="Output directory for safetensors model")
    args = parser.parse_args()
    if args.tokenizer_path is None:
        args.tokenizer_path = args.model_name

    # Load the .pt checkpoint
    print(f"Loading .pt checkpoint from {args.pt_path} ...")
    state_dict = torch.load(args.pt_path, map_location="cpu", weights_only=False)

    # Convert DTensors -> Tensors
    print("Converting DTensors to regular Tensors ...")
    state_dict = dtensor_to_tensor(state_dict)

    # Load the original HF model to get the architecture + config
    print(f"Loading base model architecture from {args.model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Resize embeddings if a special token was added during training
    ckpt_vocab_size = state_dict.get("model.embed_tokens.weight", state_dict.get("lm_head.weight")).shape[0]
    if ckpt_vocab_size != model.config.vocab_size:
        print(f"Resizing token embeddings: {model.config.vocab_size} -> {ckpt_vocab_size}")
        model.resize_token_embeddings(ckpt_vocab_size)

    # Load converted weights into the model
    print("Loading converted weights into model ...")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Warning: missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    # Save as safetensors
    print(f"Saving to {args.output_dir} ...")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    # save_pretrained() with a fast tokenizer produces a different tokenizer_config.json
    # (uses extra_special_tokens, omits chat_template, added_tokens_decoder, etc.).
    # Copy the original tokenizer_config.json to preserve the correct format.
    if os.path.isdir(args.tokenizer_path):
        src = os.path.join(args.tokenizer_path, "tokenizer_config.json")
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(args.output_dir, "tokenizer_config.json"))
            print("Restored original tokenizer_config.json from local tokenizer dir.")
    else:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download(repo_id=args.tokenizer_path, filename="tokenizer_config.json")
        shutil.copy2(src, os.path.join(args.output_dir, "tokenizer_config.json"))
        print("Restored original tokenizer_config.json from HuggingFace hub.")

    print("Done.")


if __name__ == "__main__":
    main()
