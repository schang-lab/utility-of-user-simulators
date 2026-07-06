import argparse
from pathlib import Path

from transformers import AutoTokenizer

ENDCONV_TOKEN = "<|endconversation|>"


def prepare_tokenizer(base_tokenizer_name: str, output_dir: str):
    """
    Load tokenizer, add special token, and save
    Args:
        base_tokenizer_name: Base tokenizer (e.g., "meta-llama/Meta-Llama-3-8B")
        output_dir: Directory to save modified tokenizer
    """
    print(f"Loading base tokenizer: {base_tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_tokenizer_name)
    print(f"Original vocab size: {len(tokenizer)}")

    if ENDCONV_TOKEN in tokenizer.get_vocab():
        print(f"Token '{ENDCONV_TOKEN}' already in vocabulary")
    else:
        print(f"Adding special token: {ENDCONV_TOKEN}")
        num_added = tokenizer.add_special_tokens(
            {'additional_special_tokens': [ENDCONV_TOKEN]}
        )
        print(f"Added {num_added} new token(s)")
    print(f"New vocab size: {len(tokenizer)}")

    output_dir += f"--{base_tokenizer_name.replace('/', '--')}"
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    tokenizer.save_pretrained(output_path)
    print(f"\n✓ Tokenizer saved to: {output_path}")

    print("\nVerifying saved tokenizer...")
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_path)
    assert ENDCONV_TOKEN in loaded_tokenizer.get_vocab(), "Token not found in loaded tokenizer!"
    print(f"Verification successful: {len(loaded_tokenizer)} tokens")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare tokenizer with custom tokens")
    parser.add_argument("--base_tokenizer", type=str,
                       default="meta-llama/Meta-Llama-3-8B",
                       help="Base tokenizer name or path")
    parser.add_argument("--output_dir", type=str,
                       default="./tokenizers_and_configs",
                       help="Output directory for modified tokenizer")
    args = parser.parse_args()

    prepare_tokenizer(args.base_tokenizer, args.output_dir)    