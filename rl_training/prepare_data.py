"""Convert the intent-annotated WildChat splits from sftuser_training into SkyRL parquet files.

RL uses the same user-level train/val/test partition as SFT (Appendix F.1), so test conversations are
held out from both stages. Each example keeps only what the environments need: the real opening user
turn (the assistant's first observation), the intent, and a persona seed for RPUSER3 (Appendix D.3).
The environment class is chosen at launch time, so one set of files serves every simulator.

Outputs in --output_dir:
    train.parquet, val.parquet, test.parquet
    sampled_val.parquet   deterministic subset of val used for validation during training (10K examples)
"""

import argparse
import json
import random
from pathlib import Path

import pandas as pd

DATA_SOURCE = "allenai--WildChat-1M"


def load_split(path: Path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def to_row(example, intent_model, persona_seed):
    conversation = example["conversation"]
    if not conversation or conversation[0]["role"] != "user":
        raise ValueError(f"Conversation {example['conversation_hash']} does not start with a user turn")
    if not example.get("intent"):
        raise ValueError(f"Conversation {example['conversation_hash']} has no intent")
    return {
        "data_source": DATA_SOURCE,
        "prompt": [{"role": "user", "content": conversation[0]["content"]}],
        "extra_info": {
            "conversation_hash": example["conversation_hash"],
            "intent": example["intent"],
            "intent_model": intent_model,
            "persona_seed": persona_seed,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_dir", type=Path, required=True,
                        help="sftuser_training/data_with_intents (output of 2_generate_intents.py)")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--intent_model", default="Qwen--Qwen3-32B",
                        help="Suffix of {split}_with_intents_{intent_model}.jsonl")
    parser.add_argument("--seed", type=int, default=42, help="Seed for persona seeds and the validation subset")
    parser.add_argument("--val_subset_size", type=int, default=10000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    # Splits are processed in this order so persona seeds are reproducible for a given --seed.
    for split in ("test", "val", "train"):
        examples = load_split(args.input_dir / f"{split}_with_intents_{args.intent_model}.jsonl")
        rows = [to_row(example, args.intent_model, rng.randint(0, 2**31 - 1)) for example in examples]
        frame = pd.DataFrame(rows)
        frame.to_parquet(args.output_dir / f"{split}.parquet", index=False)
        print(f"Saved {len(frame)} {split} examples")
        if split == "val":
            subset = frame.sample(n=min(args.val_subset_size, len(frame)), random_state=args.seed)
            subset.to_parquet(args.output_dir / "sampled_val.parquet", index=False)
            print(f"Saved {len(subset)} sampled_val examples")


if __name__ == "__main__":
    main()
