import argparse
import json
import time
import random
from datetime import datetime
from collections import Counter
from pathlib import Path
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

MIN_CHUNK_SIZE = 1_000
NGRAM_THRESHOLD = 100

def _parse_jsonl_lines(lines):
    """
    Helper function for multiprocessing - parses and filters JSONL lines
    """
    english_conversations = []
    for line in lines:
        try:
            data = json.loads(line)
            if data.get('language') == 'English':
                english_conversations.append(data)
        except json.JSONDecodeError:
            continue
    return english_conversations


class WildChatPreprocessor:
    def __init__(self,
                 data_path, output_dir, num_processes,
                 train_ratio, val_ratio, test_ratio,
                 include_wildbench=True):
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.num_processes = num_processes
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.include_wildbench = include_wildbench
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    def load_wildbench_ids(self):
        """
        Load WildBench and return the set of conversation IDs (session_ids) it contains.
        """
        print("Loading WildBench dataset to extract conversation IDs...")
        dataset = load_dataset("allenai/WildBench", name="v2", split="test")
        ids = set(dataset["session_id"])
        print(f"  Found {len(ids)} unique WildBench session IDs")
        return ids

    def load_wildchat(self):
        """
        Load WildChat dataset with multiprocessing for English filtering
        """
        print("Loading WildChat dataset...")
        print(f"Using {self.num_processes} processes for parallel filtering")

        conversations = []

        # Option 1: Load from JSONL file
        if self.data_path.endswith('.jsonl'):
            print("Loading from JSONL file with multiprocessing...")
            with open(self.data_path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
            print(f"Found {total_lines} lines in file")
            print(f"Processing with {self.num_processes} processes...")

            chunk_size = min(5000, max(MIN_CHUNK_SIZE, total_lines // (self.num_processes * 20)))

            def read_chunks():
                chunk = []
                with open(self.data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        chunk.append(line)
                        if len(chunk) >= chunk_size:
                            yield chunk
                            chunk = []
                    if chunk: # last chunk
                        yield chunk

            _time_start = time.time()
            with Pool(processes=self.num_processes, maxtasksperchild=10) as pool:
                results = []
                num_chunks = (total_lines + chunk_size - 1) // chunk_size
                with tqdm(total=num_chunks, desc="Processing chunks", unit="chunk") as pbar:
                    for result in pool.imap_unordered(_parse_jsonl_lines, read_chunks(), chunksize=1):
                        results.append(result)
                        pbar.update(1)
                        pbar.set_postfix({"English convs": sum(len(r) for r in results)})
            _time_elapsed = time.time() - _time_start
            print(f"Processing completed in {_time_elapsed:.2f} seconds")

            conversations = [conv for chunk_result in results for conv in chunk_result]

        else: # Option 2: Load from HuggingFace datasets

            print("Loading from HuggingFace...")
            dataset = load_dataset("allenai/WildChat-1M", split="train")
            total_convs = len(dataset)
            print(f"Dataset loaded with {total_convs} conversations")
            print(f"Filtering English conversations with {self.num_processes} processes...")

            def is_english(example):
                return example.get('language').lower() == 'english'
            
            _time_start = time.time()
            english_dataset = dataset.filter(
                is_english,
                num_proc=self.num_processes,
                desc="Filtering English conversations"
            )
            _time_elapsed = time.time() - _time_start
            print(f"Filtering completed in {_time_elapsed:.2f} seconds")
            print(f"Filtered to {len(english_dataset)} English conversations")

            _time_start = time.time()
            _batch_size = 10_000
            conversations = []
            for batch in tqdm(
                english_dataset.iter(batch_size=_batch_size),
                total=(len(english_dataset) + _batch_size - 1) // _batch_size,
                desc="Converting to list"
            ):
                # batch is a dict of lists; transpose to list of dicts
                keys = list(batch.keys())
                n = len(batch[keys[0]])
                conversations.extend([{k: batch[k][i] for k in keys} for i in range(n)])
            _time_elapsed = time.time() - _time_start
            print(f"Converted to list in {_time_elapsed:.2f} seconds")

        print(f"Loaded {len(conversations)} English conversations")
        return conversations

    def extract_ngrams(self, text, n=7):
        words = text.lower().split()
        if len(words) < n:
            return []
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

    def deduplicate_conversations(self, conversations):
        """
        Deduplicate using 7-gram counting on first-turn user prompts
        Based on paper Appendix A
        """
        print("now in deduplicate_conversations()")

        first_turns = []
        for conv in tqdm(conversations, desc="Extracting first user turns"):
            if 'conversation' in conv and len(conv['conversation']) > 0:
                for turn in conv['conversation']:
                    if turn.get('role') == 'user':
                        first_turns.append(turn.get('content', ''))
                        break

        ngram_counter = Counter()
        for text in tqdm(first_turns, desc="Extracting 7-grams"):
            ngrams = self.extract_ngrams(text, n=7)
            ngram_counter.update(ngrams)

        # Remove conversations with highly repetitive first turns
        filtered_conversations, out_conversations = [], []
        for i, conv in enumerate(tqdm(conversations, desc="Filtering duplicates")):            
            first_turn = first_turns[i]
            ngrams = self.extract_ngrams(first_turn, n=7)
            if not any(ngram_counter[ng] > NGRAM_THRESHOLD for ng in ngrams):
                filtered_conversations.append(conv)
            else:
                out_conversations.append(conv)

        print(f"\nRemoved {len(out_conversations)} near-duplicate conversations")
        print(f"Remaining: {len(filtered_conversations)} conversations")

        return filtered_conversations, out_conversations

    def create_splits(self, conversations, wildbench_ids=None):
        """
        Split conversations by users (89/5/6) ensuring same user in same split.
        Based on hashed IP addresses and countries.
        If wildbench_ids is provided, all conversations whose conversation_id
        appears in WildBench are forced into the test split before the random split.
        """
        print("now in create_splits()")

        # Separate WildBench conversations first
        wildbench_test_convs = []
        remaining_convs = conversations
        if wildbench_ids:
            remaining_convs, wildbench_test_convs = [], []
            for conv in conversations:
                if conv.get('conversation_id') in wildbench_ids:
                    wildbench_test_convs.append(conv)
                else:
                    remaining_convs.append(conv)
            print(f"  WildBench conversations forced into test: {len(wildbench_test_convs)}")
            print(f"  Remaining conversations for random split: {len(remaining_convs)}")

        user_conversations = {}
        for conv in remaining_convs:
            # Create user identifier
            user_id = f"{conv.get('hashed_ip', 'unknown')}_{conv.get('country', 'unknown')}"
            if user_id not in user_conversations:
                user_conversations[user_id] = []
            user_conversations[user_id].append(conv)

        print(f"Total unique users: {len(user_conversations)}")

        # Group users by country for stratified split
        country_users = {}
        for user_id, convs in user_conversations.items():
            country = convs[0].get('country', 'unknown')
            if country not in country_users:
                country_users[country] = []
            country_users[country].append(user_id)

        train_convs, val_convs, test_convs = [], [], []
        for country, users in country_users.items():
            random.seed(42); random.shuffle(users)
            n_users = len(users)
            n_train = int(n_users * self.train_ratio)
            n_val = int(n_users * self.val_ratio)
            train_users = users[:n_train]
            val_users = users[n_train:n_train + n_val]
            test_users = users[n_train + n_val:]
            for user_id in train_users:
                train_convs.extend(user_conversations[user_id])
            for user_id in val_users:
                val_convs.extend(user_conversations[user_id])
            for user_id in test_users:
                test_convs.extend(user_conversations[user_id])

        test_convs.extend(wildbench_test_convs)

        print(f"\nSplit statistics:")
        print(f"  Train: {len(train_convs)} conversations")
        print(f"  Val: {len(val_convs)} conversations")
        print(f"  Test: {len(test_convs)} conversations (includes {len(wildbench_test_convs)} WildBench)")

        return {
            'train': train_convs,
            'val': val_convs,
            'test': test_convs
        }

    def save_splits(self, splits):
        print("now at save_splits()")

        def datetime_handler(obj):
            """Convert datetime objects to ISO format strings for JSON serialization"""
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        for split_name, conversations in splits.items():
            output_file = self.output_dir / f"{split_name}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for conv in conversations:
                    f.write(json.dumps(conv, ensure_ascii=False, default=datetime_handler) + '\n')
            print(f"  Saved {split_name}: {output_file}")

        # Save statistics
        stats = {
            'total_conversations': sum(len(convs) for convs in splits.values()),
            'train': len(splits['train']),
            'val': len(splits['val']),
            'test': len(splits['test'])
        }

        with open(self.output_dir / 'stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        return stats

    def run_preprocessing(self):

        conversations = self.load_wildchat()
        conversations, _ = self.deduplicate_conversations(conversations)

        wildbench_ids = self.load_wildbench_ids() if self.include_wildbench else None
        splits = self.create_splits(conversations, wildbench_ids=wildbench_ids)
        stats = self.save_splits(splits)
        print("\n" + "="*80)
        print("Preprocessing complete!")
        print(f"Result: {stats['total_conversations']} total conversations")
        print("="*80)

        return splits


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocess WildChat dataset')
    parser.add_argument('--data_path',
                        type=str, default='allenai/WildChat-1M',
                        help='Path to WildChat data or HuggingFace dataset name')
    parser.add_argument('--output_dir',
                        type=str, default='./processed_data',
                        help='Output directory for processed data')
    parser.add_argument('--num_processes',
                        type=int, default=8,
                        help='Number of processes for parallel filtering (default: 8)')
    parser.add_argument('--train_ratio',
                        type=float, default=0.89,
                        help='Ratio of training data')
    parser.add_argument('--val_ratio',
                        type=float, default=0.05,
                        help='Ratio of validation data')
    parser.add_argument('--test_ratio',
                        type=float, default=0.06,
                        help='Ratio of test data')
    parser.add_argument('--no_wildbench', action='store_true',
                        help='Skip forcing WildBench conversations into the test split')
    args = parser.parse_args()

    preprocessor = WildChatPreprocessor(
        args.data_path, args.output_dir, args.num_processes,
        args.train_ratio, args.val_ratio, args.test_ratio,
        include_wildbench=not args.no_wildbench,
    )
    preprocessor.run_preprocessing()
