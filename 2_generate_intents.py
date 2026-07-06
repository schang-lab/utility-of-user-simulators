import argparse
import json
import os
import random
import time
import threading
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from tqdm import tqdm
from openai import OpenAI


class LeastLoadedBalancer:
    """Thread-safe least-connections load balancer for a pool of ports."""
    def __init__(self, ports: List[int]):
        self._ports = ports
        self._counts = [0] * len(ports)
        self._lock = threading.Lock()

    def acquire(self) -> Tuple[int, int]:
        """Returns (index, port) for the least-loaded port."""
        with self._lock:
            idx = min(range(len(self._counts)), key=lambda i: self._counts[i])
            self._counts[idx] += 1
        return idx, self._ports[idx]

    def release(self, idx: int) -> None:
        with self._lock:
            self._counts[idx] -= 1


def _worker_process_conversation(conv_data, api_key, model, few_shot_examples, ports, balancer):
    """
    Worker function for thread pool. Each call creates its own OpenAI client.
    Args:
        conv_data: Conversation dictionary
        api_key: OpenAI API key
        model: Model name to use
        few_shot_examples: Few-shot examples for prompting
        ports: List of inference engine ports for local models
        balancer: LeastLoadedBalancer instance (None for GPT models)
    Returns:
        Tuple of (conversation_dict, success_bool)
    """

    def _format_conversation(conversation: List[Dict]) -> str:
        """
        Format conversation for the prompt
        """
        lines = []
        for turn in conversation:
            role = turn.get('role', 'unknown')
            content = turn.get('content', '')
            # Truncate long assistant responses for brevity
            if role == 'assistant' and len(content) > 2048:
                content = content[:2048] + " ..."
            lines.append(f"<{role}>: {content}")
        return '\n'.join(lines)

    def _create_prompt(conversation: List[Dict]) -> str:
        """
        Create the full prompt for intent generation
        """
        prompt_parts = [
            "You are given the conversation history between a user and assistant model and your task is to create a summary of the user's intent from the conversation.",
            "",
            "Your summary should be structured to define what the high level intent of the user is, but should not go into specific details.",
            "",
            'Format the summary to start with "You are a user chatting with an assistant language model to"',
            ""
        ]

        for i, example in enumerate(few_shot_examples, 1):
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append("")
            prompt_parts.append("Conversation History:")
            prompt_parts.append(_format_conversation(example['conversation']))
            prompt_parts.append("")
            prompt_parts.append("Intent Summary:")
            prompt_parts.append(example['intent'])
            prompt_parts.append("")

        prompt_parts.append("Now generate a summary of the user intent for the following conversation:")
        prompt_parts.append("")
        prompt_parts.append(_format_conversation(conversation))
        prompt_parts.append("")
        prompt_parts.append("Reply with only the intent summary and nothing else.")

        return '\n'.join(prompt_parts)

    idx = None
    if 'gpt' in model:
        base_url = None
        effective_api_key = api_key or os.getenv('OPENAI_API_KEY')
    else:
        if balancer is not None:
            idx, port = balancer.acquire()
        else:
            port = random.choice(ports)
        base_url = f"http://localhost:{port}/v1"
        effective_api_key = None
    client = OpenAI(api_key=effective_api_key, base_url=base_url)
    try:
        turns = conv_data.get('conversation', [])
        if not turns:
            return (conv_data, False)

        prompt = _create_prompt(turns)
        max_retries = 4
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=16384,
                    temperature=0.7,
                    top_p=0.9,
                    extra_body={ # optional; can skip thinking for higher throughput
                        # "top_k": 20,
                        # "chat_template_kwargs": {"enable_thinking": False},
                    },
                )
                intent = response.choices[0].message.content.strip()
                if intent:
                    conv_data['intent'] = intent
                    return (conv_data, True)
            except Exception as e:
                if hasattr(e, 'status_code') and e.status_code == 400:
                    print(f"Non-retryable error: {e}")
                    return (conv_data, False)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt + random.uniform(0, 1))
                    print(f"Retrying (attempt {attempt + 2}/{max_retries}) after error: {e}")
                else:
                    return (conv_data, False)

        return (conv_data, False)
    finally:
        if balancer is not None and idx is not None:
            balancer.release(idx)


class IntentGenerator:
    def __init__(self, api_key=None, model='gpt-4.1-mini', ports=None):
        """
        Initialize intent generator with OpenAI API
        Args:
            api_key: OpenAI API key (if None, reads from OPENAI_API_KEY env var)
            model: Model to use for intent generation
            ports: List of inference engine ports for local models
        """
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.ports = ports
        self.model = model

        # Few-shot examples from the original UserLM paper (Appendix D, Figure 8)
        self.few_shot_examples = self._get_few_shot_examples()

    def _get_few_shot_examples(self):
        """Get the 3-shot examples from the paper"""
        return [
            {
                "conversation": [
                    {"role": "user", "content": "List to me 5 beautiful sights in Algeria"},
                    {"role": "assistant", "content": "Algeria is a country of striking natural beauty and rich cultural heritage. Here are five breathtaking sights you should know about: Tassili n'Ajjer National Park, Djemila, Timgad, Beni Hammad Fort, Basilique Notre Dame d'Afrique"}
                ],
                "intent": "You are a user chatting with an assistant language model to obtain some recommendations of places to see in Algeria."
            },
            {
                "conversation": [
                    {"role": "user", "content": "does allertine cause weight gain"},
                    {"role": "assistant", "content": "As an AI language model, I don't have personal experiences. But according to the available literature, Allertine is not known to cause weight gain. However, ..."},
                    {"role": "user", "content": "do antihistamines cause weight gain"},
                    {"role": "assistant", "content": "Most antihistamines do not typically cause weight gain as a direct side effect. However, some people find that they experience increased appetite or cravings for sugary, high-calorie ..."},
                    {"role": "user", "content": "how much weight can i lose by cutting sugar for 40 days"},
                    {"role": "assistant", "content": "The amount of weight you can lose by cutting sugar from your diet for 40 days will depend on a variety of factors, including your starting weight, age, gender, level of physical activity, ..."}
                ],
                "intent": "You are a user chatting with an assistant language model to get information about strategies for losing weight and the impact of certain drugs or medications on weight gain."
            },
            {
                "conversation": [
                    {"role": "user", "content": "Is einstein determinism challenged by quantum mechanics?"},
                    {"role": "assistant", "content": "Quantum mechanics challenges Einstein's determinism, which held that the universe is fully predictable given complete information. Key quantum principles like the Heisenberg uncertainty ..."},
                    {"role": "user", "content": "What did hawkins say?"},
                    {"role": "assistant", "content": "It seems you are referring to Stephen Hawking, the renowned physicist. Stephen Hawking supported the principles of quantum mechanics, believing that the universe is fundamentally ..."},
                    {"role": "user", "content": "So does it mean determinism is refuted?"},
                    {"role": "assistant", "content": "Determinism is not entirely refuted but is significantly challenged at the quantum level, where subatomic behavior follows probabilistic principles rather than predictable, classical laws ..."},
                    {"role": "user", "content": "Does amything of this have implications in a phylosofical way or is it not correct to use it in this sense?"},
                    {"role": "assistant", "content": "Quantum mechanics challenges classical determinism and raises important philosophical questions in metaphysics, free will, and the nature of observation. Its probabilistic nature ..."}
                ],
                "intent": "You are a user chatting with an assistant language model to understand how quantum mechanics challenges Eistein's determinism and get the perspective of different scientists on this."
            }
        ]

    def process_conversations(self, input_file: str, output_file: str,
                            max_conversations: int = None, num_processes: int = 16):
        """
        Process all conversations from a file and generate intents using multiprocessing
        Args:
            input_file: Path to input JSONL file with conversations
            output_file: Path to output JSONL file with intents added
            max_conversations: Maximum number of conversations to process (for testing)
            num_processes: Number of parallel processes (default: CPU count)
        """
        print(f"Processing conversations from: {input_file}")
        print(f"Output will be saved to: {output_file}")

        conversations = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                conversations.append(json.loads(line))
                if max_conversations and len(conversations) >= max_conversations:
                    break
        print(f"Loaded {len(conversations)} conversations")

        # Skip conversations that already have intents generated
        Path(output_file).parent.mkdir(exist_ok=True, parents=True)
        already_done_hashes = set()
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        conv = json.loads(line)
                        ch = conv.get('conversation_hash')
                        if ch is not None:
                            already_done_hashes.add(ch)
                    except json.JSONDecodeError:
                        continue
            if already_done_hashes:
                original_count = len(conversations)
                conversations = [c for c in conversations if c.get('conversation_hash') not in already_done_hashes]
                print(f"Skipping {original_count - len(conversations)} already-processed conversations "
                      f"({len(already_done_hashes)} found in output file)")

        if not conversations:
            print("All conversations already processed, nothing to do.")
            return []

        print(f"Processing {len(conversations)} remaining conversations")
        print(f"Using {num_processes} threads")

        balancer = LeastLoadedBalancer(self.ports) if self.ports else None

        worker_fn = partial(
            _worker_process_conversation,
            api_key=self.client.api_key,
            model=self.model,
            few_shot_examples=self.few_shot_examples,
            ports=self.ports,
            balancer=balancer,
        )
        processed = []
        failed = 0
        flush_every = 1000
        with ThreadPoolExecutor(max_workers=num_processes) as pool, \
             open(output_file, 'a', encoding='utf-8') as f:
            futures = [pool.submit(worker_fn, conv) for conv in conversations]
            for future in tqdm(as_completed(futures), total=len(conversations), desc="Generating intents"):
                conv_data, success = future.result()
                if success:
                    processed.append(conv_data)
                    f.write(json.dumps(conv_data, ensure_ascii=False) + '\n')
                    if len(processed) % flush_every == 0:
                        f.flush()
                else:
                    failed += 1

        print(f"\nProcessing complete!")
        print(f"  Successfully processed: {len(processed)}")
        print(f"  Failed: {failed}")
        print(f"  Output saved to: {output_file}")
        return processed

    def process_all_splits(self, data_dir: str, output_dir: str, selective_splits: list, num_processes: int = 16):
        """Process train, val, and test splits"""
        data_dir = Path(data_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        save_model_name = self.model.replace('/', '--')

        for split in selective_splits:
            input_file = data_dir / f"{split}.jsonl"
            output_file = output_dir / f"{split}_with_intents_{save_model_name}.jsonl"

            if input_file.exists():
                print(f"\n{'='*80}")
                print(f"Processing {split.upper()} split")
                print(f"{'='*80}")
                self.process_conversations(str(input_file), str(output_file), num_processes=num_processes)
            else:
                print(f"Warning: {input_file} not found, skipping...")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate intents for WildChat conversations')
    parser.add_argument('--data_dir',
                        type=str, default='./processed_data',
                        help='Directory with preprocessed conversation files')
    parser.add_argument('--output_dir',
                        type=str, default='./data_with_intents',
                        help='Output directory for conversations with intents')
    parser.add_argument('--api_key',
                        type=str, default=None,
                        help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--model',
                        type=str, default='gpt-4.1-mini',
                        help='Model to use for intent generation')
    parser.add_argument('--test_mode',
                        action='store_true',
                        help='Test mode: only process 10 conversations')
    parser.add_argument('--num_processes',
                        type=int, default=16,
                        help='Number of parallel processes to use')
    parser.add_argument('--selective_split',
                        type=str, default=None,
                        help='Only process a specific split (train, val, or test)')
    parser.add_argument('--ports',
                        type=int, nargs='+', default=None,
                        help='Inference engine ports for local models (e.g., --ports 8002 8003)')
    args = parser.parse_args()

    generator = IntentGenerator(api_key=args.api_key, model=args.model, ports=args.ports)

    if args.test_mode:
        print("TEST MODE: Processing only 10 conversations")
        input_file = Path(args.data_dir) / "train.jsonl"
        output_file = Path(args.output_dir) / "train_with_intents--test.jsonl"
        generator.process_conversations(str(input_file), str(output_file), max_conversations=10)
    else:
        selective_splits = (
            [args.selective_split] if args.selective_split
            else ['train', 'val', 'test']
        )
        generator.process_all_splits(
            args.data_dir, args.output_dir,
            selective_splits=selective_splits,
            num_processes=args.num_processes,
        )