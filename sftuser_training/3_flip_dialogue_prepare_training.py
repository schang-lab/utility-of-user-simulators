import argparse
import json
from pathlib import Path
from typing import List, Dict, Union, Literal, Tuple
from collections.abc import Iterator
from multiprocessing import Pool, cpu_count

from tqdm import tqdm
from transformers import AutoTokenizer

from tokenizers_and_configs.tokenizer_configs import CHAT_TMPL


Role = Literal["user", "assistant"]
ENDCONV = "<|endconversation|>"


def expected_roles(start: Role = "user") -> Iterator[Role]:
    role: Role = start
    while True:
        yield role
        role = "assistant" if role == "user" else "user"


class DialogueFlipper:
    def __init__(self, tokenizer_name: str):
        """
        Initialize dialogue flipper
        """
        self._model_name = tokenizer_name

        assert tokenizer_name in CHAT_TMPL, "Unsupported tokenizer for chat template"
        self._chat_tmpl = CHAT_TMPL.get(tokenizer_name)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if "<|endconversation|>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {'additional_special_tokens': [ENDCONV]}
            )

    def flip_single_conversation(self, conv_data: Dict) -> Union[Dict, int]:
        """
        Create a single training sample with gradient flow masking.
        Strategy:
        1. Build the full conversation incrementally message by message
        2. For each message, determine what tokens are new
        3. Mark user message content tokens as trainable (actual token IDs in labels)
        4. Mark everything else as -100 (ignored in loss)
        Returns:
            Dict with 'input_ids' and 'labels', where labels=-100 for tokens
            that should not contribute to loss (assistant turns, system prompt).
            Only user turn tokens have actual token IDs in labels.
        """
        conv: List[Dict] = conv_data.get('conversation', [])
        intent: str = conv_data.get('intent', '').strip()
        
        # validation: return -1 if missing content or intent, -2 if role mismatch
        if not conv or not intent:
            return -1
        role_iterator = expected_roles()
        for uttr in conv:
            role, expected_role = uttr.get('role', ''), next(role_iterator)
            if role != expected_role:
                return -2
            content: str = uttr.get('content', '').strip()
            if not content:
                return -1
            
        def _generation_prompter(_role, messages_until_now) -> str:
            if self._model_name in ['meta-llama/Meta-Llama-3-8B']:
                return "<|start_header_id|>user<|end_header_id|>\n\n"
            elif self._model_name in ['meta-llama/Llama-3.2-3B']:
                raise ValueError("Problem with the default system prompt")
            elif self._model_name in [
                'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-7B',
                'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-32B',
                'Qwen/Qwen2.5-1.5B-Instruct', 'Qwen/Qwen2.5-3B-Instruct', 'Qwen/Qwen2.5-7B-Instruct',
                'Qwen/Qwen2.5-14B-Instruct', 'Qwen/Qwen2.5-32B-Instruct',
            ]:
                return "<|im_start|>user\n"
            elif self._model_name in ['google/gemma-3-27b-pt']:
                if _role == 'system':
                    assert messages_until_now.__len__() == 1
                    return "<start_of_turn>user\n" + messages_until_now[0]['content'] + "\n\n"
                else:
                    return "<start_of_turn>user\n"
            raise NotImplementedError("Generation prompter not defined for "
                                      f"{self._model_name}")

        messages : List[Dict] = [{'role': 'system', 'content': intent}]
        conv.append({'role': 'user', 'content': ENDCONV})
        for _idx, uttr in enumerate(conv):
            # must add .strip() to prevent tokenization issue of two "\n"s vs. "\n\n"
            conv[_idx] = {'role': uttr['role'], 'content': uttr['content'].strip()}
        messages.extend(conv)
        input_ids, labels = [], []

        for i, message in enumerate(messages):
            role = message['role']
            current_messages = messages[:i+1]
            formatted = self.tokenizer.apply_chat_template(
                current_messages,
                tokenize=False,
                add_generation_prompt=False,
                chat_template=self._chat_tmpl,
            ) + (_generation_prompter(role, current_messages) if role != 'user' else "")

            current_tokens = self.tokenizer.encode(formatted, add_special_tokens=False)
            new_tokens = current_tokens[len(input_ids):]
            if role == 'user':
                labels.extend(new_tokens)
            else:
                labels.extend([-100] * len(new_tokens))
            input_ids.extend(new_tokens)
            
        # labels = labels[1:] + [-100] # labels shift-by-one to match prediction
        # UPDATE: No need to shift labels since we are using causal LM loss directly
        sample = {'input_ids': input_ids, 'labels': labels}
        
        ## sanity checks
        # at this point, current_tokens = tokenization of full conversation
        assert input_ids == current_tokens, "Input IDs do not match tokenized conversation"

        sample['input_ids'].append(self.tokenizer.eos_token_id)
        sample['labels'].append(-100)
        return sample
        
    def process_samples(self, conversations: List[Dict]) -> Tuple[List[Dict], int, int]:
        """
        Process a list of conversations with intents
        """
        preprocessed = []
        skipped_missing_cont = 0
        skipped_role_mismatch = 0

        for conv in tqdm(conversations, desc="Creating masked training samples"):
            sample = self.flip_single_conversation(conv)
            if isinstance(sample, dict):
                preprocessed.append(sample)
            else:
                if sample == -1:
                    skipped_missing_cont += 1
                elif sample == -2:
                    skipped_role_mismatch += 1

        return preprocessed, skipped_missing_cont, skipped_role_mismatch


def _mp_worker(args: Tuple[List[Dict], str]) -> Tuple[List[Dict], int, int]:
    """
    Worker function for multiprocessing (must be at module level for pickling)
    """
    conv_chunk, tokenizer_model = args
    return DialogueFlipper(tokenizer_model).process_samples(conv_chunk)


def process_file(input_file: str, output_file: str, tokenizer_model: str, perform_mp: bool = False):

    print(f"Processing: {input_file}")
    conversations = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            conversations.append(json.loads(line))
    print(f"Loaded {len(conversations)} conversations")

    if not perform_mp:
        flipper = DialogueFlipper(tokenizer_model)
        preprocessed, skip_miss_content, skip_role_mismatch = flipper.process_samples(conversations)
    else:

        num_workers = max(1, cpu_count() // 4)
        chunk_size = max(1, len(conversations) // num_workers)
        chunks = [conversations[i:i + chunk_size] for i in range(0, len(conversations), chunk_size)]
        worker_args = [(chunk, tokenizer_model) for chunk in chunks]
        print(f"Using {num_workers} workers to process {len(chunks)} chunks")

        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(_mp_worker, worker_args),
                total=len(worker_args),
                desc="Processing chunks"
            ))

        preprocessed = []
        skip_miss_content = 0
        skip_role_mismatch = 0
        for chunk_preprocessed, chunk_skip_miss, chunk_skip_role in results:
            preprocessed.extend(chunk_preprocessed)
            skip_miss_content += chunk_skip_miss
            skip_role_mismatch += chunk_skip_role        

    Path(output_file).parent.mkdir(exist_ok=True, parents=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in preprocessed:
            f.write(
                json.dumps(sample, ensure_ascii=False) + '\n'
            )

    print(f"\nProcessing complete!")
    print(f"  Conversations skipped (missing content): {skip_miss_content}")
    print(f"  Conversations skipped (role mismatch): {skip_role_mismatch}")
    print(f"  Training samples created: {len(preprocessed)}")
    print(f"  Output saved to: {output_file}")
    return preprocessed


def process_all_splits(intent_gen_model: str,
                       tokenizer_model: str,
                       data_dir: str,
                       output_dir: str,
                       debug: bool) -> None:
    """Process train, val, and test splits"""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    tokenizer_model_string = tokenizer_model.replace('/', '--')

    stats = {}
    for split in ['test', 'val', 'train']:
        input_file = data_dir / f"{split}_with_intents_{intent_gen_model}.jsonl"
        output_file = output_dir / f"{split}_{tokenizer_model_string}_samples.jsonl"
        assert input_file.exists(), f"Input file {input_file} does not exist."
        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*80}")
        samples = process_file(str(input_file), str(output_file), tokenizer_model, perform_mp=not debug)
        stats[split] = len(samples)

    print(f"\n{'='*80}")
    print("FINAL STATISTICS")
    print(f"{'='*80}")
    for split, count in stats.items():
        print(f"  {split}: {count:,} samples")

    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Flip dialogues to create training samples')
    parser.add_argument('--data_dir',type=str,
                        default='./data_with_intents',
                        help='Directory with conversations and intents')
    parser.add_argument('--output_dir', type=str,
                        default='./training_data',
                        help='Output directory for training samples')
    parser.add_argument('--tokenizer', type=str,
                        default='meta-llama/Meta-Llama-3-8B',
                        help='Tokenizer to use')
    parser.add_argument('--intent_gen_model', type=str,
                        default='Qwen--Qwen3-32B',
                        help='Model had been used for intent generation')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Run in debug mode without multiprocessing')
    args = parser.parse_args()

    process_all_splits(
        intent_gen_model=args.intent_gen_model,
        tokenizer_model=args.tokenizer,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        debug=args.debug,
    )
