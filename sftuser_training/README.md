# User Simulator SFT (SFTuser)

Supervised fine-tuning pipeline for training a **user language model** (a model that plays the *user* side of a conversation) from WildChat. It reproduces the "flipping the dialogue" recipe: preprocess WildChat, summarize each conversation's intent, mask so that only user turns contribute to the loss, and fine-tune a base LLM.

Training is driven by a lightly customized fork of [`llama-cookbook`](https://github.com/meta-llama/llama-cookbook) (in [llama-cookbook/](llama-cookbook/)) that adds a `userlm_dataset` and a `<|endconversation|>` special token (token embedding initialized from the EOS token of base LLM vocabulary), plus FSDP + torch.compile implementation. For more details, please refer to the Appendix of our paper.

## Pipeline

| Step | Script | Input → Output |
|------|--------|----------------|
| 1. Preprocess | [1_load_and_preprocess_wildchat.py](1_load_and_preprocess_wildchat.py) | WildChat → `processed_data/{split:train,val,test}.jsonl` |
| 2. Generate intents | [2_generate_intents.py](2_generate_intents.py) | `processed_data/` → `data_with_intents/{split}_with_intents_{intent-generation-model}.jsonl` |
| 3. Flip & tokenize | [3_flip_dialogue_prepare_training.py](3_flip_dialogue_prepare_training.py) | `data_with_intents/` → `training_data/{split}_{base-model-tokenizer}_samples.jsonl` |
| 4. Prepare tokenizer | [4_prepare_tokenizer.py](4_prepare_tokenizer.py) | base tokenizer → `tokenizers_and_configs--{base-model}/` |
| 5. Train | [launch_multi_gpu.sh](launch_multi_gpu.sh) | `training_data/`, `tokenizers_and_configs--{base-model}/` → checkpoints |

## Installation

Tested with Python 3.12, torch 2.7 (CUDA 12.8), transformers 4.55.
Please review `requirements.txt` for the specific CUDA version we used and modify according to your compute situation.

```bash
conda create -n usersim python=3.12 -y
conda activate usersim
pip install -e llama-cookbook
pip install -r requirements.txt
```

You will also need a Hugging Face account with access to the base model you fine-tune (e.g. `meta-llama/Meta-Llama-3-8B`): `huggingface-cli login`.

## Usage

```bash
# 1. Preprocess WildChat (streams allenai/WildChat-1M from the Hub)
python 1_load_and_preprocess_wildchat.py --output_dir ./processed_data

# 2. Generate intents. Use an OpenAI model...
export OPENAI_API_KEY=...
python 2_generate_intents.py \
    --data_dir ./processed_data \
    --output_dir ./data_with_intents \
    --model gpt-4.1-mini
# ...or a local model served with an OpenAI-compatible endpoint (e.g. vLLM):
#   vllm serve Qwen/Qwen3-32B --port 8000; vllm serve Qwen/Qwen3-32B --port 8001
#   python 2_generate_intents.py --model Qwen/Qwen3-32B --ports 8000 8001

# 3. Flip dialogues into masked training samples
python 3_flip_dialogue_prepare_training.py \
    --data_dir ./data_with_intents \
    --output_dir ./training_data \
    --tokenizer Qwen/Qwen2.5-14B-Instruct \
    --intent_gen_model Qwen/Qwen3-32B

# 4. Add the <|endconversation|> token to the tokenizer
python 4_prepare_tokenizer.py \
    --base_tokenizer Qwen/Qwen2.5-14B-Instruct \
    --output_dir ./tokenizers_and_configs

# 5. Train (multi-GPU FSDP)
NUM_GPUS=2 MODEL_NAME=meta-llama/Meta-Llama-3-8B \
    DATA_PATH=./training_data OUTPUT_DIR=./userlm_checkpoints \
    ./launch_multi_gpu.sh
```

Step 5 is configured through environment variables (`NUM_GPUS`, `MODEL_NAME`, `TOTAL_BS`, `GRAD_ACC`, `DATA_PATH`, `OUTPUT_DIR`, ...); see the top of [launch_multi_gpu.sh](launch_multi_gpu.sh). Weights & Biases logging is on by default — edit the `--wandb_config.*` flags in that script or pass `--use_wandb False`.

We also provide two conversion scripts (`convert_full_dict_to_safetensor.py` used when trained with world_size=1; `convert_sharded_dict_to_safetensor.py` used with FSDP training) for converting trained model weights into vLLM-compatible safetensor format.

## Supported base models

Chat templates for the masking step live in [tokenizers_and_configs/tokenizer_configs.py](tokenizers_and_configs/tokenizer_configs.py) and currently cover the Llama-3 and Qwen2.5 families.
