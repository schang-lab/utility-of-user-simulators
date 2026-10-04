# Quantifying the Utility of User Simulators for Building Collaborative LLM Assistants

User simulators are widely used to train interactive LLM assistants, but it is unclear how to measure whether a simulator is any good. We propose to measure a simulator by its **downstream utility**: how well an assistant trained against it performs with real people.

We train the same initial assistant with multi-turn reinforcement learning against a range of user simulators:
- **RPUser1–3**: an instruction-tuned LLM prompted to role-play a user, made progressively more realistic with a real opening turn and sampled personas.
- **SFTUser**: an LLM fine-tuned on the user turns of WildChat.

Only the simulator varies between runs. We then evaluate the trained assistants in three ways: a user study with 283 participants, WildBench, and cross-simulator evaluation. In the user study, the SFTUser-trained assistant has a 58% win rate against the initial assistant and 57% against the RPUser1-trained assistant. The RPUser1-trained assistant is statistically indistinguishable from the initial one (51%).

This repository contains the code for every stage of that pipeline.

## Repository layout

| Directory | Contents | Paper |
|---|---|---|
| [sftuser_training/](sftuser_training/README.md) | WildChat preprocessing, user-intent generation, and supervised fine-tuning of the learned user simulator (SFTUser) on a customized copy of llama-cookbook | §3.2, App. C |
| [rl_training/](rl_training/README.md) | GRPO training of assistants against SFTUser and RPUser1–3 with an LLM-judge reward, and cross-simulator evaluation, on a trimmed copy of SkyRL | §2, §3.1, §4.1, §4.3, §4.4, App. D–F |
| [wildbench_evaluation/](wildbench_evaluation/README.md) | WildBench checklist satisfaction rates and pairwise win rates between assistants | §4.2, App. G |
| [pairwise_evaluation_web_serving/](pairwise_evaluation_web_serving/README.md) | Web app for the human pairwise study (writing tasks, side-by-side comparison, preference collection) and the quality-control scripts used to screen submissions | §5, App. H |

Each directory is self-contained, with its own README, dependencies, and installation instructions. The components pin different GPU stacks, so use a separate Python environment for each. `rl_training` manages its own environment with uv.

## Released user simulators

| Model | Base model | Training data |
|---|---|---|
| [`jjssuh/sftuser-base-Qwen2.5-14B-Instruct`](https://huggingface.co/jjssuh/sftuser-base-Qwen2.5-14B-Instruct) (SFTUser) | Qwen2.5-14B-Instruct | WildChat-1M |
| [`jjssuh/sftuser-base-Qwen2.5-7B-Instruct`](https://huggingface.co/jjssuh/sftuser-base-Qwen2.5-7B-Instruct) | Qwen2.5-7B-Instruct | WildChat-1M |
| [`jjssuh/sftuser-base-Qwen2.5-32B-Instruct`](https://huggingface.co/jjssuh/sftuser-base-Qwen2.5-32B-Instruct) | Qwen2.5-32B-Instruct | WildChat-1M |
| [`jjssuh/sftuser2-base-Qwen2.5-14B-Instruct`](https://huggingface.co/jjssuh/sftuser2-base-Qwen2.5-14B-Instruct) (SFTUser-WC-4.8M) | Qwen2.5-14B-Instruct | WildChat-4.8M |

These checkpoints include the tokenizer with the `<|endconversation|>` token that the simulator emits to end a conversation, and can be served directly with vLLM.

## Reproducing the paper

1. **Learned user simulator.** In [sftuser_training](sftuser_training/README.md), preprocess WildChat-1M, generate a user intent for each conversation, and split the data into train/val/test by user. Then fine-tune SFTUser, or skip the fine-tuning and use a released checkpoint. The splits are still needed in step 2.
2. **Assistant training.** In [rl_training](rl_training/README.md), convert the same splits to RL data, serve the frozen simulator and the two judges with vLLM, and train the assistant against each simulator (`SIMULATOR=sftuser|rpuser1|rpuser2|rpuser3`).
3. **Evaluation.**
   - WildBench checklist and pairwise win rates: [wildbench_evaluation](wildbench_evaluation/README.md).
   - Cross-simulator evaluation on the held-out WildChat test split: `run_eval.sh` in [rl_training](rl_training/README.md).
   - Human study: [pairwise_evaluation_web_serving](pairwise_evaluation_web_serving/README.md).

Datasets (WildChat, WildBench) are downloaded from the Hugging Face Hub by the scripts. Participant records from the user study are not included in this repository.

## License

The code in this repository is released under the [BSD 3-Clause License](LICENSE), except for these third-party components, which keep their original licenses:

- [`sftuser_training/llama-cookbook/`](sftuser_training/llama-cookbook/): customized copy of [llama-cookbook](https://github.com/meta-llama/llama-cookbook), MIT License.
- [`rl_training/SkyRL/`](rl_training/SkyRL/README.md): trimmed and modified copy of [SkyRL](https://github.com/NovaSky-AI/SkyRL), Apache License 2.0; modifications are listed in its README.

Datasets and model weights used or released here are subject to their own licenses and terms of use.
