# SkyRL (trimmed copy)

This directory is a trimmed copy of [SkyRL](https://github.com/NovaSky-AI/SkyRL) (Apache-2.0, see [LICENSE](LICENSE)), the framework we used to train assistants against user simulators. It is based on upstream `main` at commit [`021b9d6`](https://github.com/NovaSky-AI/SkyRL/tree/021b9d65bc864cda6c6a81cf0146cafe4d0aecd2) (skyrl-train v0.3.0, December 2025).

Only `skyrl-train` and `skyrl-gym` are kept. Upstream documentation, tests, CI, integrations, example scripts, and the other bundled environments were removed.

## Modifications

Added:

- `skyrl-gym/skyrl_gym/envs/user_simulator/`: the `sftuser`, `rpuser`, and `rpuser_persona` environments, the paper's prompts, and the judge rubrics (Appendix E by default).
- `skyrl-train/examples/user_simulator/`: training and evaluation launch scripts.

Changed (each file is marked with a header comment):

| File | Change |
| --- | --- |
| `skyrl-gym/skyrl_gym/envs/__init__.py` | Registers only the user-simulator environments. |
| `skyrl-train/skyrl_train/config/skyrl_gym_config/default.yaml` | Defaults for the user-simulator environments. |
| `skyrl-train/skyrl_train/config/ppo_base_config.yaml` | Adds `data.filter_num_workers` and the optional `generator.{rollout,user_simulator,llm_judge}_log_path` JSONL logs. |
| `skyrl-train/skyrl_train/generators/skyrl_gym_generator.py` | Writes the JSONL logs above, logs user-simulator calls made in `env.init()`, and accumulates rollout logprobs across turns. |
| `skyrl-train/skyrl_train/entrypoints/main_base.py`, `main_generate.py` | Log level set by `LOG_LEVEL` (default `WARNING`); dataset filtering workers set by `data.filter_num_workers`. |
| `skyrl-train/skyrl_train/evaluate.py` | Logs one example evaluation input. |
| `skyrl-train/skyrl_train/inference_engines/inference_engine_client.py` | Per-request debug logging is opt-in. |
| `skyrl-train/pyproject.toml`, `uv.lock` | Adds `openai` as a direct dependency. |

The trainer, algorithms, workers, and inference engines are unchanged from upstream.
