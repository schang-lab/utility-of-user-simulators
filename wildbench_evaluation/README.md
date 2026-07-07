# WildBench Evaluation

Evaluation harness built on [WildBench](https://arxiv.org/abs/2406.04770) (Lin et al., 2024) for scoring model responses on real-user tasks with an LLM judge.

Each WildBench v2 task comes with a per-task checklist. This harness runs an **binary checklist** protocol: a model answers each task, and the judge marks every checklist item as satisfied / not satisfied, yielding a per-item satisfaction rate. Two such runs (judged by the same judge) can then be turned into a **pairwise win rate** by counting which model satisfied more items per task.

Tasks are read from the Hub (`allenai/WildBench`) by default. Models under test and the judge can be OpenAI, Anthropic, Gemini, or any OpenAI-compatible local endpoint (e.g. a vLLM server). We use this to compare our various RL-trained assistant models.

## Scripts

| Script | Purpose |
|--------|---------|
| [wildbench_absolute_binary_checking.py](wildbench_absolute_binary_checking.py) | Per-item binary checklist satisfaction for a single model. Generation and judging are cached separately so multiple judges can reuse one set of generations. Local models generate in-process via `vllm.LLM`; API models use the OpenAI client. |
| [wildbench_absolute_binary_checking_then_pairwise.py](wildbench_absolute_binary_checking_then_pairwise.py) | Derive a pairwise win rate from two binary-checklist result files (the model with more satisfied items per task wins; equal counts are ties), i.e. the output from `wildbench_absolute_binary_checking.py`. |
| [wildbench_utils.py](wildbench_utils.py) | Shared helpers (dataset loading, message building, model generation) imported by the scripts above. Not run directly. |

## Installation

```bash
conda create -n usereval python=3.12 -y
conda activate usereval
pip install -r requirements.txt
```

Set the API key(s) for whichever providers you use:

```bash
export OPENAI_API_KEY=...      # gpt-* models
export ANTHROPIC_API_KEY=...   # claude-* models
export GEMINI_API_KEY=...      # gemini-* models
```

Model identifiers are routed by prefix: `gpt-*` → OpenAI, `claude-*` → Anthropic, `gemini-*` → Gemini, anything else → an OpenAI-compatible server at `http://localhost:<port>/v1` (set the port with the matching `--port_*` flag).

## Usage

### Binary checklist (single model)

```bash
python wildbench_absolute_binary_checking.py \
    --model gpt-4o --judge_model gpt-4o

# Local model generated with vLLM, judged by an API model:
python wildbench_absolute_binary_checking.py \
    --model my-local-model \
    --model_path /path/to/weights \
    --judge_model gpt-4o
```

where `--model` is any name one wants to save the result as (e.g., `qwen2.5-3b-instruct`) and `--model_path` is either the API model name (e.g., `gpt-4.1-mini`), path to the local Huggingface models, or the Huggingface model name (e.g., `Qwen/Qwen2.5-3B-Instruct`).
Outputs are laid out so judging can be repeated with different judges over one set of genersations:

```
<output_dir>/generation/<model>_generations.jsonl
<output_dir>/post_generation_judge/<model>_<judge>_details.jsonl
<output_dir>/post_generation_judge/<model>_<judge>_summary.json
```

If the generation cache already exists it is reused; if the judge details/summary already exist the run is skipped. The summary reports per-task and global satisfaction rates plus a per-category (primary tag) breakdown. Use `--n_tasks` to limit to the first N tasks while testing.
Resulting files will be located at `results/generation` and `results/post_generation_judge` by default.

### Pairwise win rate from two binary runs

Run the binary checklist for two models with the **same judge**, then compare:

```bash
python wildbench_absolute_binary_checking.py --model model-A --judge_model gpt-4o
python wildbench_absolute_binary_checking.py --model model-B --judge_model gpt-4o

python wildbench_absolute_binary_checking_then_pairwise.py \
    --model1 model-A --model2 model-B --judge_model gpt-4o
```

This reads the two `<model>_<judge>_details.jsonl` files from `--input_dir` (default `results/absolute_binary_checklist/post_generation_judge`), and for each shared session picks the winner by satisfied-item count. It reports win / tie / loss counts, raw and tie-adjusted win rates, and a per-category breakdown.
Resulting files will be located at `results/binary_pairwise/` by default.
