"""
WildBench Binary Checklist Evaluation
======================================
Evaluates an AI model's response against each WildBench checklist item individually,
producing a binary (satisfied / not satisfied) verdict per item.

Unlike WB-Score (which uses the checklist as a soft guide for a holistic 1-10 score),
this script asks the judge to explicitly evaluate each bullet point.

Generation and judging are split so that judging with multiple judge models can
reuse a single set of generations:

    <output_dir>/generation/<model>_generations.jsonl
    <output_dir>/post_generation_judge/<model>_<judge>_details.jsonl
    <output_dir>/post_generation_judge/<model>_<judge>_summary.json

If the generation cache already exists, generation is skipped and only judging runs.

Local models are loaded in-process via `vllm.LLM`
and generated with a single batched `LLM.chat(...)` call.
API models (gpt*/claude*/gemini*) keep using the OpenAI client + thread pool path.

Usage:
    python wildbench_absolute_binary_checking.py --model gpt-4o --judge_model gpt-4o
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

from wildbench_utils import (
    _build_messages,
    _get_client,
    load_wildbench,
    _generate_all,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


BINARY_CHECKLIST_PROMPT = """\
# Instruction
You are an expert evaluator. Your task is to evaluate an AI-generated response against a set of specific quality criteria. For **each** criterion you must make a binary judgment: satisfied or not satisfied.

# Conversation between User and AI
## History
<|begin_of_history|>
{history}
<|end_of_history|>

## Current User Query
<|begin_of_query|>
{user_query}
<|end_of_query|>

## AI Response
<|begin_of_response|>
{model_output}
<|end_of_response|>

# Evaluation Criteria
You must evaluate the AI response against each of the following checklist items. For every item, decide whether the response **satisfies** the criterion (true) or **does not satisfy** it (false).

{numbered_checklist}

# Rules
- Evaluate each item independently.
- A criterion is "satisfied" only if the response **clearly and sufficiently** meets it. If the response partially meets a criterion but has notable gaps, mark it as not satisfied.
- Provide a brief reasoning (1-2 sentences) for each judgment.

# Output Format
Return a JSON array with one object per checklist item, in the same order as the criteria above. Each object must have exactly three fields: "item" (the criterion number, 1-indexed), "satisfied" (boolean true/false), and "reasoning" (string).

```json
[
  {{"item": 1, "satisfied": true, "reasoning": "..."}},
  {{"item": 2, "satisfied": false, "reasoning": "..."}}
]
```

IMPORTANT: You must output a valid JSON array inside a ```json code block. The array must contain exactly {n_items} objects, one per criterion, in order."""


def _safe_name(name: str) -> str:
    return name.replace("/", "_")


def _is_api_model(model: str) -> bool:
    name = model.lower()
    return name.startswith("gpt") or name.startswith("claude") or name.startswith("gemini")


def _generate_with_vllm(
    model: str,
    model_path: str,
    tasks: List[Dict],
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = 32768,
    cuda_devices: Optional[str] = None,
) -> List[Dict]:
    """
    Load `model_path` in-process with vLLM and run a single batched chat()
    call over all tasks. Returns generations aligned with `tasks` in the same
    {"text", "usage", "input_messages"} schema as the OpenAI-client path.
    """
    if cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

    from vllm import LLM, SamplingParams  # lazy: heavy import, only needed for local

    logger.info(
        "Loading vLLM in-process: model_path=%s, tp=%d, gpu_mem_util=%.2f, max_len=%d",
        model_path, tensor_parallel_size, gpu_memory_utilization, max_model_len,
    )
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=True,
    )

    sp_kwargs: Dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    if top_p is not None:
        sp_kwargs["top_p"] = top_p
    sampling_params = SamplingParams(**sp_kwargs)

    messages_list = [_build_messages(t["history_list"], t["user_query"]) for t in tasks]

    logger.info("Running batched chat() over %d tasks ...", len(messages_list))
    outputs = llm.chat(messages_list, sampling_params=sampling_params)

    gen: List[Dict] = []
    for msgs, out in zip(messages_list, outputs):
        first = out.outputs[0] if out.outputs else None
        prompt_tokens = len(out.prompt_token_ids) if out.prompt_token_ids is not None else None
        completion_tokens = (
            len(first.token_ids) if first is not None and first.token_ids is not None else None
        )
        total = (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None else None
        )
        gen.append({
            "text": first.text if first is not None else "",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
            },
            "input_messages": msgs,
        })

    # Free the model so the judging step (or any later GPU work) can use the device.
    del llm
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    return gen


def _extract_checklist_results(text: str, n_items: int) -> Optional[List[Dict]]:
    """
    Extract the JSON array of per-item binary verdicts from judge output.
    Returns a list of dicts or None on failure.
    """
    code_block = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    raw = code_block.group(1) if code_block else text

    # Try to find a JSON array in the text
    if not code_block:
        arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if arr_match:
            raw = arr_match.group(0)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Judge models sometimes emit LaTeX like \( \theta \cos inside JSON strings,
        # which are invalid JSON escape sequences. Fix them and retry.
        fixed = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', raw)
        try:
            data = json.loads(fixed)
        except json.JSONDecodeError:
            return None

    if not isinstance(data, list):
        return None

    # Validate structure
    results = []
    for entry in data:
        if not isinstance(entry, dict):
            return None
        satisfied = entry.get("satisfied")
        if not isinstance(satisfied, bool):
            # Try to coerce string "true"/"false"
            if isinstance(satisfied, str):
                if satisfied.lower() == "true":
                    satisfied = True
                elif satisfied.lower() == "false":
                    satisfied = False
                else:
                    return None
            else:
                return None
        results.append({
            "item": entry.get("item"),
            "satisfied": satisfied,
            "reasoning": str(entry.get("reasoning", "")),
        })

    if len(results) != n_items:
        logger.warning("Expected %d checklist items but got %d", n_items, len(results))
        # Accept if we got at least something
        if len(results) == 0:
            return None

    return results


def judge_binary_checklist(
    judge_client: OpenAI, judge_model: str,
    task: Dict,
    model_output: str,
    judge_max_tokens: int = 32768,
    max_retries: int = 8,
) -> Dict:
    """
    Ask the judge to evaluate each checklist item as satisfied/not satisfied.
    """
    checklist = task["checklist"]
    n_items = len(checklist)
    numbered_checklist = "\n".join(
        f"{i+1}. {item}" for i, item in enumerate(checklist)
    )
    prompt = BINARY_CHECKLIST_PROMPT.format(
        history=task["history"],
        user_query=task["user_query"],
        model_output=model_output,
        numbered_checklist=numbered_checklist,
        n_items=n_items,
    )
    judge_messages = [{"role": "user", "content": prompt}]

    checklist_results = None
    raw_output = ""
    judge_usage: Dict = {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}

    judge_lower = judge_model.lower()
    is_gpt = judge_lower.startswith("gpt")
    is_gpt5 = judge_lower.startswith("gpt-5")

    extra_kwargs: Dict = {}
    if is_gpt: # gpt-5 uses max_completion_tokens instead of max_tokens
        extra_kwargs["max_completion_tokens"] = judge_max_tokens
    else:
        extra_kwargs["max_tokens"] = judge_max_tokens

    for _retry in range(max_retries):
        try:
            retry_temp = 0.0 if _retry == 0 else 0.3
            resp = judge_client.chat.completions.create(
                model=judge_model,
                messages=judge_messages,
                temperature=1 if is_gpt5 else retry_temp, # gpt-5 accepts only temp=1
                **extra_kwargs,
            )
            raw_output = resp.choices[0].message.content or ""
            usage = resp.usage
            if usage:
                judge_usage = {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "total_tokens": usage.total_tokens,
                }
            extracted = _extract_checklist_results(raw_output, n_items)
            if extracted is not None:
                checklist_results = extracted
                break
            else:
                logger.warning(
                    "Judge retry %d/%d: could not extract checklist results (first 20000 chars): %s",
                    _retry + 1, max_retries, raw_output[:20000],
                )
        except Exception as e:
            logger.warning("Judge retry %d/%d error: %s", _retry + 1, max_retries, e)
            time.sleep(_retry * 2 + 0.5)

    failed = checklist_results is None
    if failed:
        return {
            "checklist_results": None,
            "satisfaction_rate": None,
            "raw_output": raw_output,
            "usage": judge_usage,
            "input_messages": judge_messages,
            "failed": True,
        }

    n_satisfied = sum(1 for r in checklist_results if r["satisfied"])
    satisfaction_rate = n_satisfied / len(checklist_results) if checklist_results else 0.0

    return {
        "checklist_results": checklist_results,
        "satisfaction_rate": satisfaction_rate,
        "n_satisfied": n_satisfied,
        "n_total": len(checklist_results),
        "raw_output": raw_output,
        "usage": judge_usage,
        "input_messages": judge_messages,
        "failed": False,
    }


def _generate_or_load_cache(
    model: str,
    port_model: int,
    tasks: List[Dict],
    gen_path: Path,
    max_workers_gen: int,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    model_path: Optional[str] = None,
    cuda_devices: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = 32768,
) -> List[Dict]:
    """
    Return generations aligned with tasks.
    Load from gen_path if it exists, otherwise generate and persist.
    Local models go through in-process `vllm.LLM().chat(...)`;
    API models (gpt*/claude*/gemini*) use the OpenAI client + thread pool path.
    """
    if gen_path.exists():
        logger.info("Loading cached generations from %s (skipping generation step) …", gen_path)
        gen: List[Optional[Dict]] = [None] * len(tasks)
        session_to_idx = {t["session_id"]: i for i, t in enumerate(tasks)}
        with open(gen_path) as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line:
                    continue
                _rec = json.loads(_line)
                _idx = session_to_idx.get(_rec["session_id"])
                if _idx is not None:
                    gen[_idx] = _rec["gen"]
        missing = sum(1 for g in gen if g is None)
        if missing:
            logger.warning("%d tasks have no cached generation; they will produce empty outputs.", missing)
            for i in range(len(tasks)):
                if gen[i] is None:
                    gen[i] = {"text": "", "usage": {}, "input_messages": []}
        return gen  # type: ignore[return-value]

    if _is_api_model(model):
        logger.info("Generating responses for API model %s …", model)
        client = _get_client(model, port_model)
        gen = _generate_all(
            client, model, tasks, max_workers_gen, max_tokens, temperature, top_p,
        )
    else:
        gen = _generate_with_vllm(
            model=model,
            model_path=model_path or model,
            tasks=tasks,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            cuda_devices=cuda_devices,
        )

    gen_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Saving generations cache to %s …", gen_path)
    with open(gen_path, "w") as _f:
        for i, task in enumerate(tasks):
            _f.write(json.dumps({
                "session_id": task["session_id"],
                "primary_tag": task.get("primary_tag"),
                "gen": gen[i],
            }) + "\n")
    return gen


def evaluate_binary_checklist(
    model: str,
    judge_model: str,
    port_model: int = 8000,
    port_judge: int = 8002,
    max_workers: int = 8,
    max_workers_gen: int = 8,
    max_tokens: int = 2048,
    judge_max_tokens: int = 32768,
    output_dir: str = "results/absolute_binary_checklist",
    n_tasks: Optional[int] = None,
    temperature: float = 0.0,
    top_p: Optional[float] = None,
    model_path: Optional[str] = None,
    cuda_devices: Optional[str] = None,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int = 32768,
) -> Dict:
    """
    Run binary checklist evaluation for a single model.
    """
    output_path = Path(output_dir)
    gen_dir = output_path / "generation"
    judge_dir = output_path / "post_generation_judge"
    gen_dir.mkdir(parents=True, exist_ok=True)
    judge_dir.mkdir(parents=True, exist_ok=True)

    safe_model = _safe_name(model)
    safe_judge = _safe_name(judge_model)

    out_prefix = f"{safe_model}_{safe_judge}"
    detail_path = judge_dir / f"{out_prefix}_details.jsonl"
    summary_path = judge_dir / f"{out_prefix}_summary.json"
    if detail_path.exists() and summary_path.exists():
        logger.info(
            "Skipping %s (judge=%s): outputs already exist at %s and %s",
            model, judge_model, detail_path, summary_path,
        )
        with open(summary_path) as f:
            summary = json.load(f)
        _print_summary(summary)
        return {"summary": summary, "results": None}

    tasks = load_wildbench()
    if n_tasks:
        tasks = tasks[:n_tasks]

    gen_path = gen_dir / f"{safe_model}_generations.jsonl"
    gen = _generate_or_load_cache(
        model=model, port_model=port_model, tasks=tasks, gen_path=gen_path,
        max_workers_gen=max_workers_gen, max_tokens=max_tokens,
        temperature=temperature, top_p=top_p,
        model_path=model_path,
        cuda_devices=cuda_devices,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
    )

    # Judge each response
    judge_client = _get_client(judge_model, port_judge)
    logger.info("Running binary checklist evaluation with %s ...", judge_model)

    def _judge_one(idx: int) -> Dict:
        task = tasks[idx]
        g = gen[idx]
        model_text = g["text"]
        j = judge_binary_checklist(
            judge_client, judge_model, task, model_text,
            judge_max_tokens=judge_max_tokens,
        )

        return {
            "session_id": task["session_id"],
            "primary_tag": task["primary_tag"],
            "checklist": task["checklist"],
            "model_input": g["input_messages"],
            "model_output": model_text,
            "model_usage": g["usage"],
            "judge_result": j,
            "judge_failed": j["failed"],
            "checklist_results": j["checklist_results"],
            "satisfaction_rate": j["satisfaction_rate"],
        }

    results: List[Dict] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_judge_one, i): i for i in range(len(tasks))}
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="Binary Checklist"):
            try:
                rec = fut.result()
                with lock:
                    results.append(rec)
            except Exception as e:
                logger.warning("Error evaluating task %d: %s", futures[fut], e)

    # Aggregate
    judge_failures = sum(1 for r in results if r["judge_failed"])
    valid = [r for r in results if not r["judge_failed"]]
    sat_rates = [r["satisfaction_rate"] for r in valid]
    avg_sat_rate = sum(sat_rates) / len(sat_rates) if sat_rates else 0.0

    total_items = sum(len(r["checklist_results"]) for r in valid)
    total_satisfied = sum(
        sum(1 for c in r["checklist_results"] if c["satisfied"])
        for r in valid
    )
    global_sat_rate = total_satisfied / total_items if total_items else 0.0

    # Per-category breakdown
    category_rates: Dict[str, List[float]] = {}
    for r in valid:
        tag = r["primary_tag"] or "unknown"
        category_rates.setdefault(tag, []).append(r["satisfaction_rate"])
    category_avg = {tag: round(sum(s) / len(s) * 100, 2) for tag, s in category_rates.items()}

    # Token usage
    model_usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    judge_usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in results:
        for k in model_usage_total:
            v = (r.get("model_usage") or {}).get(k)
            if v is not None:
                model_usage_total[k] += v
            v = ((r.get("judge_result") or {}).get("usage") or {}).get(k)
            if v is not None:
                judge_usage_total[k] += v

    summary = {
        "model": model,
        "n_tasks": len(valid),
        "avg_satisfaction_rate": round(avg_sat_rate * 100, 2),
        "global_satisfaction_rate": round(global_sat_rate * 100, 2),
        "total_checklist_items": total_items,
        "total_satisfied": total_satisfied,
        "judge_failures": judge_failures,
        "judge_model": judge_model,
        "category_satisfaction_rates": category_avg,
        "sampling_params": {
            "temperature": temperature,
            "top_p": top_p,
        },
        "token_usage": {
            "model": model_usage_total,
            "judge": judge_usage_total,
        },
    }

    # Save under post_generation_judge/
    with open(detail_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Judge results saved to %s", judge_dir)
    _print_summary(summary)
    return {"summary": summary, "results": results}


def _print_summary(s: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"Binary Checklist Evaluation: {s['model']}")
    print("=" * 60)
    print(f"  Tasks evaluated          : {s['n_tasks']}")
    print(f"  Avg satisfaction rate     : {s['avg_satisfaction_rate']:.1f}%  (per-task avg)")
    print(f"  Global satisfaction rate  : {s['global_satisfaction_rate']:.1f}%  ({s['total_satisfied']}/{s['total_checklist_items']} items)")
    print(f"  Judge failures           : {s.get('judge_failures', 0)}")
    print(f"  Judge model              : {s['judge_model']}")
    sp = s.get("sampling_params", {})
    print(f"  Temperature              : {sp.get('temperature')}")
    print(f"  Top-p                    : {sp.get('top_p')}")
    usage = s.get("token_usage", {})
    for who in ("model", "judge"):
        u = usage.get(who, {})
        label = s["model"] if who == "model" else f"judge({s['judge_model']})"
        print(f"  Tokens [{label}]: prompt={u.get('prompt_tokens')} completion={u.get('completion_tokens')}")
    cat = s.get("category_satisfaction_rates", {})
    if cat:
        print(f"  --- Per-category satisfaction rate ---")
        for tag, avg in sorted(cat.items(), key=lambda x: -x[1]):
            print(f"    {tag:30s}: {avg:.1f}%")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="WildBench binary checklist evaluation.")
    parser.add_argument("--model",
                        required=True,
                        help="Model identifier to evaluate.")
    parser.add_argument("--port_model",
                        type=int, default=8000,
                        help="Port used only when --model is an OpenAI-compatible custom "
                             "endpoint (default 8000). Ignored for local models (which now "
                             "load in-process via vllm.LLM) and for API models.")
    parser.add_argument("--judge_model",
                        required=True,
                        help="Judge model identifier.")
    parser.add_argument("--port_judge",
                        type=int, default=8002,
                        help="Port for local vLLM server for judge model (default 8002).")
    parser.add_argument("--max_workers",
                        type=int, default=64,
                        help="Thread pool concurrency for judging.")
    parser.add_argument("--max_workers_gen",
                        type=int, default=256,
                        help="Thread pool concurrency for generation.")
    parser.add_argument("--max_tokens",
                        type=int, default=8192,
                        help="Max tokens per generation.")
    parser.add_argument("--judge_max_tokens",
                        type=int, default=32768,
                        help="Max tokens per judge response (default 32768).")
    parser.add_argument("--output_dir",
                        default="results/absolute_binary_checklist",
                        help="Directory for output files. Generation cache goes to "
                             "<output_dir>/generation/, judge results to "
                             "<output_dir>/post_generation_judge/.")
    parser.add_argument("--n_tasks",
                        type=int, default=None,
                        help="Limit evaluation to first N tasks (useful for testing).")
    parser.add_argument("--temperature",
                        type=float, default=0.0,
                        help="Sampling temperature for model (default 0.0; not applied to judge).")
    parser.add_argument("--top_p",
                        type=float, default=0.9,
                        help="Top-p nucleus sampling for model (default: not set; not applied to judge).")
    parser.add_argument("--model_path",
                        default=None,
                        help="Path or HF repo id of the weights to load with vllm.LLM(). "
                             "Defaults to --model when not set. Only used for local models "
                             "on a generation-cache miss.")
    parser.add_argument("--cuda_devices",
                        default=None,
                        help="Value to set as CUDA_VISIBLE_DEVICES before vllm.LLM() is "
                             "instantiated (e.g. '0' or '1,2'). Inherited from environment "
                             "if unset.")
    parser.add_argument("--gpu_memory_utilization",
                        type=float, default=0.9,
                        help="vllm.LLM(gpu_memory_utilization=...) (default 0.9).")
    parser.add_argument("--tensor_parallel_size",
                        type=int, default=1,
                        help="vllm.LLM(tensor_parallel_size=...) (default 1).")
    parser.add_argument("--max_model_len",
                        type=int, default=32768,
                        help="vllm.LLM(max_model_len=...) (default 32768).")
    args = parser.parse_args()

    evaluate_binary_checklist(
        model=args.model, port_model=args.port_model,
        judge_model=args.judge_model, port_judge=args.port_judge,
        max_workers=args.max_workers,
        max_workers_gen=args.max_workers_gen,
        max_tokens=args.max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        output_dir=args.output_dir,
        n_tasks=args.n_tasks,
        temperature=args.temperature,
        top_p=args.top_p,
        model_path=args.model_path,
        cuda_devices=args.cuda_devices,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )


if __name__ == "__main__":
    main()
