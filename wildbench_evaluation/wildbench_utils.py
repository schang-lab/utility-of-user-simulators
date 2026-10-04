"""
WildBench shared generation helpers
===================================
Loading WildBench v2 and generating model responses, shared by the
WildBench evaluation scripts (e.g. wildbench_absolute_binary_checking.py).

Model identifiers are routed by prefix:
  - OpenAI API     (e.g. gpt-4o, gpt-4o-mini, gpt-4-turbo)
  - Anthropic API  (e.g. claude-3-5-sonnet-20241022)
  - Gemini API     (e.g. gemini-2.5-pro)
  - Local vLLM server, OpenAI-compatible  (any other name; pass --port_* 8000)
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _get_client(model: str, port: int) -> OpenAI:
    if model.lower().startswith("claude"):
        logger.info(f"Using Anthropic API client for model {model}")
        return OpenAI(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.anthropic.com/v1/",
        )

    if model.lower().startswith("gpt"):
        logger.info(f"Using OpenAI API client for model {model}")
        return OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )

    if model.lower().startswith("gemini"):
        logger.info(f"Using Gemini API client for model {model}")
        return OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

    logger.info(f"Using custom OpenAI-compatible client for model {model}")
    client = OpenAI(
        base_url="http://localhost:{port}/v1".format(port=port)
    )
    try:
        models_list = client.models.list()
        model_ids = [m.id for m in models_list.data]
        if model not in model_ids:
             logger.warning(f"Model {model} not found in the server's model list: {model_ids}")
    except Exception as e:
        logger.warning(f"Error connecting to custom client for model {model} on port {port}: {e}")
    return client


def load_wildbench(split: str = "test") -> List[Dict]:
    """
    Load WildBench v2 from HuggingFace and return list of task dicts.
    """
    def _format_history(turns: List[Dict]) -> str:
        if not turns:
            return "(No prior conversation history.)"
        lines = []
        for t in turns:
            role = "Human" if t["role"] == "user" else "AI"
            lines.append(f"{role}: {t['content']}")
        return "\n\n".join(lines)

    logger.info("Loading WildBench v2 dataset …")
    ds = load_dataset("allenai/WildBench", "v2", split=split)
    tasks = []
    for item in ds:
        turns = item["conversation_input"]
        last_user_idx = max(i for i, t in enumerate(turns) if t["role"] == "user")
        history_turns = turns[:last_user_idx]
        current_query = turns[last_user_idx]["content"]
        history_text = _format_history(history_turns)
        tasks.append(
            {
                "session_id": item["session_id"],
                "history": history_text,
                "history_list": history_turns,
                "user_query": current_query,
                "checklist": item["checklist"],
                "primary_tag": item.get("primary_tag"),
            }
        )
    logger.info("Loaded %d tasks.", len(tasks))
    return tasks


def _generate_all(
    client: OpenAI, model: str,
    tasks: List[Dict],
    max_workers: int,
    max_tokens: int, temperature: float = 0.0, top_p: Optional[float] = None,
) -> List[Dict]:
    """
    Generate responses for all tasks, return list of dicts aligned with tasks.
    Each dict has 'text', 'usage', and 'input_messages'.
    """
    responses: List[Optional[Dict]] = [None] * len(tasks)
    lock = threading.Lock()

    def _gen(idx: int) -> None:
        task = tasks[idx]
        result = generate_response(
            client, model, task["history_list"], task["user_query"],
            max_tokens, temperature, top_p=top_p,
        )
        with lock:
            responses[idx] = result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_gen, i) for i in range(len(tasks))]
        for fut in tqdm(as_completed(futures), total=len(tasks), desc=f"Generating [{model}]"):
            try:
                fut.result()
            except Exception as e:
                logger.warning("Generation error: %s", e)

    return responses


def generate_response(
    client: OpenAI, model: str,
    history: List[Dict],
    user_query: str,
    max_tokens: int = 2048, temperature: float = 0.0, top_p: Optional[float] = None,
    max_retries: int = 8,
) -> Dict:
    """
    Generate a single model response for a WildBench task.
    Returns a dict with 'text', 'usage', and 'input_messages'.
    """
    messages = _build_messages(history, user_query)

    extra_kwargs: Dict = {}
    if top_p is not None:
        extra_kwargs["top_p"] = top_p
    if model.lower().startswith("gpt"):
        if model.lower().startswith("gpt-5"):
            extra_kwargs["reasoning_effort"] = "minimal"
        extra_kwargs["max_completion_tokens"] = max_tokens
    else:
        extra_kwargs["max_tokens"] = max_tokens
    if model.lower().startswith("gemini"):
        extra_kwargs["extra_body"] = {"reasoning_effort": "none"}

    last_exc = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **extra_kwargs,
            )
            usage = resp.usage
            return {
                "text": resp.choices[0].message.content,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens if usage else None,
                    "completion_tokens": usage.completion_tokens if usage else None,
                    "total_tokens": usage.total_tokens if usage else None,
                },
                "input_messages": messages,
            }
        except Exception as e:
            last_exc = e
    logger.warning("Generation error for model %s: %s", model, last_exc)
    return {
        "text": "Failed to generate response.",
        "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
        "input_messages": messages,
    }


def _build_messages(history: List[Dict],
                    user_query: str) -> List[Dict]:
    messages: List[Dict] = []
    _expected_role = "user"
    for uttr in history:
        _role = uttr["role"]
        assert _role == _expected_role, f"Expected role {_expected_role} but got {_role} in history"
        messages.append(
            {"role": _role, "content": uttr["content"]}
        )
        _expected_role = "assistant" if _expected_role == "user" else "user"
    messages.append(
        {"role": "user", "content": user_query}
    )
    return messages
