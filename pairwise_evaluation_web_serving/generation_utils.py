"""
Generation helpers for the pairwise evaluation app.

Thin wrappers around the OpenAI-compatible chat API used to (a) build a client
for each assistant model in the study pool and (b) query the two models in
parallel at every turn.

Model identifiers are routed by prefix: ``gpt-*`` -> OpenAI, ``claude-*`` ->
Anthropic, ``gemini-*`` -> Gemini, anything else -> a local OpenAI-compatible
server (e.g. vLLM) at ``http://localhost:<port>/v1``.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import Callable, Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)


def get_client(model: str, port: int) -> OpenAI:
    """Return an OpenAI-compatible client for ``model``.

    For local servers the model name is checked against the server's model
    list, so a misconfigured pool fails at admin-config time rather than when
    the first participant sends a message.
    """
    if model.lower().startswith("gpt"):
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    if model.lower().startswith("claude"):
        return OpenAI(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url="https://api.anthropic.com/v1/",
        )
    if model.lower().startswith("gemini"):
        return OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    # Local vLLM / OpenAI-compatible server. The SDK requires a non-empty
    # api_key even when the server ignores it, so fall back to a sentinel.
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
        base_url=f"http://localhost:{port}/v1",
    )
    model_ids = [m.id for m in client.models.list().data]
    if model not in model_ids:
        raise ValueError(
            f"Model {model!r} not found on server at port {port}. Available: {model_ids}"
        )
    return client


def chat(
    client: OpenAI,
    model: str,
    messages: List[Dict],
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    max_retries: int = 8,
) -> str:
    """Call the chat completions API and return the response text ('' on failure)."""
    extra_kwargs: Dict = {"top_p": top_p} if top_p is not None else {}
    last_exc: Optional[Exception] = None
    for _ in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **extra_kwargs,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001 — retry on any API error
            last_exc = e
    logger.warning("Chat error for model %s: %s", model, last_exc)
    return ""


def build_agent_messages(history: List[Dict], user_query: str) -> List[Dict]:
    """Append the new user query to a {role, content} history (no system prompt)."""
    return history + [{"role": "user", "content": user_query}]


def generate_pair(
    client1: OpenAI,
    model1: str,
    hist1: List[Dict],
    client2: OpenAI,
    model2: str,
    hist2: List[Dict],
    user_query: str,
    max_tokens: int,
    temperature: float,
    top_p: Optional[float],
    is_allowed: Optional[Callable[[str], bool]] = None,
) -> Tuple[str, str]:
    """Generate responses from model1 and model2 in parallel for a single turn."""
    msgs1 = build_agent_messages(hist1, user_query)
    msgs2 = build_agent_messages(hist2, user_query)
    results: List[Optional[str]] = [None, None]

    def _gen(idx: int, client: OpenAI, model: str, msgs: List[Dict]) -> None:
        response = chat(client, model, msgs, max_tokens, temperature, top_p)
        while is_allowed is not None and not is_allowed(response):
            response = chat(client, model, msgs, max_tokens, temperature, top_p)
        results[idx] = response

    t1 = threading.Thread(target=_gen, args=(0, client1, model1, msgs1))
    t2 = threading.Thread(target=_gen, args=(1, client2, model2, msgs2))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    return results[0] or "", results[1] or ""
