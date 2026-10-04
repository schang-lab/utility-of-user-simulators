"""Multi-turn conversation environments in which a user simulator plays the user (Sections 2-3).

The assistant under training only observes the conversation. The user simulator and the LLM judges
additionally see the user intent z. A conversation ends when the simulator emits the terminal signal
or after `max_turns` assistant responses; the judges then score the full conversation with the rubric
selected by `judge_rubric` (default: Appendix E), and the reward is the average normalized judge score.

Registered environments (see `skyrl_gym/envs/__init__.py`):
    sftuser         learned user simulator fine-tuned on WildChat (Section 3.2, Appendix C.4)
    rpuser          role-playing LLM; RPUSER1 when `generate_first_turn` is true, otherwise RPUSER2
                    (Appendices D.1-D.2)
    rpuser_persona  RPUSER2 with a persona guideline sampled per conversation (RPUSER3, Appendix D.3)
"""

import ast
import json
import os
import random
import re
import threading
import time
from typing import Any, ClassVar, Dict, Optional, Tuple

import httpx
from loguru import logger
from omegaconf import DictConfig
from openai import OpenAI
from transformers import AutoTokenizer

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType
from skyrl_gym.envs.user_simulator.judge_rubrics import DEFAULT_JUDGE_RUBRIC, JUDGE_RUBRICS
from skyrl_gym.envs.user_simulator.prompts import (
    RPUSER_JSON_SCHEMA,
    RPUSER_PROMPTS,
    RPUSER_TASK_DESCRIPTION,
)

logger.disable("httpx")
logger.disable("httpcore")

TERMINAL_SIGNAL = "<|endconversation|>"

# A new env is constructed per trajectory, so HTTP clients are shared per base_url instead of being
# built per instance. The pool is bounded so that the number of live connections per endpoint stays
# fixed under bursts; keep it at or above `environment.skyrl_gym.max_env_workers`.
_HTTP_LIMITS = httpx.Limits(max_connections=48, max_keepalive_connections=48, keepalive_expiry=30.0)
# `pool` is how long a request waits for a free connection; requests queue once the pool is in use.
_HTTP_TIMEOUT = httpx.Timeout(timeout=1800.0, connect=10.0, read=1800.0, write=1800.0, pool=600.0)
_CLIENTS: Dict[str, OpenAI] = {}
_CLIENTS_LOCK = threading.Lock()


def _get_client(base_url: str) -> OpenAI:
    """Return the process-wide OpenAI-compatible client for `base_url`."""
    with _CLIENTS_LOCK:
        if base_url not in _CLIENTS:
            _CLIENTS[base_url] = OpenAI(
                base_url=base_url,
                # Local vLLM servers accept any token; the SDK only requires a non-empty one.
                api_key=os.environ.get("OPENAI_API_KEY") or "EMPTY",
                http_client=httpx.Client(limits=_HTTP_LIMITS, timeout=_HTTP_TIMEOUT),
                timeout=_HTTP_TIMEOUT,
                max_retries=5,
            )
        return _CLIENTS[base_url]


def _thinking_disabled(model: str) -> Dict[str, Any]:
    """Qwen3 models are queried in non-thinking mode."""
    if "qwen3" in model.lower():
        return {"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    return {}


def _outer_braces_span(s: str) -> Optional[Tuple[int, int]]:
    """Return (start, end_exclusive) of the outermost {...} in `s`, skipping braces inside strings."""
    start = s.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    return None


def extract_outer_dict(s: str) -> Dict[str, Any]:
    """Parse the outermost {...} in a free-form LLM response (used when structured output is off)."""
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.DOTALL).strip()
    span = _outer_braces_span(s)
    if span is None:
        raise ValueError(f"No balanced {{...}} found in: {s[:6400]!r}")
    obj_str = s[span[0] : span[1]]
    try:
        parsed = ast.literal_eval(obj_str)
    except Exception as e1:
        try:
            parsed = json.loads(obj_str, strict=False)
        except json.JSONDecodeError as e2:
            raise ValueError(
                f"Failed to parse as Python literal: {e1}\nFailed to parse as JSON: {e2}\nObject: {obj_str[:6400]!r}"
            )
    if not isinstance(parsed, dict):
        raise ValueError(f"Extracted object is not a dict but {type(parsed)}")
    return parsed


def _extract_judge_dict(s: str) -> Dict[str, Any]:
    """Like `extract_outer_dict`, with a regex fallback for the score field.

    Some judges emit single-quoted values with unescaped apostrophes (e.g. "thought": 'it's good'),
    which are neither valid JSON nor valid Python literals.
    """
    try:
        return extract_outer_dict(s)
    except ValueError:
        score_match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]*)?)', s)
        if score_match is None:
            raise
        return {"score": float(score_match.group(1)), "thought": None}


def format_conversation(conversation: ConversationType) -> str:
    """Render a conversation as 'USER: ...' / 'ASSISTANT: ...' blocks separated by blank lines."""
    return "\n\n".join(f"{message['role'].upper()}: {message['content']}" for message in conversation)


class UserSimulatorEnv(BaseTextEnv):
    """Shared logic: conversation bookkeeping, termination, and the multi-judge reward.

    Subclasses implement `_generate_user_turn`, which returns the next user utterance and metadata.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()
        self._extra_info = extras.get("extra_info", {})
        self._intent = self._extra_info.get("intent")
        if not self._intent:
            raise ValueError("Each example needs `extra_info.intent`; see rl_training/prepare_data.py")
        self._conversation_hash = self._extra_info.get("conversation_hash")
        self.max_turns = env_config.get("max_turns", extras.get("max_turns", 5))
        self.turns = 0

        sim_cfg = env_config.user_simulator
        self._sim_model = sim_cfg.model
        self._sim_client = _get_client(sim_cfg.base_url)
        self._sim_temperature = sim_cfg.get("temperature", 0.7)
        self._sim_top_p = sim_cfg.get("top_p", 0.9)
        self._sim_max_tokens = sim_cfg.get("max_tokens", 1024)
        self._sim_max_retries = sim_cfg.get("max_retries", 8)
        self._terminal_signal = sim_cfg.get("terminal_signal", TERMINAL_SIGNAL)

        self._judges = [
            {
                "model": judge.model,
                "client": _get_client(judge.base_url),
                "temperature": judge.get("temperature", 0.0),
                "structured_output": judge.get("structured_output", True),
                "max_retries": judge.get("max_retries", 8),
            }
            for judge in env_config.llm_judges
        ]
        if not self._judges:
            raise ValueError("At least one entry in `llm_judges` is required")
        self._rubric_name = env_config.get("judge_rubric", DEFAULT_JUDGE_RUBRIC)
        if self._rubric_name not in JUDGE_RUBRICS:
            raise ValueError(f"Unknown judge_rubric {self._rubric_name!r}; choose from {list(JUDGE_RUBRICS)}")
        self._rubric = JUDGE_RUBRICS[self._rubric_name]

        # Messages seen by the user simulator. The assistant's own history lives in the generator.
        self._conversation: ConversationType = []

    def _generate_user_turn(self) -> Tuple[str, Dict[str, Any]]:
        raise NotImplementedError

    def _start_conversation(self, first_user_turn: str) -> None:
        self._conversation.append({"role": "user", "content": first_user_turn})

    def _dialogue(self) -> ConversationType:
        """The user/assistant turns, without any simulator-side system message."""
        return [m for m in self._conversation if m["role"] != "system"]

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        if len(prompt) != 1 or prompt[0]["role"] != "user":
            raise ValueError("The prompt must be the single opening user turn of the conversation")
        self.turns = 0
        self._start_conversation(prompt[0]["content"])
        # The intent is not returned, so the assistant never conditions on it.
        return [{"role": "user", "content": prompt[0]["content"]}], {}

    def _judge_once(self, judge: Dict[str, Any], message: str, structured: bool) -> Tuple[float, str]:
        kwargs = _thinking_disabled(judge["model"])
        if structured:
            kwargs["response_format"] = self._rubric.json_schema
        completion = judge["client"].chat.completions.create(
            model=judge["model"],
            messages=[{"role": "user", "content": message}],
            temperature=judge["temperature"],
            **kwargs,
        )
        response = completion.choices[0].message.content.strip()
        parsed = json.loads(response) if structured else _extract_judge_dict(response)
        return float(parsed["score"]) / self._rubric.max_score, response

    def _score_with_judge(self, judge: Dict[str, Any], message: str) -> Dict[str, Any]:
        """Score with one judge, retrying with backoff; returns reward 0.0 if every attempt fails."""
        for attempt in range(1, judge["max_retries"] + 1):
            try:
                score, response = self._judge_once(judge, message, judge["structured_output"])
                return {"model": judge["model"], "reward": score, "raw_response": response}
            except Exception:
                if judge["structured_output"]:
                    # The server may not support json_schema outputs; retry once with free-form parsing.
                    try:
                        score, response = self._judge_once(judge, message, structured=False)
                        return {"model": judge["model"], "reward": score, "raw_response": response}
                    except Exception:
                        pass
                logger.exception(f"Attempt {attempt}/{judge['max_retries']}: LLM judge ({judge['model']}) failed.")
                if attempt < judge["max_retries"]:
                    time.sleep(1.5 ** (attempt - 1))
        logger.error(f"LLM judge ({judge['model']}) failed after {judge['max_retries']} attempts; reward 0.0.")
        return {"model": judge["model"], "reward": 0.0, "raw_response": None}

    def _get_reward(self) -> Dict[str, Any]:
        """Average of the judges' scores (each in [0, 1]) for the whole conversation."""
        dialogue = self._dialogue()
        if not dialogue or dialogue[-1]["role"] != "assistant":
            raise ValueError("The judged conversation must end with an assistant turn")
        message = self._rubric.prompt.format(question=self._intent, chat_history=format_conversation(dialogue))
        judgements = [self._score_with_judge(judge, message) for judge in self._judges]
        reward = sum(j["reward"] for j in judgements) / len(judgements)
        return {
            "reward": reward,
            "judges": judgements,
            "rubric": self._rubric_name,
            "prompt": message,
            "conversation_hash": self._conversation_hash,
        }

    def step(
        self,
        action: str,
        return_judge_metadata: bool = False,
        return_user_simulator_metadata: bool = False,
    ) -> BaseTextEnvStepOutput:
        self._conversation.append({"role": "assistant", "content": action})
        self.turns += 1
        done = self.turns >= self.max_turns
        observations: ConversationType = []
        user_metadata: Dict[str, Any] = {}
        judge_metadata: Dict[str, Any] = {}
        reward = 0.0

        if not done:
            user_turn, user_metadata = self._generate_user_turn()
            done = self._terminal_signal in user_turn.strip()
            if not done:
                user_message = {"role": "user", "content": user_turn}
                self._conversation.append(user_message)
                observations = [user_message]

        if done:
            judge_metadata = self._get_reward()
            reward = judge_metadata["reward"]

        return BaseTextEnvStepOutput(
            observations=observations,
            reward=reward,
            done=done,
            metadata={
                "judge": judge_metadata if return_judge_metadata else {},
                "user_simulator": user_metadata if return_user_simulator_metadata else {},
            },
        )

    def close(self) -> None:
        # Clients are shared across envs in this process; closing one would break the others.
        pass


class SFTUserEnv(UserSimulatorEnv):
    """SFTUSER: a model fine-tuned on WildChat user turns, served by a vLLM completions endpoint.

    The intent is the system prompt, as in SFT (Appendix C.3). The first user turn is the real WildChat
    opening; later turns are sampled from the model. If the model repeats the previous user turn verbatim,
    it is resampled with the temperature raised by 0.05 per attempt, capped at 1.0 (Appendix C.4).
    """

    _tokenizers: ClassVar[Dict[str, Any]] = {}
    _tokenizers_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)
        sim_cfg = env_config.user_simulator
        self._max_dedup_retries = sim_cfg.get("max_dedup_retries", 5)
        tokenizer_path = sim_cfg.get("tokenizer") or self._sim_model
        with self._tokenizers_lock:
            if tokenizer_path not in self._tokenizers:
                self._tokenizers[tokenizer_path] = AutoTokenizer.from_pretrained(tokenizer_path)
        self._tokenizer = self._tokenizers[tokenizer_path]
        if "<|im_start|>" not in (self._tokenizer.chat_template or ""):
            raise NotImplementedError("SFTUSER prompting is implemented for ChatML (Qwen2.5) chat templates only")
        self._user_turn_prefix = "<|im_start|>user\n"
        self._stop_token = "<|im_end|>"
        self._conversation = [{"role": "system", "content": self._intent}]

    def _prompt_text(self) -> str:
        """Chat-templated history followed by an open user turn for the simulator to complete."""
        text = self._tokenizer.apply_chat_template(self._conversation, tokenize=False, add_generation_prompt=False)
        bos = self._tokenizer.bos_token or ""
        return bos + text + self._user_turn_prefix

    def _generate_user_turn(self) -> Tuple[str, Dict[str, Any]]:
        previous = next((m["content"] for m in reversed(self._conversation) if m["role"] == "user"), None)
        prompt = self._prompt_text()
        dedup_attempt, error_attempt = 0, 0
        while True:
            temperature = min(self._sim_temperature + 0.05 * dedup_attempt, 1.0)
            try:
                response = self._sim_client.completions.create(
                    model=self._sim_model,
                    prompt=prompt,
                    stop=[self._stop_token, self._terminal_signal],
                    temperature=temperature,
                    top_p=self._sim_top_p,
                    max_tokens=self._sim_max_tokens,
                    # Keep <|endconversation|> and the stop string visible, and do not re-add special
                    # tokens to a prompt that is already chat-templated.
                    extra_body={
                        "skip_special_tokens": False,
                        "include_stop_str_in_output": True,
                        "spaces_between_special_tokens": False,
                        "add_special_tokens": False,
                    },
                )
            except Exception as e:
                error_attempt += 1
                logger.exception(f"Attempt {error_attempt}/{self._sim_max_retries}: user simulator request failed.")
                if error_attempt >= self._sim_max_retries:
                    logger.error("User simulator failed; ending the conversation.")
                    return self._terminal_signal, {"error": str(e), "conversation_hash": self._conversation_hash}
                time.sleep(1.3 ** (error_attempt - 1))
                continue

            raw = response.choices[0].text
            user_turn = raw.strip().removesuffix(self._stop_token)
            if previous is not None and user_turn.strip() == previous.strip() and dedup_attempt < self._max_dedup_retries:
                dedup_attempt += 1
                continue
            return user_turn, {
                "conversation_hash": self._conversation_hash,
                "input": prompt,
                "raw_response": raw,
                "model": self._sim_model,
                "temperature": temperature,
                "dedup_retries": dedup_attempt,
            }


class RPUserEnv(UserSimulatorEnv):
    """RPUSER1/RPUSER2: an instruction-tuned LLM prompted to role-play the user (Appendices D.1-D.2).

    With `generate_first_turn: true` (RPUSER1) the simulator also writes the opening turn from the intent,
    replacing the real WildChat opening; otherwise (RPUSER2) the real opening is kept.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)
        self._generate_first_turn = env_config.get("generate_first_turn", False)
        self._structured_output = env_config.user_simulator.get("structured_output", True)
        self._persona = "default"

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        if not self._generate_first_turn:
            return super().init(prompt)
        self.turns = 0
        first_turn, metadata = self._generate_user_turn()
        self._start_conversation(first_turn)
        return [{"role": "user", "content": first_turn}], {"user_simulator": metadata}

    def _generate_user_turn(self) -> Tuple[str, Dict[str, Any]]:
        # The chat history is left empty when the simulator writes the opening turn.
        prompt = RPUSER_PROMPTS[self._persona].format(
            task_desc=RPUSER_TASK_DESCRIPTION,
            single_turn_prompt=self._intent,
            chat_history=format_conversation(self._dialogue()),
            terminal_signal=self._terminal_signal,
        )
        kwargs = _thinking_disabled(self._sim_model)
        if self._structured_output:
            kwargs["response_format"] = RPUSER_JSON_SCHEMA
        for attempt in range(1, self._sim_max_retries + 1):
            try:
                response = self._sim_client.chat.completions.create(
                    model=self._sim_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._sim_temperature,
                    top_p=self._sim_top_p,
                    max_tokens=self._sim_max_tokens,
                    **kwargs,
                )
                text = response.choices[0].message.content.strip()
                parsed = json.loads(text) if self._structured_output else extract_outer_dict(text)
                return parsed.get("response", ""), {
                    "conversation_hash": self._conversation_hash,
                    "persona": self._persona,
                    "input": prompt,
                    "raw_response": text,
                    "model": self._sim_model,
                    "temperature": self._sim_temperature,
                }
            except Exception as e:
                logger.exception(f"Attempt {attempt}/{self._sim_max_retries}: user simulator request failed.")
                if attempt < self._sim_max_retries:
                    time.sleep(1.5 ** (attempt - 1))
                else:
                    logger.error("User simulator failed; ending the conversation.")
                    return self._terminal_signal, {"error": str(e), "conversation_hash": self._conversation_hash}


class RPUserPersonaEnv(RPUserEnv):
    """RPUSER3: RPUSER2 with a persona guideline drawn uniformly per conversation (Appendix D.3).

    The persona is drawn from `extra_info.persona_seed`, which is fixed per dataset example, so all rollouts
    in a GRPO group face the same persona and advantages are not confounded by persona difficulty.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__(env_config, extras)
        persona_seed = self._extra_info.get("persona_seed")
        if persona_seed is None:
            raise ValueError("rpuser_persona needs `extra_info.persona_seed`; see rl_training/prepare_data.py")
        self._persona = random.Random(persona_seed).choice(list(RPUSER_PROMPTS))
