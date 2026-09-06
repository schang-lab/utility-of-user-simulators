"""
Flag low-quality human-study submissions from the pairwise study.

Reads each participant JSON from `--input-dir` (one file per participant, written
by `app.py` / `_save_participant`), asks a judge LLM (default gpt-5-mini) to
score whether the participant followed the instructions, and writes a summary
JSON listing flagged participants and per-turn issues.

Per-participant data the judge sees (see `app.py` and an example file for the
exact schema):
  - writing_setup: doc_type, intent {id,title,brief}, prewriting Q&A
  - real-session conversation: list of turns, each with user_query + rationale
  - final_document

The training session (is_training=true, 1-turn warmup) is excluded by default.

Usage:
  python flag_low_quality.py \
      --input-dir results/<batch> \
      --output    results/<batch>_flags.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI


JUDGE_SYSTEM_PROMPT = """\
You are screening a human study for BAD-FAITH participants — people who did
not actually try to do the task. You are NOT a writing-quality reviewer. Most
crowdworkers will produce imperfect prose, terse rationales, or final
documents that don't precisely match every constraint in the brief. Those
participants are FINE — flag them as `ok`. Default to `ok`.

Participants in this study were assigned a writing topic and asked to chat
with two AI assistants for ~5 turns, pick a preferred reply each turn with a
short rationale, and submit a ~200-word final document on the assigned topic.

Only flag a submission as `low_quality` when there is clear, unambiguous
evidence the participant did not engage in good faith. Concretely:

  - Almost ALL user queries are clearly off-topic — random trivia, jokes,
    personal chit-chat, prompt-injection attempts, or content unrelated to
    the assigned doc_type/intent. (One or two tangential queries inside an
    otherwise on-topic conversation is NOT enough.)
  - The user queries are obvious junk: gibberish, keyboard mashing,
    "test test test", lorem ipsum, repeated identical text, or pure verbatim
    pastes of the prewriting answers with no other engagement.
  - The pairwise rationales are uniformly meaningless across essentially all
    turns — e.g. every rationale is empty, a single character, "ok", "good",
    or copy-pasted text. A few terse rationales mixed with substantive ones
    is FINE.
  - The final document is clearly unrelated to the assigned intent (totally
    different topic), is boilerplate / lorem ipsum, or is so short or
    incoherent that it is obviously not a genuine attempt.

Use `borderline` ONLY when there is real evidence of disengagement on a
significant fraction of turns but it isn't overwhelming — e.g. ~half the
queries are off-topic, or ~half the rationales are non-substantive AND the
final document is also weak. If you find yourself reaching for `borderline`
because the writing is "imperfect" or "doesn't fully meet the brief",
that is `ok`, not `borderline`.

Things that DO NOT justify a `low_quality` or `borderline` flag:
  - Imperfect grammar, spelling, or non-native English.
  - Final document that misses some details from the intent brief, has the
    wrong word count, wrong tone, wrong formatting, or extraneous headings.
  - A few short or generic rationales mixed in with substantive ones.
  - Writing that is bland, repetitive, or just not very good — quality of
    output is irrelevant; we only care about effort/engagement.
  - Disagreement between the participant's stated preference and what you
    would have picked.

You MUST respond with a single JSON object (no surrounding text, no code
fences) with EXACTLY these keys:

{
  "flag": "ok" | "borderline" | "low_quality",
  "overall_quality_score": <integer 1..5, where 5 = clearly engaged in good faith, 3 = some effort issues, 1 = clearly bad-faith>,
  "off_topic_query_turns": <list of integer turn indices where the user_query is clearly off-topic>,
  "low_effort_query_turns": <list of integer turn indices where the user_query is gibberish / one-word / clearly junk>,
  "poor_rationale_turns":  <list of integer turn indices where the rationale is empty / a single character / pure copy-paste>,
  "final_document_issue": <string — only describe disengagement issues (off-topic, lorem ipsum, gibberish, far too short). Leave empty if final doc is a genuine attempt at the topic, even if imperfect.>,
  "reason": <string — 1-3 sentence justification for the flag>
}

Turn indices refer to the `turn_idx` values shown in the submission.
"""


def _build_user_message(participant: Dict) -> Optional[str]:
    """Return the prompt body shown to the judge, or None if there is no real
    session to evaluate."""
    setup = participant.get("writing_setup") or {}
    intent = setup.get("intent") or {}
    questions = setup.get("prewriting_questions") or []
    answers = setup.get("prewriting_answers") or []

    real_convos = [c for c in participant.get("conversations", []) if not c.get("is_training")]
    if not real_convos:
        return None
    convo = real_convos[0]
    turns = convo.get("turns", [])
    final_document = convo.get("final_document", "") or ""

    qa_lines = []
    for i, q in enumerate(questions):
        a = answers[i] if i < len(answers) else ""
        qa_lines.append(f"  Q{i+1}: {q}\n  A{i+1}: {a}")
    qa_block = "\n".join(qa_lines) if qa_lines else "(none)"

    turn_lines = []
    for t in turns:
        idx = t.get("turn_idx")
        uq = t.get("user_query", "") or ""
        rat = t.get("rationale", "") or ""
        pref = t.get("preference_ab", t.get("preference", ""))
        turn_lines.append(
            f"--- Turn {idx} ---\n"
            f"  user_query: {uq}\n"
            f"  preference: {pref}\n"
            f"  rationale:  {rat}"
        )
    turns_block = "\n".join(turn_lines) if turn_lines else "(no turns)"

    return (
        f"## Assigned writing task\n"
        f"  doc_type: {setup.get('doc_type', '')}\n"
        f"  intent_title: {intent.get('title', '')}\n"
        f"  intent_brief: {intent.get('brief', '')}\n\n"
        f"## Pre-writing questions and the participant's answers\n"
        f"{qa_block}\n\n"
        f"## Real-session conversation (the participant's queries to the two AIs and their pairwise rationales)\n"
        f"{turns_block}\n\n"
        f"## Final document submitted by the participant\n"
        f"{final_document}\n"
    )


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json(raw: str) -> Optional[Dict]:
    if not raw:
        return None
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        raw = re.sub(r"\n?```\s*$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _call_judge(
    client: OpenAI,
    model: str,
    user_message: str,
    max_completion_tokens: int,
    max_retries: int,
) -> Dict:
    is_gpt = model.lower().startswith("gpt")
    is_gpt5 = model.lower().startswith("gpt-5")

    extra_kwargs: Dict = {}
    if is_gpt:
        extra_kwargs["max_completion_tokens"] = max_completion_tokens
        if is_gpt5:
            extra_kwargs["reasoning_effort"] = "minimal"
    else:
        extra_kwargs["max_tokens"] = max_completion_tokens

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    last_raw = ""
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=1 if is_gpt5 else 0.0,
                **extra_kwargs,
            )
            last_raw = resp.choices[0].message.content or ""
            parsed = _extract_json(last_raw)
            if parsed is not None and "flag" in parsed:
                parsed["_raw"] = last_raw
                return parsed
            last_err = "could not parse JSON from judge output"
        except Exception as e:
            last_err = repr(e)
            time.sleep(min(2 ** attempt, 10))

    return {
        "flag": "judge_failed",
        "overall_quality_score": None,
        "off_topic_query_turns": [],
        "low_effort_query_turns": [],
        "poor_rationale_turns": [],
        "final_document_issue": "",
        "reason": f"judge call failed: {last_err}",
        "_raw": last_raw,
    }


def _process_one(
    path: Path,
    client: OpenAI,
    model: str,
    max_completion_tokens: int,
    max_retries: int,
) -> Dict:
    try:
        participant = json.loads(path.read_text())
    except Exception as e:
        return {
            "file": path.name,
            "participant_id": None,
            "prolific_id": None,
            "skipped": True,
            "skip_reason": f"failed to read JSON: {e}",
        }

    pid = participant.get("participant_id")
    prolific_id = participant.get("prolific_id")

    if not participant.get("consent_given"):
        return {
            "file": path.name,
            "participant_id": pid,
            "prolific_id": prolific_id,
            "skipped": True,
            "skip_reason": "no consent given",
        }

    user_message = _build_user_message(participant)
    if user_message is None:
        return {
            "file": path.name,
            "participant_id": pid,
            "prolific_id": prolific_id,
            "skipped": True,
            "skip_reason": "no real (non-training) session present",
        }

    judge_out = _call_judge(client, model, user_message, max_completion_tokens, max_retries)

    real_convos = [c for c in participant.get("conversations", []) if not c.get("is_training")]
    convo = real_convos[0]
    final_doc = convo.get("final_document", "") or ""

    return {
        "file": path.name,
        "participant_id": pid,
        "prolific_id": prolific_id,
        "completed": bool(participant.get("completed")),
        "doc_type": (participant.get("writing_setup") or {}).get("doc_type"),
        "intent_id": ((participant.get("writing_setup") or {}).get("intent") or {}).get("id"),
        "n_real_turns": len(convo.get("turns", [])),
        "final_doc_word_count": len(final_doc.split()),
        "judge_model": model,
        "judge": judge_out,
        "skipped": False,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory of per-participant JSONs written by app.py (e.g. results/).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <input-dir>_flags.json next to the input dir).",
    )
    p.add_argument("--model", default="gpt-5-mini", help="Judge model (default: gpt-5-mini).")
    p.add_argument("--max-workers", type=int, default=16, help="Parallel judge calls (default: 16).")
    p.add_argument("--max-completion-tokens", type=int, default=4096)
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N participant files (debugging).",
    )
    args = p.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"[error] input dir does not exist: {input_dir}", file=sys.stderr)
        return 2

    output_path = args.output
    if output_path is None:
        output_path = input_dir.parent / f"{input_dir.name}_flags.json"
    else:
        output_path = output_path.resolve()

    files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix == ".json" and not f.name.startswith("_")
    )
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        print(f"[error] no participant JSON files found in {input_dir}", file=sys.stderr)
        return 2

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[error] OPENAI_API_KEY env var is not set.", file=sys.stderr)
        return 2
    client = OpenAI(api_key=api_key)

    print(f"[info] judging {len(files)} participant file(s) from {input_dir}")
    print(f"[info] judge model: {args.model}, parallel workers: {args.max_workers}")

    results: List[Dict] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futs = {
            pool.submit(_process_one, f, client, args.model, args.max_completion_tokens, args.max_retries): f
            for f in files
        }
        for i, fut in enumerate(as_completed(futs), 1):
            f = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {
                    "file": f.name,
                    "skipped": True,
                    "skip_reason": f"unhandled error: {e!r}",
                }
            results.append(r)
            if r.get("skipped"):
                print(f"[{i:3d}/{len(files)}] {f.name}: SKIPPED — {r.get('skip_reason')}")
            else:
                flag = (r.get("judge") or {}).get("flag")
                score = (r.get("judge") or {}).get("overall_quality_score")
                print(f"[{i:3d}/{len(files)}] {f.name}: flag={flag} score={score}")

    results.sort(key=lambda r: r.get("file", ""))

    counts: Dict[str, int] = {}
    for r in results:
        if r.get("skipped"):
            counts["skipped"] = counts.get("skipped", 0) + 1
            continue
        flag = (r.get("judge") or {}).get("flag", "unknown")
        counts[flag] = counts.get(flag, 0) + 1

    flagged = [
        {
            "file": r["file"],
            "participant_id": r.get("participant_id"),
            "prolific_id": r.get("prolific_id"),
            "flag": (r.get("judge") or {}).get("flag"),
            "overall_quality_score": (r.get("judge") or {}).get("overall_quality_score"),
            "reason": (r.get("judge") or {}).get("reason"),
        }
        for r in results
        if not r.get("skipped")
        and (r.get("judge") or {}).get("flag") in ("low_quality", "borderline", "judge_failed")
    ]

    summary = {
        "input_dir": str(input_dir),
        "judge_model": args.model,
        "n_files": len(files),
        "counts": counts,
        "flagged": flagged,
        "results": results,
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print()
    print(f"[done] wrote {output_path}")
    print(f"[done] counts: {counts}")
    print(f"[done] flagged (low_quality + borderline + judge_failed): {len(flagged)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
