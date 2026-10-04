"""
WildBench Binary Checklist Majority Vote
=========================================
Combine the binary-checklist results of three judges, produced by
`wildbench_absolute_binary_checking.py`, by majority vote on each checklist item.

Outputs use the judge name `majority`, so pairwise win rates can be computed with
`wildbench_absolute_binary_checking_then_pairwise.py --judge_model majority`:

    <input_dir>/<model>_majority_details.jsonl
    <input_dir>/<model>_majority_summary.json

Usage:
    python wildbench_absolute_binary_checking_majority.py --model my-model-A
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from wildbench_absolute_binary_checking_then_pairwise import _load_details, _safe_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_JUDGES = ["gpt-5-mini", "gemini-2.5-flash", "claude-haiku-4-5-20251001"]


def _verdicts(rec: Dict, n_items: int) -> Optional[Dict[int, bool]]:
    results = rec.get("checklist_results")
    if rec.get("judge_failed") or results is None:
        return None
    verdicts: Dict[int, bool] = {}
    for pos, entry in enumerate(results):
        item = entry.get("item")
        idx = item if isinstance(item, int) and 1 <= item <= n_items else pos + 1
        verdicts[idx] = bool(entry["satisfied"])
    if set(verdicts) != set(range(1, n_items + 1)):
        return None
    return verdicts


def majority_vote(model: str, judges: List[str], input_dir: Path) -> Dict:
    details = [
        _load_details(input_dir / f"{_safe_name(model)}_{_safe_name(judge)}_details.jsonl")
        for judge in judges
    ]
    sessions = set(details[0]).intersection(*details[1:])
    missing = set().union(*details) - sessions
    if missing:
        logger.warning("%d sessions are not judged by every judge; ignoring.", len(missing))

    results: List[Dict] = []
    for sid in sorted(sessions):
        recs = [d[sid] for d in details]
        base = recs[0]
        n_items = len(base["checklist"])
        votes = [_verdicts(rec, n_items) for rec in recs]
        failed = any(v is None for v in votes)
        checklist_results = None if failed else [
            {"item": i, "satisfied": 2 * sum(v[i] for v in votes) > len(votes)}
            for i in range(1, n_items + 1)
        ]
        results.append({
            "session_id": sid,
            "primary_tag": base.get("primary_tag"),
            "checklist": base["checklist"],
            "model_output": base.get("model_output"),
            "judge_failed": failed,
            "checklist_results": checklist_results,
            "satisfaction_rate": None if failed else sum(c["satisfied"] for c in checklist_results) / n_items,
        })

    valid = [r for r in results if not r["judge_failed"]]
    total_items = sum(len(r["checklist_results"]) for r in valid)
    total_satisfied = sum(sum(c["satisfied"] for c in r["checklist_results"]) for r in valid)
    category_rates: Dict[str, List[float]] = {}
    for r in valid:
        category_rates.setdefault(r["primary_tag"] or "unknown", []).append(r["satisfaction_rate"])

    summary = {
        "model": model,
        "n_tasks": len(valid),
        "avg_satisfaction_rate": round(sum(r["satisfaction_rate"] for r in valid) / len(valid) * 100, 2) if valid else 0.0,
        "global_satisfaction_rate": round(total_satisfied / total_items * 100, 2) if total_items else 0.0,
        "total_checklist_items": total_items,
        "total_satisfied": total_satisfied,
        "judge_failures": len(results) - len(valid),
        "judge_model": "majority",
        "judges": judges,
        "category_satisfaction_rates": {
            tag: round(sum(s) / len(s) * 100, 2) for tag, s in category_rates.items()
        },
    }

    prefix = f"{_safe_name(model)}_majority"
    with open(input_dir / f"{prefix}_details.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    with open(input_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Majority-vote results saved to %s", input_dir)
    _print_summary(summary)
    return summary


def _print_summary(s: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"Binary Checklist Majority Vote: {s['model']}")
    print("=" * 60)
    print(f"  Judges                   : {', '.join(s['judges'])}")
    print(f"  Tasks evaluated          : {s['n_tasks']}")
    print(f"  Avg satisfaction rate     : {s['avg_satisfaction_rate']:.1f}%  (per-task avg)")
    print(f"  Global satisfaction rate  : {s['global_satisfaction_rate']:.1f}%  ({s['total_satisfied']}/{s['total_checklist_items']} items)")
    print(f"  Judge failures           : {s['judge_failures']}")
    cat = s.get("category_satisfaction_rates", {})
    if cat:
        print("  --- Per-category satisfaction rate ---")
        for tag, avg in sorted(cat.items(), key=lambda x: -x[1]):
            print(f"    {tag:30s}: {avg:.1f}%")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Majority vote over three binary-checklist judges.")
    parser.add_argument("--model", required=True,
                        help="Model identifier used during binary-checklist evaluation.")
    parser.add_argument("--judges", nargs=3, default=DEFAULT_JUDGES,
                        help="The three judge model identifiers to vote over.")
    parser.add_argument("--input_dir",
                        default="results/absolute_binary_checklist/post_generation_judge",
                        help="Directory containing <model>_<judge>_details.jsonl files; "
                             "the majority-vote files are written here as well.")
    args = parser.parse_args()

    majority_vote(model=args.model, judges=args.judges, input_dir=Path(args.input_dir))


if __name__ == "__main__":
    main()
