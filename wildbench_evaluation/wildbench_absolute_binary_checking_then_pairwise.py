"""
WildBench Binary-Checklist Pairwise Win-Rate
=============================================
Compute a pairwise win rate by comparing two models that have already been
evaluated by `wildbench_absolute_binary_checking.py`.

For each session_id present in BOTH detail files, the model whose response
satisfied a strictly larger number of checklist items wins. Equal counts are
ties.

Usage:
    python wildbench_absolute_binary_checking_then_pairwise.py \\
        --model1 my-model-A --model2 my-model-B \\
        --judge_model gpt-4o
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    return name.replace("/", "_")


def _load_details(path: Path) -> Dict[str, Dict]:
    """Load a binary-checklist details JSONL file, keyed by session_id."""
    if not path.exists():
        raise FileNotFoundError(f"Details file not found: {path}")
    out: Dict[str, Dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["session_id"]] = rec
    return out


def _count_satisfied(rec: Dict) -> Optional[int]:
    cr = rec.get("checklist_results")
    if cr is None:
        return None
    return sum(1 for c in cr if c.get("satisfied"))


def compute_pairwise(
    model1: str,
    model2: str,
    judge_model: str,
    input_dir: Path,
    output_dir: Path,
) -> Dict:
    safe1 = _safe_name(model1)
    safe2 = _safe_name(model2)
    safe_judge = _safe_name(judge_model)

    path1 = input_dir / f"{safe1}_{safe_judge}_details.jsonl"
    path2 = input_dir / f"{safe2}_{safe_judge}_details.jsonl"

    detail1 = _load_details(path1)
    detail2 = _load_details(path2)

    common = sorted(set(detail1) & set(detail2))
    if not common:
        raise ValueError(f"No common session_ids between {path1} and {path2}.")

    only1 = set(detail1) - set(detail2)
    only2 = set(detail2) - set(detail1)
    if only1:
        logger.warning("%d sessions only in %s; ignoring.", len(only1), model1)
    if only2:
        logger.warning("%d sessions only in %s; ignoring.", len(only2), model2)

    per_task: List[Dict] = []
    wins = ties = losses = failed = 0
    cat_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"wins": 0, "ties": 0, "losses": 0})

    for sid in common:
        r1 = detail1[sid]
        r2 = detail2[sid]
        n1 = _count_satisfied(r1)
        n2 = _count_satisfied(r2)
        tag = r1.get("primary_tag") or r2.get("primary_tag") or "unknown"

        if n1 is None or n2 is None:
            failed += 1
            outcome = "failed"
        elif n1 > n2:
            wins += 1
            outcome = "win"
            cat_stats[tag]["wins"] += 1
        elif n1 < n2:
            losses += 1
            outcome = "loss"
            cat_stats[tag]["losses"] += 1
        else:
            ties += 1
            outcome = "tie"
            cat_stats[tag]["ties"] += 1

        per_task.append({
            "session_id": sid,
            "primary_tag": tag,
            "model1_n_satisfied": n1,
            "model2_n_satisfied": n2,
            "outcome": outcome,
        })

    n_eval = wins + ties + losses
    win_rate = wins / n_eval if n_eval else 0.0
    tie_rate = ties / n_eval if n_eval else 0.0
    loss_rate = losses / n_eval if n_eval else 0.0
    adj_win_rate = (wins + 0.5 * ties) / n_eval if n_eval else 0.0

    category_summary = {
        tag: {
            **counts,
            "n": counts["wins"] + counts["ties"] + counts["losses"],
            "win_rate": round(
                counts["wins"] / (counts["wins"] + counts["ties"] + counts["losses"]) * 100, 2
            ) if (counts["wins"] + counts["ties"] + counts["losses"]) else 0.0,
        }
        for tag, counts in cat_stats.items()
    }

    summary = {
        "model1": model1,
        "model2": model2,
        "judge_model": judge_model,
        "n_common_sessions": len(common),
        "n_tasks_evaluated": n_eval,
        "n_failed": failed,
        "wins_model1": wins,
        "ties": ties,
        "losses_model1": losses,
        "win_rate_model1": round(win_rate * 100, 2),
        "tie_rate": round(tie_rate * 100, 2),
        "loss_rate_model1": round(loss_rate * 100, 2),
        "adjusted_win_rate_model1": round(adj_win_rate * 100, 2),
        "category_stats": category_summary,
        "input_files": {"model1": str(path1), "model2": str(path2)},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = f"{safe1}_vs_{safe2}_{safe_judge}_binary_pairwise"
    summary_path = output_dir / f"{out_prefix}_summary.json"
    detail_path = output_dir / f"{out_prefix}_details.jsonl"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    with open(detail_path, "w") as f:
        for r in per_task:
            f.write(json.dumps(r) + "\n")
    logger.info("Pairwise results saved to %s", output_dir)

    _print_summary(summary)
    return summary


def _print_summary(s: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"Binary-Checklist Pairwise: {s['model1']}  vs  {s['model2']}")
    print(f"Judge model              : {s['judge_model']}")
    print("=" * 60)
    print(f"  Common sessions          : {s['n_common_sessions']}")
    print(f"  Tasks evaluated          : {s['n_tasks_evaluated']}")
    print(f"  Failed (any side)        : {s['n_failed']}")
    print(f"  Wins / Ties / Losses (m1): {s['wins_model1']} / {s['ties']} / {s['losses_model1']}")
    print(f"  Win rate (m1)            : {s['win_rate_model1']:.1f}%")
    print(f"  Tie rate                 : {s['tie_rate']:.1f}%")
    print(f"  Adjusted win rate (m1)   : {s['adjusted_win_rate_model1']:.1f}%  (ties = 0.5)")
    cat = s.get("category_stats", {})
    if cat:
        print(f"  --- Per-category win rate (m1) ---")
        for tag, info in sorted(cat.items(), key=lambda x: -x[1]["win_rate"]):
            print(f"    {tag:30s}: {info['win_rate']:.1f}%  "
                  f"(W/T/L = {info['wins']}/{info['ties']}/{info['losses']})")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute pairwise win rate from two binary-checklist judge result files."
    )
    parser.add_argument("--model1", required=True,
                        help="Identifier of the first model (matches the prefix used "
                             "during binary-checklist evaluation).")
    parser.add_argument("--model2", required=True,
                        help="Identifier of the second model.")
    parser.add_argument("--judge_model", required=True,
                        help="Judge model identifier used during binary-checklist evaluation.")
    parser.add_argument("--input_dir",
                        default="results/absolute_binary_checklist/post_generation_judge",
                        help="Directory containing <model>_<judge>_details.jsonl files.")
    parser.add_argument("--output_dir",
                        default="results/absolute_binary_checklist/binary_pairwise",
                        help="Directory for pairwise summary/details output.")
    args = parser.parse_args()

    compute_pairwise(
        model1=args.model1,
        model2=args.model2,
        judge_model=args.judge_model,
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
