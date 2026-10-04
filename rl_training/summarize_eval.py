"""Average test reward with a 95% bootstrap confidence interval, on the 0-100 scale of Table 3.

Reads the per-conversation dumps that run_eval.sh writes to $OUTPUT_DIR/dumped_evals/eval_only/.

    python summarize_eval.py /path/to/eval-run/dumped_evals/eval_only [more runs ...]
"""

import argparse
import json
import random
from pathlib import Path


def conversation_score(entry):
    # Multi-turn rollouts store per-token rewards; the judge reward sits on the final assistant turn.
    score = entry["score"]
    return sum(score) if isinstance(score, list) else float(score)


def dump_files(path: Path):
    if path.is_file():
        return [path]
    return sorted(p for p in path.glob("*.jsonl") if p.name != "aggregated_results.jsonl")


def bootstrap_ci(scores, n_bootstrap, rng):
    n = len(scores)
    means = sorted(sum(rng.choices(scores, k=n)) / n for _ in range(n_bootstrap))
    return means[int(0.025 * n_bootstrap)], means[int(0.975 * n_bootstrap) - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", type=Path, nargs="+", help="dumped_evals/eval_only directories or .jsonl files")
    parser.add_argument("--n_bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for path in args.paths:
        for dump in dump_files(path):
            with open(dump) as f:
                scores = [100 * conversation_score(json.loads(line)) for line in f if line.strip()]
            mean = sum(scores) / len(scores)
            low, high = bootstrap_ci(scores, args.n_bootstrap, rng)
            print(f"{dump}: {mean:.1f} ± {(high - low) / 2:.1f} (95% CI [{low:.1f}, {high:.1f}], n={len(scores)})")


if __name__ == "__main__":
    main()
