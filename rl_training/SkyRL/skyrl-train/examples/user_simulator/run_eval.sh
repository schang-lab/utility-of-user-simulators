#!/bin/bash
# Cross-simulator evaluation (Section 4.4): pair an assistant with a user simulator on the held-out
# test split and score each conversation with the training-time judges.
#
# Run from rl_training/SkyRL/skyrl-train after starting the simulator and judge servers:
#   SIMULATOR=sftuser AGENT_MODEL=/path/to/exports/global_step_N/policy \
#     bash examples/user_simulator/run_eval.sh [extra Hydra overrides]
# Mean reward is printed at the end; per-conversation scores are written under
# $OUTPUT_DIR/dumped_evals/eval_only/ (summarize with rl_training/summarize_eval.py).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${EVAL_SPLIT:=test}"
: "${RUN_NAME:=eval--${SIMULATOR}--$(basename "$AGENT_MODEL")}"
: "${OUTPUT_DIR:=$HOME/ckpts/user_simulator_rl/$RUN_NAME}"

uv run --isolated --extra "$INFERENCE_BACKEND" -m skyrl_train.entrypoints.main_generate \
  "data.val_data=['${DATA_DIR}/${EVAL_SPLIT}.parquet']" \
  trainer.placement.colocate_all=false \
  trainer.dump_eval_results=true \
  trainer.export_path="$OUTPUT_DIR" \
  trainer.project_name=user-simulator-rl-eval \
  trainer.run_name="$RUN_NAME" \
  "${GENERATOR_OVERRIDES[@]}" \
  "${ENV_OVERRIDES[@]}" \
  "$@"
