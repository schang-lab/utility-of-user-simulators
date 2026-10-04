#!/bin/bash
# GRPO training of an assistant against one user simulator (Section 4.1, Appendix F.2).
#
# Run from rl_training/SkyRL/skyrl-train after starting the simulator and judge servers:
#   SIMULATOR=sftuser bash examples/user_simulator/run_train.sh [extra Hydra overrides]
# See common.sh for the variables shared with run_eval.sh. Extra Hydra overrides are applied last.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

: "${RUN_NAME:=${SIMULATOR}--$(basename "$AGENT_MODEL")}"
: "${OUTPUT_DIR:=$HOME/ckpts/user_simulator_rl/$RUN_NAME}"
: "${EVAL_BEFORE_TRAIN:=true}"
: "${MAX_CKPTS_TO_KEEP:=2}"   # resumable FSDP checkpoints; HF exports under $OUTPUT_DIR/exports are all kept

uv run --isolated --extra "$INFERENCE_BACKEND" -m skyrl_train.entrypoints.main_base \
  "data.train_data=['${DATA_DIR}/train.parquet']" \
  "data.val_data=['${DATA_DIR}/sampled_val.parquet']" \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.strategy=fsdp2 \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.critic_num_gpus_per_node="$NUM_GPUS" \
  trainer.placement.ref_num_gpus_per_node="$NUM_GPUS" \
  trainer.epochs=1 \
  trainer.train_batch_size=64 \
  generator.n_samples_per_prompt=5 \
  trainer.policy_mini_batch_size=16 \
  trainer.update_epochs_per_batch=1 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.policy.optimizer_config.lr=8.0e-7 \
  trainer.policy.optimizer_config.scheduler=constant_with_warmup \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_estimator_type=k3 \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.2 \
  trainer.eval_before_train="$EVAL_BEFORE_TRAIN" \
  trainer.eval_interval=20 \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=20 \
  trainer.max_ckpts_to_keep="$MAX_CKPTS_TO_KEEP" \
  trainer.resume_mode=latest \
  trainer.ckpt_path="${OUTPUT_DIR}/ckpts" \
  trainer.export_path="${OUTPUT_DIR}/exports" \
  trainer.project_name=user-simulator-rl \
  trainer.run_name="$RUN_NAME" \
  "${GENERATOR_OVERRIDES[@]}" \
  "${ENV_OVERRIDES[@]}" \
  "$@"
