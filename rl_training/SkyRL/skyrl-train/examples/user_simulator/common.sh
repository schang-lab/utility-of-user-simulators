# Settings shared by run_train.sh and run_eval.sh (sourced, not executed).
# Every variable below can be overridden from the environment.

# Which user simulator the assistant talks to (Section 3).
#   sftuser  learned simulator, real opening turn          -> env `sftuser`
#   rpuser1  role-playing LLM, generated opening turn      -> env `rpuser` (generate_first_turn=true)
#   rpuser2  role-playing LLM, real opening turn           -> env `rpuser`
#   rpuser3  rpuser2 + persona sampled per conversation    -> env `rpuser_persona`
: "${SIMULATOR:=sftuser}"
case "$SIMULATOR" in
  sftuser)         ENV_CLASS=sftuser;        DEFAULT_USER_MODEL="jjssuh/sftuser-base-Qwen2.5-14B-Instruct" ;;
  rpuser1|rpuser2) ENV_CLASS=rpuser;         DEFAULT_USER_MODEL="Qwen/Qwen2.5-14B-Instruct" ;;
  rpuser3)         ENV_CLASS=rpuser_persona; DEFAULT_USER_MODEL="Qwen/Qwen2.5-14B-Instruct" ;;
  *) echo "Unknown SIMULATOR=$SIMULATOR (expected sftuser, rpuser1, rpuser2, or rpuser3)" >&2; exit 1 ;;
esac

: "${DATA_DIR:=$HOME/data/user_simulator_rl}"   # output of rl_training/prepare_data.py
: "${AGENT_MODEL:=Qwen/Qwen2.5-3B-Instruct}"     # assistant (policy) to train or evaluate
: "${NUM_GPUS:=2}"                               # GPUs for the assistant; simulator and judges are served separately
: "${INFERENCE_BACKEND:=vllm}"
: "${LOGGER:=wandb}"                             # or console, tensorboard

# OpenAI-compatible servers for the frozen models. Model names must match what each server exposes.
: "${USER_MODEL:=$DEFAULT_USER_MODEL}"
: "${USER_TOKENIZER:=$USER_MODEL}"               # sftuser only; needs the <|endconversation|> token
: "${USER_URL:=http://localhost:8002/v1}"
: "${JUDGE1_MODEL:=mistralai/Mistral-Small-3.1-24B-Instruct-2503}"
: "${JUDGE1_URL:=http://localhost:8003/v1}"
: "${JUDGE2_MODEL:=Qwen/Qwen3-32B}"
: "${JUDGE2_URL:=http://localhost:8004/v1}"
: "${JUDGE_RUBRIC:=paper}"                       # Appendix E; alternatives in skyrl_gym/envs/user_simulator/judge_rubrics.py

# The skyrl_train entrypoints default to WARNING, which hides console metrics.
export LOG_LEVEL="${LOG_LEVEL:-INFO}"

ENV_PREFIX="environment.skyrl_gym.${ENV_CLASS}"
ENV_OVERRIDES=(
  "environment.env_class=${ENV_CLASS}"
  "environment.skyrl_gym.max_env_workers=48"
  "${ENV_PREFIX}.max_turns=5"
  "${ENV_PREFIX}.user_simulator.model='${USER_MODEL}'"
  "${ENV_PREFIX}.user_simulator.base_url='${USER_URL}'"
  # Two judges at temperature 0 with JSON-schema decoding (the environment's defaults); the reward is their mean
  # score normalized to [0, 1] (score / 10 under the Appendix E rubric).
  "${ENV_PREFIX}.llm_judges=[{model:'${JUDGE1_MODEL}',base_url:'${JUDGE1_URL}'},{model:'${JUDGE2_MODEL}',base_url:'${JUDGE2_URL}'}]"
  "${ENV_PREFIX}.judge_rubric=${JUDGE_RUBRIC}"
)
if [ "$ENV_CLASS" = sftuser ]; then
  ENV_OVERRIDES+=("${ENV_PREFIX}.user_simulator.tokenizer='${USER_TOKENIZER}'")
else
  ENV_OVERRIDES+=("${ENV_PREFIX}.generate_first_turn=$([ "$SIMULATOR" = rpuser1 ] && echo true || echo false)")
fi

# Rollout settings shared by training and evaluation (Appendix F.2): at most five assistant turns,
# 2,048 assistant tokens per turn at temperature 1, identical sampling at evaluation time.
GENERATOR_OVERRIDES=(
  "trainer.policy.model.path=${AGENT_MODEL}"
  "trainer.max_prompt_length=2048"
  "trainer.eval_batch_size=1024"
  "trainer.logger=${LOGGER}"
  "generator.backend=${INFERENCE_BACKEND}"
  "generator.num_inference_engines=${NUM_GPUS}"
  "generator.inference_engine_tensor_parallel_size=1"
  "generator.gpu_memory_utilization=0.8"
  "generator.run_engines_locally=true"
  "generator.weight_sync_backend=nccl"
  "generator.async_engine=true"
  "generator.batched=false"
  "generator.use_conversation_multi_turn=true"
  "generator.max_turns=5"
  "generator.max_input_length=16384"
  "generator.sampling_params.temperature=1.0"
  "generator.sampling_params.max_generate_length=2048"
  "generator.eval_sampling_params.temperature=1.0"
  "generator.eval_sampling_params.max_generate_length=2048"
  "generator.eval_n_samples_per_prompt=1"
)
