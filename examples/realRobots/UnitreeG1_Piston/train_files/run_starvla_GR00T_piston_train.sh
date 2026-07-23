#!/usr/bin/env bash
# Fine-tune QwenGR00T (Qwen-VL backbone + GR00T flow-matching action head) on
# the piston89 Unitree G1 (Inspire) dataset.
#
# Differences vs the PI05 launcher (run_starvla_pi05_piston_train.sh):
#   - No converted OpenPI checkpoint / paligemma tokenizer. The Qwen-VL
#     backbone is pulled from BASE_VLM (HF id auto-downloads); the GR00T
#     action head trains FROM SCRATCH.
#   - Exact dims (action 26 / state 29), no 32-dim padding.
#   - Per-module LRs (VLM low, action head high), cosine schedule, and
#     gradient checkpointing (supported by the Qwen backbone; PI05 lacks it).
#
# Memory guide (Qwen3-VL-4B + ~0.5B head):
#   - full fine-tune (FREEZE_MODULES=''): ~80GB+; fits H100-80GB tightly /
#     Blackwell-96GB comfortably thanks to gradient checkpointing.
#   - VLM frozen (FREEZE_MODULES='qwen_vl_interface'): ~20GB; fits 24GB cards,
#     or 16GB with OFFLOAD_OPTIMIZER=true and PER_DEVICE_BATCH_SIZE=1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# -- Runtime / hardware ------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-false}"

# -- Data / model ------------------------------------------------------------
CONFIG_YAML="${CONFIG_YAML:-examples/realRobots/UnitreeG1_Piston/train_files/starvla_qwengr00t_piston.yaml}"
DATA_ROOT="${DATA_ROOT:-${HOME}/Datasets/piston89_0260713}"
DATA_MIX="${DATA_MIX:-piston_g1_inspire}"
# HF id or local path of the Qwen VL backbone.
BASE_VLM="${BASE_VLM:-Qwen/Qwen3-VL-4B-Instruct}"
# Optional: resume/init from an existing starVLA checkpoint (.pt). Empty = the
# GR00T head starts from scratch and only the VLM comes pretrained.
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT-}"

# GR00T's head has no padding slack: this must equal the dataset's action dim
# exactly (piston89: 26, pickplace: 30). State dim is 29 for both.
ACTION_DIM="${ACTION_DIM:-26}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/starvla}"
RUN_ID="${RUN_ID:-starvla_qwengr00t_piston_g1}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA_unitree_g1_piston}"
WANDB_ENTITY="${WANDB_ENTITY:-jren313-georgia-institute-of-technology}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# '' = full fine-tune (VLM at low LR + head at high LR, the SONIC-example
# recipe). Use 'qwen_vl_interface' to train the action head (+ projector
# internals) only. NOTE the '-' (not ':-'): empty means "freeze nothing".
FREEZE_MODULES="${FREEZE_MODULES-}"

# true = resume from the newest checkpoint inside this RUN_ID's checkpoints/.
IS_RESUME="${IS_RESUME:-false}"

# -- Training knobs -----------------------------------------------------------
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
# The step counter counts MICRO-batches when DeepSpeed manages accumulation:
# 124000 micro x batch 4 = ~7 epochs over piston89's 70,985 frames.
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-124000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-8000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-18000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
VIDEO_BACKEND="${VIDEO_BACKEND:-decord}"

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${CONFIG_YAML}" "${output_dir}/"

# -- DeepSpeed / accelerate configs (generated per run) -----------------------
# Same rationale as the PI05 launcher: gradient accumulation is only honored
# from the DeepSpeed json, so it is generated here with the env values baked in.
if [[ -z "${ACCEL_CONFIG:-}" ]]; then
  DS_JSON="${output_dir}/ds_config.json"
  ACCEL_CONFIG="${output_dir}/accelerate_config.yaml"
  if [[ "${OFFLOAD_OPTIMIZER}" == "true" ]]; then
    OFFLOAD_BLOCK='"offload_optimizer": {"device": "cpu", "pin_memory": true},'
  else
    OFFLOAD_BLOCK=''
  fi
  cat > "${DS_JSON}" <<EOF
{
    "fp16": {"enabled": false},
    "bf16": {"enabled": true},
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": ${GRAD_ACCUM_STEPS},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 2e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 2e8,
        "overlap_comm": true,
        "contiguous_gradients": true,
        ${OFFLOAD_BLOCK}
        "round_robin_gradients": false
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}
EOF
  cat > "${ACCEL_CONFIG}" <<EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: "${DS_JSON}"
  deepspeed_multinode_launcher: standard
  zero3_init_flag: false
distributed_type: DEEPSPEED
num_machines: 1
num_processes: ${NUM_PROCESSES}
EOF
fi

echo "[launcher] framework=QwenGR00T base_vlm='${BASE_VLM}' freeze_modules='${FREEZE_MODULES}'" \
     "action_dim=${ACTION_DIM} data_mix=${DATA_MIX}" \
     "grad_accum=${GRAD_ACCUM_STEPS} offload_optimizer=${OFFLOAD_OPTIMIZER}"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.action_model.action_dim "${ACTION_DIM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.video_backend "${VIDEO_BACKEND}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --datasets.vla_data.prefetch_factor "${PREFETCH_FACTOR}" \
  --datasets.vla_data.persistent_workers true \
  --datasets.vla_data.pin_memory true \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.is_resume "${IS_RESUME}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  "${EXTRA_ARGS[@]}"
