#!/usr/bin/env bash
# Fine-tune PI05 (OpenPI base) on the piston89 Unitree G1 (Inspire) dataset.
# DataConfig: examples/realRobots/UnitreeG1_Piston/train_files/data_registry/data_config.py
#   robot_type unitree_g1_inspire_piston -> 29-dim upper-body state, 26-dim action.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives at examples/realRobots/UnitreeG1_Piston/train_files/ (4 deep).
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# -- Runtime / hardware ------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
# CPU-offload the AdamW states (needed on <40GB cards; pass false on 80GB
# GPUs for full speed).
OFFLOAD_OPTIMIZER="${OFFLOAD_OPTIMIZER:-true}"

# -- Data / checkpoint / config ----------------------------------------------
# Framework block (action_dim=32, max_state_dim=32, action_horizon=10, PI05)
# is reused from the LIBERO PI05 yaml; only the data + run bits are overridden.
CONFIG_YAML="${CONFIG_YAML:-examples/simBenchmarks/LIBERO/train_files/openpi/pi05_libero_8gpu.yaml}"
DATA_ROOT="${DATA_ROOT:-${HOME}/Datasets/piston89_0260713}"
DATA_MIX="${DATA_MIX:-piston_g1_inspire}"
TOKENIZER_MODEL="${TOKENIZER_MODEL:-paligemma_tokenizer.model}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-openpi_converted_protocol/pi05_base_starvla/fp32/model.safetensors}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/starvla}"
RUN_ID="${RUN_ID:-starvla_pi05_piston_g1}"
WANDB_PROJECT="${WANDB_PROJECT:-starvla_pi}"
WANDB_ENTITY="${WANDB_ENTITY:-jren313-georgia-institute-of-technology}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# The PI05 base is 4.14B params (vlm 3.45B + action_head 0.69B). A full
# fine-tune needs ~50GB for AdamW states; on a single 16GB card freeze the
# VLM and train the action expert. Pass FREEZE_MODULES='' (empty) for a full
# fine-tune on big GPUs. NOTE the '-' (not ':-') expansion below: an
# explicitly empty FREEZE_MODULES must mean "freeze nothing", not the default.
FREEZE_MODULES="${FREEZE_MODULES-vlm}"

# true = resume from the newest steps_N_pytorch_model.pt inside this RUN_ID's
# checkpoints/ dir (restores weights + step count + LR schedule position;
# optimizer moments restart). RUN_ID must match the interrupted run.
IS_RESUME="${IS_RESUME:-false}"

# -- Training knobs (89-episode fine-tune; smaller/shorter than LIBERO) -------
# Batch 1 + grad-accum 4 fits a 16GB card (no gradient checkpointing in
# PI05; batch 2 OOMs in backward at ~13.8GB). Raise batch on bigger GPUs.
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
# piston89 videos are mpeg4 -> decord decodes them fine and doesn't depend on
# torchvision's removed VideoReader API (torchvision >= 0.23 dropped it, which
# breaks the 'torchvision_av' backend the LIBERO/AV1 yaml uses).
VIDEO_BACKEND="${VIDEO_BACKEND:-decord}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-30000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-2000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-5}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${CONFIG_YAML}" "${output_dir}/"

# -- DeepSpeed / accelerate configs (generated per run) -----------------------
# The trainer reads gradient accumulation from the DeepSpeed config, NOT from
# --trainer.gradient_accumulation_steps (which nothing consumes). The repo's
# static ds_config.yaml hardcodes accumulation=1 and silently ignores the env
# knob, so we generate the config here with the requested values baked in.
# Pass ACCEL_CONFIG=<accelerate yaml> to bypass generation entirely (then
# GRAD_ACCUM_STEPS/OFFLOAD_OPTIMIZER come from that file's ds_config, not env).
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

echo "[launcher] freeze_modules='${FREEZE_MODULES}' grad_accum=${GRAD_ACCUM_STEPS}" \
     "offload_optimizer=${OFFLOAD_OPTIMIZER} accel_config=${ACCEL_CONFIG}"

accelerate launch \
  --config_file "${ACCEL_CONFIG}" \
  --num_processes "${NUM_PROCESSES}" \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.name PI05 \
  --framework.tokenizer.model_path "${TOKENIZER_MODEL}" \
  --framework.action_horizon 10 \
  --framework.action_dim 32 \
  --framework.max_state_dim 32 \
  --framework.max_token_len 200 \
  --framework.discrete_state_input false \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.action_mode abs \
  --datasets.vla_data.video_backend "${VIDEO_BACKEND}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --datasets.vla_data.prefetch_factor "${PREFETCH_FACTOR}" \
  --datasets.vla_data.persistent_workers true \
  --datasets.vla_data.pin_memory true \
  --trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}" \
  --trainer.freeze_modules "${FREEZE_MODULES}" \
  --trainer.is_resume "${IS_RESUME}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}" \
  --trainer.lr_scheduler_type constant_with_warmup \
  --trainer.learning_rate.base "${LEARNING_RATE}" \
  --trainer.learning_rate.action_head "${LEARNING_RATE}" \
  --trainer.optimizer.weight_decay 1.0e-10 \
  --trainer.gradient_clipping 1.0 \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}"
