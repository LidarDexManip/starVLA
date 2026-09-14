#!/usr/bin/env bash
# QwenOFT launcher for the pipette three-view datasets (2026-09-11).
#
# The piston OFT launcher's CLI (plain accelerate, NO DeepSpeed, grad
# accumulation through ACCELERATE_GRADIENT_ACCUMULATION_STEPS) with the
# pipette GR00T launcher's multi-GPU plumbing: `accelerate launch
# --num_processes N` WITHOUT --multi_gpu silently runs N unrelated copies
# of a single-process job (no DDP, no gradient sync) -- it looks like a
# slow run, not a misconfiguration -- so --multi_gpu is added whenever
# NUM_PROCESSES > 1, and the rendezvous port is settable because the node
# is shared (another user's launch on the default port poisons it).
#
# Nothing GR00T-specific is passed: no tune_llm/tune_visual, no grasp
# weighting, no include_state override (the yaml's `false` is the recipe).
#
# Smoke (1 GPU, no save):
#   CUDA_VISIBLE_DEVICES=0 MAX_TRAIN_STEPS=10 SAVE_INTERVAL=100000 \
#   EVAL_INTERVAL=100000 PER_DEVICE_BATCH_SIZE=4 RUN_ID=smoke_oft \
#   bash examples/realRobots/UnitreeG1_Pipette/train_files/run_starvla_QwenOFT_pipette_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# NVLink-SHARP multicast OFF by default. On node 2 (NVSwitch), nvidia-fabricmanager
# was restarted 2026-09-12 without a reboot or GPU reset, so it holds no gpuHandle
# table: any NVLS multicast team setup arrives as gpuHandle 0x0, fabricmanager logs
# "cannot find exporter GPU in partition Id <p> gpuHandle 0x0 ... failed to add
# multicast team", and the driver BLOCKS waiting for a response that never comes.
# NCCL hangs rather than erroring, so it reads as a dead job: every rank sits at
# ~100% SYSTEM time with no user time, right after the per-rank "Using mixture"
# lines and before "Dataset lengths" (the first collective is the barrier in
# gr00t_lerobot/datasets.py). Cost 20 min on 2026-09-13 and twice before on
# 2026-09-02. Diagnose with `grep 'cannot find exporter GPU' /var/log/fabricmanager.log`.
# Every hand-written launcher on node 2 sets this; keeping it here means a launch
# that forgets the wrapper cannot reintroduce the hang. Overridable: the fabric is
# only broken until all GPUs in the partition are reset.
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MULTI_GPU_ARGS=()
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then MULTI_GPU_ARGS+=(--multi_gpu); fi
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"

CONFIG_YAML="${CONFIG_YAML:-examples/realRobots/UnitreeG1_Pipette/train_files/starvla_qwenoft_pipette_3view_hil293.yaml}"
DATA_ROOT="${DATA_ROOT:-${HOME}/Datasets}"
DATA_MIX="${DATA_MIX:-unitree_g1_pipette_3view_oft_hil293_mix}"
# Local snapshot or HF id; the yaml's value is the node-2 snapshot path.
BASE_VLM="${BASE_VLM:-/data/jren313/models/Qwen3-VL-4B-Instruct}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT-}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/starvla}"
RUN_ID="${RUN_ID:-starvla_qwenoft_pipette_3view_hil293}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA_unitree_g1_pipette}"
WANDB_ENTITY="${WANDB_ENTITY:-jren313-georgia-institute-of-technology}"
export WANDB_MODE="${WANDB_MODE:-offline}"
IS_RESUME="${IS_RESUME:-false}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-60000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-600}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
EVAL_INTERVAL="${EVAL_INTERVAL:-500}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
VIDEO_BACKEND="${VIDEO_BACKEND:-decord}"

# Plain accelerate: accumulation is real here (accelerate.accumulate(); the
# trainer gates completed_steps / lr_scheduler.step() on sync_gradients) and
# comes ONLY from this env var -- the Accelerator is built at module import.
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM_STEPS}"
export ACCELERATE_USE_DEEPSPEED=false

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${CONFIG_YAML}" "${output_dir}/"

echo "[launcher] framework=QwenOFT base_vlm='${BASE_VLM}' FULL fine-tune (no freeze)" \
     "data_mix=${DATA_MIX} procs=${NUM_PROCESSES} batch=${PER_DEVICE_BATCH_SIZE}" \
     "accum=${GRAD_ACCUM_STEPS} (plain accelerate, no DeepSpeed)"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi

accelerate launch \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --num_processes "${NUM_PROCESSES}" \
  "${MULTI_GPU_ARGS[@]}" \
  --mixed_precision no \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.video_backend "${VIDEO_BACKEND}" \
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --datasets.vla_data.num_workers "${NUM_WORKERS}" \
  --datasets.vla_data.prefetch_factor "${PREFETCH_FACTOR}" \
  --datasets.vla_data.persistent_workers true \
  --datasets.vla_data.pin_memory true \
  --trainer.is_resume "${IS_RESUME}" \
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
  --trainer.num_warmup_steps "${NUM_WARMUP_STEPS}" \
  --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
  --trainer.save_interval "${SAVE_INTERVAL}" \
  --trainer.eval_interval "${EVAL_INTERVAL}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  "${EXTRA_ARGS[@]}"
