#!/usr/bin/env bash
# Fine-tune StarVLA's CosmosGR00TN1d7 policy on the RFT pipette/tube reference
# trajectories. The default initial checkpoint is the locally downloaded
# G1-piston GR00T-N1.7 policy. Set PRETRAINED_CHECKPOINT='' to omit that
# transfer checkpoint (PRETRAINED_N1D7 must then name an official N1.7 snapshot).
#
# Required dataset root:
#   RFT/vega_wuji_isaac6/outputs/pipette_tube_press_lerobot
#
# Smoke test:
#   MAX_TRAIN_STEPS=1 NUM_WORKERS=0 GRAD_ACCUM_STEPS=1 RUN_ID=tube_press_smoke \
#     bash "$0"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"
# The shared training environment does not install this checkout as an
# editable package.  Make the in-tree ``starVLA`` package visible to the
# accelerate worker regardless of the worker script's own directory.
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export ACCELERATE_USE_DEEPSPEED=false
export WANDB_MODE="${WANDB_MODE:-online}"
# All gated/base weights are pre-staged locally.  Avoid delayed network/HF
# failures on compute nodes with no outbound route.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

NUM_PROCESSES="${NUM_PROCESSES:-1}"
MULTI_GPU_ARGS=()
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  MULTI_GPU_ARGS+=(--multi_gpu)
fi
CONFIG_YAML="${CONFIG_YAML:-examples/realRobots/UnitreeG1_PipetteTubePress/train_files/starvla_gr00t_n1d7_pipette_tube_press.yaml}"
DATA_ROOT="${DATA_ROOT:-/coc/flash12/jwang3617/robot/RFT/vega_wuji_isaac6/outputs}"
DATASET_DIR="${DATA_ROOT}/pipette_tube_press_lerobot"
DATA_MIX="${DATA_MIX:-unitree_g1_pipette_tube_press_n1d7}"
BASE_VLM="${BASE_VLM:-${REPO_ROOT}/playground/Pretrained_models/Cosmos-Reason2-2B}"
PRETRAINED_N1D7="${PRETRAINED_N1D7-}"
DEFAULT_TRANSFER="${REPO_ROOT}/playground/Pretrained_models/g1-inspire-piston-starvla-n1d7-wristcam/final_model/pytorch_model.pt"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT-${DEFAULT_TRANSFER}}"

FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
VIDEO_BACKEND="${VIDEO_BACKEND:-decord}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/starvla}"
RUN_ID="${RUN_ID:-starvla_gr00t_n1d7_pipette_tube_press}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA_unitree_g1_pipette_tube_press}"
WANDB_ENTITY="${WANDB_ENTITY:-geertswon-georgia-institute-of-technology}"
VALIDATE_DATASET="${VALIDATE_DATASET:-true}"
IS_RESUME="${IS_RESUME:-false}"

if [[ ! -f "${DATASET_DIR}/meta/info.json" ]]; then
  echo "[launcher] missing LeRobot dataset: ${DATASET_DIR}" >&2
  echo "[launcher] run the RFT collector before training" >&2
  exit 2
fi
if [[ ! -f "${BASE_VLM}/config.json" ]]; then
  echo "[launcher] missing Cosmos backbone: ${BASE_VLM}" >&2
  exit 2
fi
if [[ -n "${PRETRAINED_CHECKPOINT}" && ! -f "${PRETRAINED_CHECKPOINT}" ]]; then
  echo "[launcher] missing transfer checkpoint: ${PRETRAINED_CHECKPOINT}" >&2
  exit 2
fi

if [[ "${VALIDATE_DATASET}" == "true" ]]; then
  python examples/realRobots/UnitreeG1_PipetteTubePress/train_files/validate_pipette_tube_press_dataset.py \
    --dataset "${DATASET_DIR}"
fi

if [[ "${FREEZE_BACKBONE}" == "true" ]]; then
  TUNE_LLM=false
  TUNE_VISUAL=false
else
  TUNE_LLM=true
  TUNE_VISUAL=true
fi

export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM_STEPS}"

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${CONFIG_YAML}" "${output_dir}/"

echo "[launcher] framework=CosmosGR00TN1d7 data=${DATASET_DIR}"
echo "[launcher] 4 cameras, state/action=54/54, horizon=30, abs_qpos"
echo "[launcher] transfer=${PRETRAINED_CHECKPOINT:-official-N1.7-only} freeze_backbone=${FREEZE_BACKBONE}"
echo "[launcher] batch=${PER_DEVICE_BATCH_SIZE} accum=${GRAD_ACCUM_STEPS} steps=${MAX_TRAIN_STEPS}"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_N1D7}" ]]; then
  EXTRA_ARGS+=(--framework.pretrained_gr00t_n1d7 "${PRETRAINED_N1D7}")
fi
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  "${MULTI_GPU_ARGS[@]}" \
  --mixed_precision no \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.tune_llm "${TUNE_LLM}" \
  --framework.tune_visual "${TUNE_VISUAL}" \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.include_state true \
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
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
