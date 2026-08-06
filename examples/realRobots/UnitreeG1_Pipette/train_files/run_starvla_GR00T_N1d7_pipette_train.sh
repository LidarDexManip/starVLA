#!/usr/bin/env bash
# Fine-tune GR00T-N1.7 (Cosmos-Reason2-2B backbone + 32-layer alternate-VL
# flow-matching DiT head) on the G1/Inspire PIPETTE TIP dataset, INITIALISED
# DIRECTLY from the public nvidia/GR00T-N1.7-3B parameters.
#
# Dataset: jren313/g1-pipette-tip-teleop (LeRobot v2.1, 64 eps / 56,742 frames
# @ 60 fps, single ego view). Fetch it once with:
#   hf download jren313/g1-pipette-tip-teleop --repo-type dataset \
#     --local-dir "${DATA_ROOT:-$HOME/Datasets}/g1-pipette-tip-teleop"
#
# Framework CosmosGR00TN1d7 loads all 1031 checkpoint tensors 1:1 (backbone +
# head, verified) then fine-tunes through starVLA's usual pipeline.
#
#   - Backbone is Cosmos-Reason2-2B (Qwen3-VL), LLM truncated to select_layer=16.
#   - Head is flow-matching and CONDITIONS ON STATE -> include_state is TRUE.
#   - ~3.1B params. Full fine-tune fits the 96GB Blackwell card. On 24GB set
#     FREEZE_BACKBONE=true (trains the 1.6B head only, VL features detached)
#     and/or PER_DEVICE_BATCH_SIZE=1.
#
# PLAIN accelerate + bitsandbytes PagedAdamW8bit, NO DeepSpeed (accum via
# ACCELERATE_GRADIENT_ACCUMULATION_STEPS; do not mix accum with DeepSpeed).
#
# INITIALISATION: leaving PRETRAINED_CHECKPOINT empty (the default) trains from
# the OFFICIAL nvidia/GR00T-N1.7-3B weights alone. Set it ONLY to continue from a
# previous starVLA run -- it is applied AFTER the N1.7 load and overwrites it.
#
# Action/state spaces here are 24/17, NOT the piston recipe's 30/29 -- several
# hand channels in this recording are constant and would reach the head
# un-normalised. See data_registry/data_config.py for the measurement.
#
# Smoke test before committing a GPU (1 step, tiny batch, no wandb):
#   MAX_TRAIN_STEPS=1 PER_DEVICE_BATCH_SIZE=1 GRAD_ACCUM_STEPS=1 \
#   NUM_WORKERS=0 RUN_ID=pipette_smoke bash "$0"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "${REPO_ROOT}"

# -- Runtime / hardware ------------------------------------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

# -- Data / model ------------------------------------------------------------
CONFIG_YAML="${CONFIG_YAML:-examples/realRobots/UnitreeG1_Pipette/train_files/starvla_gr00t_n1d7_pipette.yaml}"
DATA_ROOT="${DATA_ROOT:-${HOME}/Datasets}"
DATA_MIX="${DATA_MIX:-unitree_g1_pipette_n1d7}"
BASE_VLM="${BASE_VLM:-nvidia/Cosmos-Reason2-2B}"
# N1.7 checkpoint dir (with model-*.safetensors). Empty => auto-find in the HF
# cache (models--nvidia--GR00T-N1.7-3B). Download once with:
#   huggingface-cli download nvidia/GR00T-N1.7-3B
PRETRAINED_N1D7="${PRETRAINED_N1D7-}"
# Optional: init/resume weights from an existing starVLA checkpoint (.pt).
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT-}"

# Freeze the Cosmos backbone and train the flow-matching head only (memory-
# light; recommended on <=24GB). Full fine-tune (false) needs the 96GB card.
FREEZE_BACKBONE="${FREEZE_BACKBONE:-false}"

# Up-weight the (short) tip-ejector press in the loss. 1.0 = off (default:
# get a baseline first). Dim 18 == action.hand_right[4] == thumb_bend; under
# min_max on 0..1000 the normalised rest value is 0.0 and a +0.04 threshold
# (raw > 520) flags ~1.7% of frames. Try GRASP_LOSS_WEIGHT=3.0 if a baseline
# run under-presses the ejector.
GRASP_LOSS_WEIGHT="${GRASP_LOSS_WEIGHT:-1.0}"
GRASP_WEIGHT_DIM="${GRASP_WEIGHT_DIM:-18}"
GRASP_WEIGHT_THRESH="${GRASP_WEIGHT_THRESH:-0.04}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/starvla}"
RUN_ID="${RUN_ID:-starvla_gr00t_n1d7_pipette_g1}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA_unitree_g1_pipette}"
WANDB_ENTITY="${WANDB_ENTITY:-jren313-georgia-institute-of-technology}"
export WANDB_MODE="${WANDB_MODE:-offline}"

IS_RESUME="${IS_RESUME:-false}"

# -- Training knobs (eff. batch 32 = 8 x accum 4) -----------------------------
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-2}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-50}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-10}"
VIDEO_BACKEND="${VIDEO_BACKEND:-decord}"

# -- Plain accelerate (NO DeepSpeed) ------------------------------------------
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM_STEPS}"
export ACCELERATE_USE_DEEPSPEED=false

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${CONFIG_YAML}" "${output_dir}/"

# tune flags derived from FREEZE_BACKBONE
if [[ "${FREEZE_BACKBONE}" == "true" ]]; then TUNE_LLM=false; TUNE_VISUAL=false; else TUNE_LLM=true; TUNE_VISUAL=true; fi

echo "[launcher] framework=CosmosGR00TN1d7 base_vlm='${BASE_VLM}' freeze_backbone=${FREEZE_BACKBONE}" \
     "data_mix=${DATA_MIX} batch=${PER_DEVICE_BATCH_SIZE} accum=${GRAD_ACCUM_STEPS}" \
     "(plain accelerate, no DeepSpeed; optimizer=paged_adamw_8bit; loads nvidia/GR00T-N1.7-3B)"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_N1D7}" ]]; then
  EXTRA_ARGS+=(--framework.pretrained_gr00t_n1d7 "${PRETRAINED_N1D7}")
fi
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision no \
  starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG_YAML}" \
  --framework.qwenvl.base_vlm "${BASE_VLM}" \
  --framework.tune_llm "${TUNE_LLM}" \
  --framework.tune_visual "${TUNE_VISUAL}" \
  --framework.action_model.grasp_loss_weight "${GRASP_LOSS_WEIGHT}" \
  --framework.action_model.grasp_weight_dim "${GRASP_WEIGHT_DIM}" \
  --framework.action_model.grasp_weight_thresh "${GRASP_WEIGHT_THRESH}" \
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
  "${EXTRA_ARGS[@]}"
