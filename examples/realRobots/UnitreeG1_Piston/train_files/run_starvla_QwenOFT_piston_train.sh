#!/usr/bin/env bash
# Fine-tune QwenOFT (Qwen3-VL-2B backbone + MLP L1-regression action head) on
# the G1/Inspire piston pick-and-place dataset — the recipe of the reference
# checkpoint birbirll/g1-inspire-piston-starvla-oft, reproduced exactly.
#
# Differences vs the QwenGR00T launcher (run_starvla_GR00T_piston_train.sh):
#   - PLAIN accelerate, NO DeepSpeed. train_starvla.py only engages DeepSpeed
#     when ACCELERATE_USE_DEEPSPEED=true; here gradient accumulation is real
#     (accelerate accumulate(); completed_steps / LR schedule gate on
#     sync_gradients) and comes from ACCELERATE_GRADIENT_ACCUMULATION_STEPS.
#     Do NOT mix accumulation with DeepSpeed on a single GPU — under DS the
#     step counter counts micro-batches and the LR schedule runs accum× fast.
#   - bitsandbytes PagedAdamW8bit (8-bit states, paged to host RAM): the 2B
#     FULL fine-tune fits one 24GB card. pip install bitsandbytes.
#   - FULL fine-tune on purpose: QwenOFT has no separate projector (the "🔍"
#     action token rides through the VLM), so a frozen backbone leaves nothing
#     to adapt visual features — don't freeze the VLM.
#   - Vision+language-only (include_state: false in the yaml, on purpose):
#     with proprio packed the policy can fit the data from state alone and
#     ignore the camera, then never visually correct in closed loop.
#   - Color jitter (0.3/0.4/0.5/0.08) lives in PistonPickPlaceG1OFTDataConfig.
#
# Memory guide (Qwen3-VL-2B, grad ckpt on, paged 8-bit optimizer):
#   - batch 8 fits a 24GB card (reference run: RTX 4090, ~2h15m for 10k steps).
#   - on 16GB try PER_DEVICE_BATCH_SIZE=4 GRAD_ACCUM_STEPS=8 (same eff. 32).
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
CONFIG_YAML="${CONFIG_YAML:-examples/realRobots/UnitreeG1_Piston/train_files/starvla_qwenoft_piston.yaml}"
DATA_ROOT="${DATA_ROOT:-${HOME}/Datasets/pickplace_hf}"
DATA_MIX="${DATA_MIX:-unitree_g1_piston_pnp}"
# HF id or local path. Plain -Instruct is fine (QwenOFT's action token is a
# regular tokenizer symbol, no special -Action checkpoint needed).
BASE_VLM="${BASE_VLM:-Qwen/Qwen3-VL-2B-Instruct}"
# Optional: init/resume weights from an existing starVLA checkpoint (.pt).
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT-}"

RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/starvla}"
RUN_ID="${RUN_ID:-starvla_qwenoft_piston_g1}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA_unitree_g1_piston}"
WANDB_ENTITY="${WANDB_ENTITY:-jren313-georgia-institute-of-technology}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# true = resume from the newest checkpoint inside this RUN_ID's checkpoints/.
IS_RESUME="${IS_RESUME:-false}"

# -- Training knobs (reference recipe: eff. batch 32 = 8 x accum 4) -----------
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
# train_starvla.py builds its Accelerator at module import, before the yaml is
# parsed — so accumulation MUST be supplied via this env var (the yaml's
# trainer.gradient_accumulation_steps is informational; keep them in sync).
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS="${GRAD_ACCUM_STEPS}"
# Belt-and-braces: make sure no ambient accelerate/DS config flips DS on.
export ACCELERATE_USE_DEEPSPEED=false

output_dir="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${output_dir}"
cp "$0" "${output_dir}/"
cp "${CONFIG_YAML}" "${output_dir}/"

echo "[launcher] framework=QwenOFT base_vlm='${BASE_VLM}' FULL fine-tune (no freeze)" \
     "data_mix=${DATA_MIX} batch=${PER_DEVICE_BATCH_SIZE} accum=${GRAD_ACCUM_STEPS}" \
     "(plain accelerate, no DeepSpeed; optimizer=paged_adamw_8bit)"

EXTRA_ARGS=()
if [[ -n "${PRETRAINED_CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--trainer.pretrained_checkpoint "${PRETRAINED_CHECKPOINT}")
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
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
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --run_root_dir "${RUN_ROOT_DIR}" \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity "${WANDB_ENTITY}" \
  "${EXTRA_ARGS[@]}"
