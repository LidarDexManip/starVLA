#!/usr/bin/env bash
# Direct (non-Slurm) launcher for Vega/Wuji pipette-tube imitation learning.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
WORKSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
PYTHON_BIN="${STARVLA_PYTHON:-${WORKSPACE_ROOT}/.conda-envs/starVLA/bin/python}"
BASE_LAUNCHER="${REPO_ROOT}/examples/realRobots/UnitreeG1_Piston/train_files/run_starvla_GR00T_N1d7_piston_train.sh"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "[preflight] starVLA Python is not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if ! "${PYTHON_BIN}" -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "[preflight] CUDA is not visible. Run this launcher from a GPU compute shell." >&2
  exit 2
fi

export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
export CONFIG_YAML="${CONFIG_YAML:-examples/realRobots/VegaWuji_PipetteTube/train_files/starvla_gr00t_n1d7_pipette_tube.yaml}"
export DATA_ROOT="${DATA_ROOT:-${WORKSPACE_ROOT}/datasets}"
export DATA_MIX="${DATA_MIX:-vega_wuji_pipette_tube_phase_n1d7}"
export DATASET_NAME="${DATASET_NAME:-g1-pipette-tube-drive-phase-labeled}"
export RUN_ROOT_DIR="${RUN_ROOT_DIR:-outputs/starvla}"
export RUN_ID="${RUN_ID:-starvla_gr00t_n1d7_vega_wuji_pipette_tube_phase_clean_official}"
export WANDB_ENTITY="${WANDB_ENTITY:-geertswon-georgia-institute-of-technology}"
export WANDB_PROJECT="${WANDB_PROJECT:-starVLA_vega_wuji_pipette_tube}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_OFFLINE="${HF_OFFLINE:-1}"
export NUM_PROCESSES="${NUM_PROCESSES:-1}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-32}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"

DATASET_DIR="${DATA_ROOT}/${DATASET_NAME}"
if [[ ! -f "${DATASET_DIR}/meta/info.json" ]] || [[ ! -d "${DATASET_DIR}/videos" ]]; then
  echo "[preflight] dataset is incomplete: ${DATASET_DIR}" >&2
  exit 2
fi

# Start from the clean official NVIDIA checkpoint. Do not silently fall back to
# the G1/Inspire piston model: its backbone was fully fine-tuned on a different
# embodiment, which defeats the purpose of this Vega/Wuji run.
DEFAULT_N1D7="${WORKSPACE_ROOT}/models/nvidia-GR00T-N1.7-3B"
export PRETRAINED_N1D7="${PRETRAINED_N1D7:-${DEFAULT_N1D7}}"
export PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT-}"

if ! compgen -G "${PRETRAINED_N1D7}/*.safetensors" >/dev/null; then
  echo "[preflight] official GR00T-N1.7 shards not found: ${PRETRAINED_N1D7}" >&2
  echo "[preflight] set PRETRAINED_N1D7 to the nvidia/GR00T-N1.7-3B snapshot." >&2
  exit 2
fi

# The current RTX PRO 5000 has 48 GiB, so default to the memory-light head-only
# path. Set FREEZE_BACKBONE=false only on an >=80 GiB accelerator.
export FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"

mkdir -p "${REPO_ROOT}/${RUN_ROOT_DIR}/${RUN_ID}"
cp "$0" "${REPO_ROOT}/${RUN_ROOT_DIR}/${RUN_ID}/"

echo "[launcher] direct GPU run (no Slurm)"
echo "[launcher] dataset=${DATASET_DIR} mixture=${DATA_MIX} views=4 state_dim=54 action_dim=54"
echo "[launcher] official_n1d7=${PRETRAINED_N1D7} transfer_checkpoint=${PRETRAINED_CHECKPOINT:-none} freeze_backbone=${FREEZE_BACKBONE}"

cd "${REPO_ROOT}"
exec bash "${BASE_LAUNCHER}" "$@"
