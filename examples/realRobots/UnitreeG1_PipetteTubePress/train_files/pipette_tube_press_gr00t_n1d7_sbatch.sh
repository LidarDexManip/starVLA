#!/usr/bin/env bash
#SBATCH --job-name=pipette_tube_press_gr00t
#SBATCH --output=/coc/flash12/jwang3617/robot/starVLA/outputs/slurm/pipette_tube_press_gr00t_%j.out
#SBATCH --error=/coc/flash12/jwang3617/robot/starVLA/outputs/slurm/pipette_tube_press_gr00t_%j.err
#SBATCH --partition=overcap
#SBATCH --nodelist=dendrite
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00

set -euo pipefail

STARVLA_DIR=/coc/flash12/jwang3617/robot/starVLA
LAUNCHER="${STARVLA_DIR}/examples/realRobots/UnitreeG1_PipetteTubePress/train_files/run_starvla_GR00T_N1d7_pipette_tube_press_train.sh"

mkdir -p "${STARVLA_DIR}/outputs/slurm"
source "${STARVLA_DIR}/.venv/bin/activate"
cd "${STARVLA_DIR}"

export DATA_ROOT=/coc/flash12/jwang3617/robot/RFT/vega_wuji_isaac6/outputs
export BASE_VLM="${STARVLA_DIR}/playground/Pretrained_models/Cosmos-Reason2-2B"
export PRETRAINED_CHECKPOINT="${STARVLA_DIR}/playground/Pretrained_models/g1-inspire-piston-starvla-n1d7-wristcam/final_model/pytorch_model.pt"
export FREEZE_BACKBONE="${FREEZE_BACKBONE:-true}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-16}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
export RUN_ID="${RUN_ID:-pipette_tube_press_gr00t_n1d7_${SLURM_JOB_ID:-manual}}"

echo "[job] id=$(printf '%s' "${SLURM_JOB_ID:-manual}") host=$(hostname)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
exec bash "${LAUNCHER}" "$@"
