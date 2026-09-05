#!/usr/bin/env bash
# EgoVLA training launcher for the B200 node (plain accelerate DDP, no DeepSpeed).
#   CONFIG_YAML=... CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_PROCESSES=4 bash run_train_node.sh [extra --overrides]
# Every GPU launch on the node must be user-approved first (which GPUs, how long).
set -euo pipefail
export PATH=/data/hding95/envs/starvla/bin:$PATH
export PYTHONPATH=/data/hding95/code/starVLA_egovla:${PYTHONPATH:-}
export HF_HOME=/data/hding95/hf_home HF_OFFLINE=1 TMPDIR=/data/hding95/tmp
export WANDB_MODE=${WANDB_MODE:-offline}
export DDP_FIND_UNUSED_PARAMETERS=${DDP_FIND_UNUSED_PARAMETERS:-true}
# plain accelerate DDP (the trainer constructs a DeepSpeedPlugin; this env switch is how the
# working N1.7 runs on this node disabled it -- DeepSpeed here tries an MPI discovery and dies)
export ACCELERATE_USE_DEEPSPEED=false
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=${GRAD_ACCUM_STEPS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
NUM_PROCESSES=${NUM_PROCESSES:-1}
MAIN_PORT=${MAIN_PORT:-29700}
CONFIG_YAML=${CONFIG_YAML:-examples/realRobots/VegaWuji_Pipette/train_files/starvla_egovla_vega_toS5_phase3_fm.yaml}
mkdir -p "$TMPDIR" /data/hding95/ckpts/logs
cd /data/hding95/code/starVLA_egovla
echo "[launcher] EgoVLA config=$CONFIG_YAML gpus=$CUDA_VISIBLE_DEVICES procs=$NUM_PROCESSES extra=$*"
accelerate launch --num_processes "$NUM_PROCESSES" --num_machines 1 --mixed_precision no --main_process_port "$MAIN_PORT" \
  starVLA/training/train_starvla.py --config_yaml "$CONFIG_YAML" "$@"
