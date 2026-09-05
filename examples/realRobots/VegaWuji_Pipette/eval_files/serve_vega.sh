#!/usr/bin/env bash
# Serve an EgoVLA Vega/Wuji checkpoint over the GR00T ZMQ protocol (node side).
#   CKPT=/data/hding95/ckpts/egovla_vega_toS5_phase3_fm/checkpoints/steps_5000_pytorch_model.pt \
#   CUDA_VISIBLE_DEVICES=7 PORT=5555 bash serve_vega.sh
# The obs/action contract (3 views ego/side/right_wrist, 54-D state, 54-D action groups) is
# derived from the checkpoint's DataConfig; the sim runner reads it via get_modality_config.
# Any GPU use on the node must be user-approved first (which GPU, how long).
set -euo pipefail
export PATH=/data/hding95/envs/starvla/bin:$PATH
export PYTHONPATH=/data/hding95/code/starVLA_egovla:${PYTHONPATH:-}
export HF_HOME=/data/hding95/hf_home HF_OFFLINE=1 TMPDIR=/data/hding95/tmp
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
CKPT=${CKPT:?set CKPT=/path/to/steps_N_pytorch_model.pt or final_model/pytorch_model.pt}
PORT=${PORT:-5555}
cd /data/hding95/code/starVLA_egovla
echo "[serve] ckpt=$CKPT gpu=$CUDA_VISIBLE_DEVICES port=$PORT"
exec python deployment/model_server/server_policy_gr00t_zmq.py --ckpt_path "$CKPT" --host 0.0.0.0 --port "$PORT" --use_bf16 "$@"
