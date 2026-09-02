#!/usr/bin/env bash
# Train the full GR00T-N1.7 action head on all successful Phase 00/01 data.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-examples/realRobots/VegaWuji_PipetteTube/train_files/starvla_gr00t_n1d7_pipette_grasp_phase00_01_full_action_head_success51.yaml}"
export DATA_MIX="${DATA_MIX:-vega_wuji_pipette_grasp_phase00_01_success51_n1d7}"
export DATASET_NAME="${DATASET_NAME:-g1-pipette-tube-reference-success42-phase00-01-global}"
export RUN_ID="${RUN_ID:-starvla_gr00t_n1d7_vega_wuji_pipette_grasp_phase00_01_full_action_head_success51_20260830}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-10000}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-500}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-2500}"

exec bash "${SCRIPT_DIR}/run_starvla_GR00T_N1d7_pipette_tube_train.sh" "$@"
