#!/usr/bin/env bash
# Train the two-phase Vega/Wuji pipette approach-and-grasp subset directly on a GPU host.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-examples/realRobots/VegaWuji_PipetteTube/train_files/starvla_gr00t_n1d7_pipette_grasp_phase00_01.yaml}"
export DATA_MIX="${DATA_MIX:-vega_wuji_pipette_grasp_phase00_01_n1d7}"
export DATASET_NAME="${DATASET_NAME:-g1-pipette-tube-drive-phase00-01-global}"
export RUN_ID="${RUN_ID:-starvla_gr00t_n1d7_vega_wuji_pipette_grasp_phase00_01_official}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-3000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-500}"

exec bash "${SCRIPT_DIR}/run_starvla_GR00T_N1d7_pipette_tube_train.sh" "$@"
