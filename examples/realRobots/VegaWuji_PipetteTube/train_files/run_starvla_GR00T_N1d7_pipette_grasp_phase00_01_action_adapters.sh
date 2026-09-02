#!/usr/bin/env bash
# Train only Vega/Wuji embodiment adapters while preserving official N1.7.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CONFIG_YAML="${CONFIG_YAML:-examples/realRobots/VegaWuji_PipetteTube/train_files/starvla_gr00t_n1d7_pipette_grasp_phase00_01_action_adapters.yaml}"
export DATA_MIX="${DATA_MIX:-vega_wuji_pipette_grasp_phase00_01_augmented_n1d7}"
export DATASET_NAME="${DATASET_NAME:-g1-pipette-tube-drive-phase00-01-global}"
export RUN_ID="${RUN_ID:-starvla_gr00t_n1d7_vega_wuji_pipette_grasp_phase00_01_right_only_augmented}"
export MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-2800}"
export NUM_WARMUP_STEPS="${NUM_WARMUP_STEPS:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-400}"

exec bash "${SCRIPT_DIR}/run_starvla_GR00T_N1d7_pipette_tube_train.sh" "$@"
