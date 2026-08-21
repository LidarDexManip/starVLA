# Unitree G1 pipette/tube imitation learning

The merged RFT reference trajectory lives in:

`RFT/vega_wuji_isaac6/scripts/pipette_tube_press/pipette_tube_press_demo_physx.py`

Its LeRobot recorder is:

`RFT/vega_wuji_isaac6/scripts/pipette_tube_press/pipette_tube_press_collect_lerobot.py`

The recorder appends episodes under
`RFT/vega_wuji_isaac6/outputs/pipette_tube_press_lerobot`.  It writes LeRobot
v2.1: per-episode Parquet, one H.264 MP4 per camera, and the standard `meta/`
files.  Each row contains a 54-D measured joint state and a 54-D absolute joint
position command at 30 Hz.  The four views are ego, side, left wrist, and right
wrist.

This directory registers that dataset with StarVLA and fine-tunes
`CosmosGR00TN1d7`. Generate the dataset first, then submit:

```bash
cd /coc/flash12/jwang3617/robot/RFT
sbatch vega_wuji_isaac6/scripts/pipette_tube_press/pipette_tube_press_collect_lerobot_sbatch.sh

# After the collector succeeds:
cd /coc/flash12/jwang3617/robot/starVLA
sbatch examples/realRobots/UnitreeG1_PipetteTubePress/train_files/pipette_tube_press_gr00t_n1d7_sbatch.sh
```

The default is a memory-safe A40 run: the Cosmos visual/language backbone is
frozen, the GR00T flow-matching action head is trained, and the locally cached
G1 piston GR00T checkpoint supplies the transfer initialization.  Set
`FREEZE_BACKBONE=false` only on a substantially larger GPU.

The collector environment needs `dexmotion==0.4.2` and `pyarrow==21.0.0`.
The StarVLA environment needs `pyarrow==21.0.0` and `bitsandbytes==0.50.0`.

Before a long training run, validate both the v2.1 files and one fully
transformed GR00T sample:

```bash
source /coc/flash12/jwang3617/robot/starVLA/.venv/bin/activate
cd /coc/flash12/jwang3617/robot/starVLA
python examples/realRobots/UnitreeG1_PipetteTubePress/train_files/validate_pipette_tube_press_dataset.py
```
