"""QwenOFT PICO-only + DAgger q1 data mixes (2026-11-04), and the GRU-memory variant of the 3-view step-3 config
(2026-11-06).

q1all: data_pico1103 (110 PICO episodes) + data_dagger_q1 (the PICO-only QwenOFT's autonomous successes at decay 1.0
and DAgger round 2), joined under one root of symlinks (/jy/pipette/outputs/aim-r3-ft/data_q1all). Dataset weights =
trajectory counts, so every trajectory is drawn equally often, as in a single-dataset run (the loader samples a dataset
by weight, then a trajectory uniformly).

mem: the same 3-view step-3 config with a 1 s history for QwenOFTMem (starVLA/model/framework/VLM4A/QwenOFTMem.py):
the three views at -60, -48, -36, -24, -12 and 0 frames (one inference every 12 frames = 5 Hz, as the robot serves),
and the input state (the wrist velocity) at the same offsets. The state modality itself stays at [0] -- action_mode
"rel" anchors on its row 0 -- and the history state rows are appended after it, so labels are untouched. Samples carry
history_images / history_state (oldest first) next to the usual image / state.
"""
import sys

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig

ROBOT = "unitree_g1_pipette_egofocuswristvel_pi05sq224_wrist12rel_lthumbrel_joint14"
ROBOT_MEM = ROBOT + "_mem6x12"
HISTORY = [-60, -48, -36, -24, -12, 0]

_base = sys.modules["_data_registry_UnitreeG1_Pipette.data_config"]
_Base = type(_base.ROBOT_TYPE_CONFIG_MAP[ROBOT])


class MemoryDataset(LeRobotSingleDataset):
    """LeRobotSingleDataset with the history views / input-state rows packed as history_images / history_state."""

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        data = super().get_step_data(trajectory_id, base_index)
        keys = [str(k) for k in (self.data_cfg.get("state_input_keys", None) or []) if str(k) in data]
        for off in HISTORY[:-1]:                                      # oldest first; clamped at the episode start
            b = max(0, base_index + off)
            for k in keys:
                data[k] = np.concatenate([data[k], self.get_data_by_modality(trajectory_id, "state", k, b)], 0)
        return data

    def _pack_sample(self, data: dict) -> dict:
        vkeys = self.modality_keys["video"]
        skeys = [str(k) for k in (self.data_cfg.get("state_input_keys", None) or []) if str(k) in data]
        n = len(HISTORY)

        def step(t):                                                  # t = 0 .. n-1, n-1 = now
            d = dict(data)
            for k in vkeys:
                d[k] = data[k][t:t + 1]
            for k in skeys:
                row = 0 if t == n - 1 else 1 + t                     # row 0 = now, rows 1.. = the history
                d[k] = data[k][row:row + 1]
            return d

        sample = super()._pack_sample(step(n - 1))
        hist = [super(MemoryDataset, self)._pack_sample(step(t)) for t in range(n - 1)]
        sample["history_images"] = [h["image"] for h in hist]
        if "state" in sample:
            sample["history_state"] = [h["state"] for h in hist]
        return sample


class Pipette3ViewEgoFocusWristVelMemDataConfig(_Base):
    """the 3-view step-3 config + a 6-step, 1 s view / input-state history (see the module docstring)"""

    def modality_config(self):
        mc = super().modality_config()
        mc["video"] = ModalityConfig(delta_indices=HISTORY, modality_keys=self.video_keys)
        return mc

    def make_dataset(self, dataset_path, modality_configs, transforms, embodiment_tag, video_backend,
                     delete_pause_frame, data_cfg, dataset_name):
        return MemoryDataset(dataset_path=dataset_path, modality_configs=modality_configs, transforms=transforms,
                             embodiment_tag=embodiment_tag, video_backend=video_backend,
                             delete_pause_frame=delete_pause_frame, data_cfg=data_cfg)


ROBOT_TYPE_CONFIG_MAP = {ROBOT_MEM: Pipette3ViewEgoFocusWristVelMemDataConfig()}

DATASET_NAMED_MIXTURES = {
    "unitree_g1_pipette_egofocuswristvel_p3_q1all_train_mix": [("pico1103-train", 110.0, ROBOT), ("dagger_q1-train", 68.0, ROBOT)],
    "unitree_g1_pipette_egofocuswristvel_p3_q1all_eval_mix": [("pico1103-eval", 19.0, ROBOT), ("dagger_q1-eval", 11.0, ROBOT)],
    # data_pico1103 alone (model #10's data) with the 1 s history, for QwenOFTMem
    "unitree_g1_pipette_egofocuswristvel_p3_pico1103_mem_train_mix": [("g1-pipette-egofocuswristvel-p3-insertnew-1028-pi05sq224-train", 1.0, ROBOT_MEM)],
    "unitree_g1_pipette_egofocuswristvel_p3_pico1103_mem_eval_mix": [("g1-pipette-egofocuswristvel-p3-insertnew-1028-pi05sq224-eval", 1.0, ROBOT_MEM)],
    # max data (2026-11-09), QwenOFTMem: data_pico1103 + data_dagger_r4 (11-09 right-arm-only DAgger) + data_auto_all
    # (every autonomous success), joined under data_maxmem; weights = trajectory counts, i.e. no special ratios
    "unitree_g1_pipette_egofocuswristvel_p3_maxdata_mem_train_mix": [("pico1103-train", 110.0, ROBOT_MEM), ("dagger_r4-train", 37.0, ROBOT_MEM),
                                                                     ("auto_all-train", 247.0, ROBOT_MEM)],
    "unitree_g1_pipette_egofocuswristvel_p3_maxdata_mem_eval_mix": [("pico1103-eval", 19.0, ROBOT_MEM), ("dagger_r4-eval", 6.0, ROBOT_MEM),
                                                                    ("auto_all-eval", 41.0, ROBOT_MEM)],
}
