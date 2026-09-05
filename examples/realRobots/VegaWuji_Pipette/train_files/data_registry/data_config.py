"""Data registry for the Vega-U + Wuji Hand 2 pipette/tube task (EgoVLA line).

Datasets (built on the node by ``/data/hding95/datasets/build_toS5.py`` from the HF copy of
``HenryZ07/RFT-reference-trajectory``; LeRobot v2.1, 30 Hz, robot_type ``vega_wuji``):
  rft_pipette_toS5_single : 128 eps truncated at the post-release hold (S5), ONE instruction
  rft_pipette_toS5_phase3 : same frames, per-frame task_index switching between three
                            sub-instructions at the two long pauses (829 / 2192)

Per frame: observation.state (54) and action (54) absolute joint positions in RFT control order
  left_arm[7] | left_hand[20] | right_arm[7] | right_hand[20]
Videos: ego_view 424x240, side_view 640x360, left_wrist_view 480x360, right_wrist_view 480x360.
The LEFT wrist view is deliberately NOT used (user decision 2026-09-05); the deploy contract
(``get_modality_config`` -> runner) is derived from ``video_keys`` below, so the sim runner sends
exactly these three views in this order.

action_indices MUST equal the model's action_horizon (EgoVLA: 30 = 1 s at 30 Hz).
"""
from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)

ACTION_HORIZON = 30


class _VegaWujiBase:
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    language_keys = ["annotation.human.action.task_description"]  # key used by the HF dataset's modality.json
    observation_indices = [0]
    action_indices = list(range(ACTION_HORIZON))
    state_keys = ["state.left_arm", "state.left_hand", "state.right_arm", "state.right_hand"]
    state_key_dims = {"state.left_arm": 7, "state.left_hand": 20, "state.right_arm": 7, "state.right_hand": 20}
    action_keys = ["action.left_arm", "action.left_hand", "action.right_arm", "action.right_hand"]
    action_key_dims = {"action.left_arm": 7, "action.left_hand": 20, "action.right_arm": 7, "action.right_hand": 20}
    video_keys: list = []

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.state_keys),
            StateActionTransform(apply_to=self.state_keys,
                                 normalization_modes={k: "q99" for k in self.state_keys}),
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(apply_to=self.action_keys,
                                 normalization_modes={k: "q99" for k in self.action_keys}),
        ])


class VegaWuji3ViewDataConfig(_VegaWujiBase):
    """ego + side + right wrist (no left wrist). Order is part of the train/serve contract."""
    video_keys = ["video.ego_view", "video.side_view", "video.right_wrist_view"]


class VegaWuji4ViewDataConfig(_VegaWujiBase):
    """all four cameras (kept for ablations)."""
    video_keys = ["video.ego_view", "video.side_view", "video.left_wrist_view", "video.right_wrist_view"]


class VegaWujiEgoOnlyDataConfig(_VegaWujiBase):
    video_keys = ["video.ego_view"]


ROBOT_TYPE_CONFIG_MAP = {
    "vega_wuji_pipette_3v": VegaWuji3ViewDataConfig(),
    "vega_wuji_pipette_4v": VegaWuji4ViewDataConfig(),
    "vega_wuji_pipette_ego": VegaWujiEgoOnlyDataConfig(),
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {k: EmbodimentTag.NEW_EMBODIMENT for k in ROBOT_TYPE_CONFIG_MAP}
DATASET_NAMED_MIXTURES = {
    "vega_pipette_toS5_single_3v": [("rft_pipette_toS5_single", 1.0, "vega_wuji_pipette_3v")],
    "vega_pipette_toS5_phase3_3v": [("rft_pipette_toS5_phase3", 1.0, "vega_wuji_pipette_3v")],
    "vega_pipette_toS5_single_4v": [("rft_pipette_toS5_single", 1.0, "vega_wuji_pipette_4v")],
    "vega_pipette_toS5_phase3_4v": [("rft_pipette_toS5_phase3", 1.0, "vega_wuji_pipette_4v")],
    "vega_pipette_toS5_phase3_ego": [("rft_pipette_toS5_phase3", 1.0, "vega_wuji_pipette_ego")],
}
