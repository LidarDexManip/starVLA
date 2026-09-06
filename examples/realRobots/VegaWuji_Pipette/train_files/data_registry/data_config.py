"""Data registry for the Vega-U + Wuji Hand 2 pipette/tube task (EgoVLA line).

Datasets (built on the node by ``/data/hding95/datasets/build_toS5.py`` from the HF copy of
``HenryZ07/RFT-reference-trajectory``; LeRobot v2.1, 30 Hz, robot_type ``vega_wuji``):
  rft_pipette_toS5_single : 128 eps truncated at the post-release hold (S5), ONE instruction
  rft_pipette_toS5_phase3 : same frames, per-frame task_index switching between three
                            sub-instructions at the two long pauses (829 / 2192)
  rft_pipette_toS5_phase6 : same frames, SIX sub-instructions with per-episode boundaries detected
                            from the action signal (build_phase6.py). Run 1 showed that with three
                            coarse phases the policy never opens the fingers for the drop-catch
                            slide (47 frames = 1.6 % of the post-grasp segment) and under-executes
                            the press; inside their own phases those frames are ~48 % / ~82 %, which
                            rebalances the action distribution without any loss weighting.

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
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoToNumpy,
    VideoToTensor,
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

    #: photometric augmentation on the camera streams. Run 1 trained with NO augmentation on
    #: 128 near-identical scripted demos and drifted out of distribution within ~300 steps.
    augment: bool = False

    def transform(self):
        video = []
        if self.augment:
            # PER VIEW: the three cameras have different native resolutions (424x240 / 640x360 /
            # 480x360), so a single transform instance covering all of them would try to stack
            # them into one array and fail. Each view therefore gets its own jitter draw.
            for key in self.video_keys:
                video += [
                    VideoToTensor(apply_to=[key]),
                    VideoColorJitter(apply_to=[key], brightness=0.3, contrast=0.4,
                                     saturation=0.5, hue=0.08),
                    VideoToNumpy(apply_to=[key]),
                ]
        return ComposedModalityTransform(transforms=[
            *video,
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


class VegaWuji3ViewAugDataConfig(VegaWuji3ViewDataConfig):
    """ego + side + right wrist, with photometric augmentation (training only)."""
    augment = True


class VegaWujiEgoOnlyDataConfig(_VegaWujiBase):
    video_keys = ["video.ego_view"]


class _VegaWujiPhaseState(_VegaWujiBase):
    """54-D proprio + 6-D phase one-hot in observation.state (dataset rft_pipette_toS5_phase6ph).
    The one-hot is a hard task-stage signal; EgoVLA's state dropout leaves it intact
    (framework.state_dropout_keep_dims: 6). At serve time the runner appends its event-ladder
    phase as state.phase. Uses ALL FOUR cameras: the left wrist view is the only close-up of the
    tube, and the tube is ~4 px wide in the head camera."""
    state_keys = ["state.left_arm", "state.left_hand", "state.right_arm", "state.right_hand", "state.phase"]
    state_key_dims = {"state.left_arm": 7, "state.left_hand": 20, "state.right_arm": 7, "state.right_hand": 20, "state.phase": 6}
    video_keys = ["video.ego_view", "video.side_view", "video.left_wrist_view", "video.right_wrist_view"]


class VegaWuji4ViewPhaseDataConfig(_VegaWujiPhaseState):
    pass


class VegaWuji4ViewPhaseAugDataConfig(_VegaWujiPhaseState):
    augment = True


ROBOT_TYPE_CONFIG_MAP = {
    "vega_wuji_pipette_3v": VegaWuji3ViewDataConfig(),
    "vega_wuji_pipette_3v_aug": VegaWuji3ViewAugDataConfig(),
    "vega_wuji_pipette_4v": VegaWuji4ViewDataConfig(),
    "vega_wuji_pipette_ego": VegaWujiEgoOnlyDataConfig(),
    "vega_wuji_pipette_4v_phase": VegaWuji4ViewPhaseDataConfig(),
    "vega_wuji_pipette_4v_phase_aug": VegaWuji4ViewPhaseAugDataConfig(),
}
ROBOT_TYPE_TO_EMBODIMENT_TAG = {k: EmbodimentTag.NEW_EMBODIMENT for k in ROBOT_TYPE_CONFIG_MAP}
DATASET_NAMED_MIXTURES = {
    "vega_pipette_toS5_single_3v": [("rft_pipette_toS5_single", 1.0, "vega_wuji_pipette_3v")],
    "vega_pipette_toS5_phase3_3v": [("rft_pipette_toS5_phase3", 1.0, "vega_wuji_pipette_3v")],
    "vega_pipette_toS5_single_4v": [("rft_pipette_toS5_single", 1.0, "vega_wuji_pipette_4v")],
    "vega_pipette_toS5_phase3_4v": [("rft_pipette_toS5_phase3", 1.0, "vega_wuji_pipette_4v")],
    "vega_pipette_toS5_phase3_ego": [("rft_pipette_toS5_phase3", 1.0, "vega_wuji_pipette_ego")],
    "vega_pipette_toS5_phase6_3v": [("rft_pipette_toS5_phase6", 1.0, "vega_wuji_pipette_3v")],
    "vega_pipette_toS5_phase6_3v_aug": [("rft_pipette_toS5_phase6", 1.0, "vega_wuji_pipette_3v_aug")],
    "vega_pipette_toS5_phase6ph_4v_aug": [("rft_pipette_toS5_phase6ph", 1.0, "vega_wuji_pipette_4v_phase_aug")],
}
