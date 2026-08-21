"""GR00T-N1.7 data contract for the RFT pipette/tube reference trajectories.

The source dataset is produced by::

    RFT/vega_wuji_isaac6/scripts/pipette_tube_press/
        pipette_tube_press_collect_lerobot.py

It is LeRobot v2.1 at 30 Hz with four RGB views and 54-dimensional measured
joint state / absolute joint-position command vectors.  The Parquet vectors are
stored in RFT control order (left arm, left hand, right arm, right hand), while
this config deliberately presents them to GR00T in the category order used by
the pretrained Unitree-G1 embodiment slot (arms first, then hands).  The
metadata slices make this reorder lossless.

Model/inference order (54 dimensions)::

    left_arm[7] | right_arm[7] | left_hand[20] | right_hand[20]

The deployment adapter must invert that permutation before passing an action to
``VegaWujiRedBallControl``, whose order is left arm, left hand, right arm,
right hand.
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
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


class PipetteTubePressG1GR00TN1d7DataConfig:
    """Four-view, full-hand GR00T-N1.7 imitation-learning configuration."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    # Camera order is part of the train/serve contract.  The native resolutions
    # differ, so each view is resized before the shared color augmentation.
    video_keys = [
        "video.ego_view",
        "video.side_view",
        "video.left_wrist_view",
        "video.right_wrist_view",
    ]

    # Arms-first order keeps the first 14 dimensions aligned with GR00T's
    # pretrained Unitree-G1 embodiment slot.  The LeRobot modality metadata
    # selects each group from the collector's original 54-D vector.
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
    ]
    state_key_dims = {
        "state.left_arm": 7,
        "state.right_arm": 7,
        "state.left_hand": 20,
        "state.right_hand": 20,
    }

    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    action_key_dims = {
        "action.left_arm": 7,
        "action.right_arm": 7,
        "action.left_hand": 20,
        "action.right_hand": 20,
    }

    language_keys = ["annotation.human.task_description"]
    observation_indices = [0]
    state_indices = [0]
    # Thirty control samples are one second at the collector's 30 Hz rate.
    action_indices = list(range(30))
    view_resize_hw = (224, 224)

    def modality_config(self):
        return {
            "video": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.video_keys,
            ),
            "state": ModalityConfig(
                delta_indices=self.state_indices,
                modality_keys=self.state_keys,
            ),
            "action": ModalityConfig(
                delta_indices=self.action_indices,
                modality_keys=self.action_keys,
            ),
            "language": ModalityConfig(
                delta_indices=self.observation_indices,
                modality_keys=self.language_keys,
            ),
        }

    def transform(self):
        # VideoToTensor validates each view against its native metadata size.
        # Resize therefore comes after it, and is done per-view because the
        # source resolutions are different.  Once resized, a shared jitter draw
        # keeps the four synchronized camera images photometrically consistent.
        height, width = self.view_resize_hw
        per_view = []
        for key in self.video_keys:
            per_view.extend(
                [
                    VideoToTensor(apply_to=[key]),
                    VideoResize(
                        apply_to=[key],
                        height=height,
                        width=width,
                        interpolation="linear",
                    ),
                ]
            )

        return ComposedModalityTransform(
            transforms=[
                *per_view,
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.2,
                    contrast=0.3,
                    saturation=0.3,
                    hue=0.05,
                ),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=self.state_keys + self.action_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={key: "q99" for key in self.state_keys},
                ),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={key: "q99" for key in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "unitree_g1_pipette_tube_press_n1d7": PipetteTubePressG1GR00TN1d7DataConfig(),
}


# data_root_dir points at RFT/vega_wuji_isaac6/outputs; the dataset collector
# creates the folder below it.
DATASET_NAMED_MIXTURES = {
    "unitree_g1_pipette_tube_press_n1d7": [
        (
            "pipette_tube_press_lerobot",
            1.0,
            "unitree_g1_pipette_tube_press_n1d7",
        ),
    ],
}
