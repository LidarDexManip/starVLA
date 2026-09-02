"""Vega-U + Wuji Hand 2 pipette/tube imitation-learning dataset config.

The Google Drive dataset is LeRobot v2.1 with four RGB streams, 54-D
proprioception, and 54-D absolute joint-position actions.  The joint order is
the same strict order used by RFT's ``VegaWujiRedBallControl``:

``left_arm[7] | left_hand[20] | right_arm[7] | right_hand[20]``.
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


class VegaWujiPipetteTubeGR00TN1d7DataConfig:
    """Four-camera, 54-DoF GR00T-N1.7 fine-tuning contract."""

    # Slot 31 is the checkpoint's reserved new-embodiment projector.
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    # Keep this in RFT CAMERA_SPECS order. Deployment must supply the same order.
    video_keys = [
        "video.ego_view",
        "video.side_view",
        "video.left_wrist_view",
        "video.right_wrist_view",
    ]
    state_keys = [
        "state.left_arm",
        "state.left_hand",
        "state.right_arm",
        "state.right_hand",
    ]
    state_key_dims = {
        "state.left_arm": 7,
        "state.left_hand": 20,
        "state.right_arm": 7,
        "state.right_hand": 20,
    }
    action_keys = [
        "action.left_arm",
        "action.left_hand",
        "action.right_arm",
        "action.right_hand",
    ]
    action_key_dims = {
        "action.left_arm": 7,
        "action.left_hand": 20,
        "action.right_arm": 7,
        "action.right_hand": 20,
    }

    # This is the literal annotation key in the downloaded modality.json.
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(30))
    # Cosmos-Reason2's Qwen image processor has a 256**2 minimum pixel area.
    # Resize directly to that effective VLM input size so training performs one
    # geometric resize instead of 224x224 here followed by an upsample to 256x256.
    view_resize_hw = (256, 256)

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
        # Native resolutions differ across cameras. Convert and resize each view
        # independently before applying one shared color-jitter draw to all views.
        height, width = self.view_resize_hw
        per_view = []
        for key in self.video_keys:
            per_view.append(VideoToTensor(apply_to=[key]))
            per_view.append(
                VideoResize(
                    apply_to=[key],
                    height=height,
                    width=width,
                    interpolation="linear",
                )
            )

        return ComposedModalityTransform(
            transforms=[
                *per_view,
                VideoColorJitter(
                    apply_to=self.video_keys,
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.08,
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
    "vega_wuji_pipette_tube_n1d7": VegaWujiPipetteTubeGR00TN1d7DataConfig(),
}


DATASET_NAMED_MIXTURES = {
    "vega_wuji_pipette_tube_n1d7": [
        ("g1-pipette-tube-drive", 1.0, "vega_wuji_pipette_tube_n1d7"),
    ],
    "vega_wuji_pipette_tube_phase_n1d7": [
        ("g1-pipette-tube-drive-phase-labeled", 1.0, "vega_wuji_pipette_tube_n1d7"),
    ],
    "vega_wuji_pipette_grasp_phase00_01_n1d7": [
        ("g1-pipette-tube-drive-phase00-01-global", 1.0, "vega_wuji_pipette_tube_n1d7"),
    ],
    "vega_wuji_pipette_grasp_phase00_01_augmented_n1d7": [
        ("g1-pipette-tube-drive-phase00-01-global", 1.0, "vega_wuji_pipette_tube_n1d7"),
        ("g1-pipette-tube-reference-phase00-01-global", 1.0, "vega_wuji_pipette_tube_n1d7"),
    ],
    "vega_wuji_pipette_grasp_phase00_01_success51_n1d7": [
        ("g1-pipette-tube-drive-phase00-01-global", 1.0, "vega_wuji_pipette_tube_n1d7"),
        (
            "g1-pipette-tube-reference-success42-phase00-01-global", 1.0,
            "vega_wuji_pipette_tube_n1d7",
        ),
    ],
}
