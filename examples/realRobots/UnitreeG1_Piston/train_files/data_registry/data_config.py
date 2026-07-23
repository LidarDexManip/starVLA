"""Unitree G1 (Inspire hands) — piston89 dataset, PI0/PI05 data config.

Dataset: ``piston89_groot_lerobot_v2_multiview60_20260713`` (LeRobot v2.1,
robot_type ``unitree_g1_inspire``, 89 episodes @ 60 fps, single task
"pick up the piston and hold it stably").

Dimensions (from the dataset's meta/info.json + meta/modality.json):

- video : 2 views, 480x640x3 — ``ego_view`` (head) + ``external_right_front``.
- language : ``annotation.human.task_description``.
- state : 41-dim full body [left_leg 6 | right_leg 6 | waist 3 | left_arm 7 |
          left_hand 6 | right_arm 7 | right_hand 6].
- action : 26-dim [left_arm 7 | right_arm 7 | left_hand 6 | right_hand 6]
          (arms + Inspire 6-DoF hands only; no legs / waist in the action).

PI0/PI05 note — pretrained-weight reuse
---------------------------------------
PI0/PI05 pads state to ``framework.max_state_dim`` and action to
``framework.action_dim``; the OpenPI base checkpoints use **32** for both.
- action (26) fits under 32 → keep ``action_dim=32`` and the pretrained
  ``action_in_proj`` / ``action_out_proj`` load unchanged (26 real + 6 zero-pad).
- full state (41) EXCEEDS 32 → it would crash the padding and force
  reinitializing the pretrained ``state_proj``. So the default config below
  feeds **upper-body state only** (waist + arms + hands = 29 ≤ 32), keeping
  ``max_state_dim=32`` and the pretrained state encoder intact — the right
  trade-off for a small (89-episode) fine-tune.

  To instead feed the full 41-dim body state, use ``FULL_STATE_KEYS`` /
  ``FULL_STATE_KEY_DIMS`` below and set ``--framework.max_state_dim 64`` at
  launch (this reinitializes ``state_proj``).
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


# Full 41-dim body state (opt-in; requires framework.max_state_dim >= 41).
FULL_STATE_KEYS = [
    "state.left_leg",
    "state.right_leg",
    "state.waist",
    "state.left_arm",
    "state.left_hand",
    "state.right_arm",
    "state.right_hand",
]
FULL_STATE_KEY_DIMS = {
    "state.left_leg": 6,
    "state.right_leg": 6,
    "state.waist": 3,
    "state.left_arm": 7,
    "state.left_hand": 6,
    "state.right_arm": 7,
    "state.right_hand": 6,
}


class PistonG1InspirePI05DataConfig:
    """Upper-body (29-dim state) config — clean PI05 pretrained-weight reuse."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    # Two camera views. Order is a TRAIN/INFERENCE CONTRACT: the eval client
    # must supply images in exactly this order.
    video_keys = [
        "video.ego_view",
        "video.external_right_front",
    ]

    # Upper body only (waist + arms + hands) = 3+7+6+7+6 = 29 <= max_state_dim 32.
    state_keys = [
        "state.waist",
        "state.left_arm",
        "state.left_hand",
        "state.right_arm",
        "state.right_hand",
    ]
    state_key_dims = {
        "state.waist": 3,
        "state.left_arm": 7,
        "state.left_hand": 6,
        "state.right_arm": 7,
        "state.right_hand": 6,
    }

    # 26-dim action = both arms + both Inspire hands.
    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
    ]
    action_key_dims = {
        "action.left_arm": 7,
        "action.right_arm": 7,
        "action.left_hand": 6,
        "action.right_hand": 6,
    }

    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    state_indices = [0]
    # Number of loaded action steps. Keep this == framework.action_horizon
    # (PI05 default 10) so no action step is zero-padded or truncated.
    action_indices = list(range(10))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        # OpenPI PI0/PI05 convention: q99 normalization on both state and action.
        return ComposedModalityTransform(
            transforms=[
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


class PistonG1InspirePI05FullStateDataConfig(PistonG1InspirePI05DataConfig):
    """Full 41-dim body state variant. Requires --framework.max_state_dim 64
    (reinitializes the pretrained state_proj)."""

    state_keys = FULL_STATE_KEYS
    state_key_dims = FULL_STATE_KEY_DIMS

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
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


class PistonPickPlaceG1PI05DataConfig(PistonG1InspirePI05DataConfig):
    """HF dataset ``birbirll/g1-inspire-piston-pick-place`` (LeRobot v2.1,
    102 episodes / 27,143 frames @ 50 fps, task "pick up the piston.").

    Differs from piston89: single ego view (240x424), and the action space
    includes the WBC locomotion channels — 30 dims total:
    [left_arm 7 | right_arm 7 | left_hand 6 | right_hand 6 |
     base_height 1 | navigate_command 3]  (fits PI05 action_dim=32).
    State is the 29-dim upper body (same groups as piston89, dataset order).
    """

    video_keys = ["video.ego_view"]

    # Order mirrors the dataset's modality.json group order.
    state_keys = [
        "state.left_arm",
        "state.right_arm",
        "state.left_hand",
        "state.right_hand",
        "state.waist",
    ]
    state_key_dims = {
        "state.left_arm": 7,
        "state.right_arm": 7,
        "state.left_hand": 6,
        "state.right_hand": 6,
        "state.waist": 3,
    }

    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.left_hand",
        "action.right_hand",
        "action.base_height",
        "action.navigate_command",
    ]
    action_key_dims = {
        "action.left_arm": 7,
        "action.right_arm": 7,
        "action.left_hand": 6,
        "action.right_hand": 6,
        "action.base_height": 1,
        "action.navigate_command": 3,
    }


class PistonPickPlaceG1OFTDataConfig(PistonPickPlaceG1PI05DataConfig):
    """QwenOFT recipe for the same pickplace dataset (see HF model card
    ``birbirll/g1-inspire-piston-starvla-oft``): MLP L1-regression head,
    vision+language-only (``include_state: false`` in the YAML — state_keys
    stay registered so stats exist, the loader just doesn't pack them).

    Differs from the PI05 variant:
      - action horizon 30 (== framework.action_model.action_horizon; the MLP
        head regresses the full 30-step chunk in one shot)
      - train-time color jitter on the ego view (no-op at eval, see
        VideoColorJitter.get_transform) to stop the policy latching onto
        exact colors/lighting — the sim-transfer failure mode.
    """

    # QwenOFT predicts t .. t+29. Keep == framework.action_model.action_horizon.
    action_indices = list(range(30))

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                # VideoToTensor/VideoToNumpy bracket the jitter so the sample
                # re-emerges as numpy for the downstream Qwen image processor.
                VideoToTensor(apply_to=self.video_keys),
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


class PistonPickPlaceG1GR00TN1d7DataConfig(PistonPickPlaceG1OFTDataConfig):
    """GR00T-N1.7 fine-tune recipe (framework ``CosmosGR00TN1d7``) on the same
    pickplace dataset.

    Identical pipeline to the QwenOFT config (30-step action chunk, ego-view
    color jitter, q99 norm on both state and action) with ONE deliberate
    difference in how it is consumed: N1.7 is a flow-matching policy that
    conditions on proprio state, so launch it with ``include_state: true``
    (the OFT recipe used ``false``). The 29-dim upper-body state and 30-dim
    action are padded to the head's 132-wide multi-embodiment projectors at
    runtime by the framework; q99-normalised values keep train/serve consistent
    with starVLA's un-normalisation contract.
    """


ROBOT_TYPE_CONFIG_MAP = {
    "unitree_g1_inspire_piston": PistonG1InspirePI05DataConfig(),
    "unitree_g1_inspire_piston_fullstate": PistonG1InspirePI05FullStateDataConfig(),
    "unitree_g1_inspire_piston_pickplace": PistonPickPlaceG1PI05DataConfig(),
    # QwenOFT recipe — name matches the reference HF checkpoint's config so
    # its config.yaml / data_mix resolve unchanged at train AND serve time.
    "unitree_g1_piston": PistonPickPlaceG1OFTDataConfig(),
    # GR00T-N1.7 recipe (Cosmos-Reason2 backbone + flow-matching DiT head).
    "unitree_g1_piston_n1d7": PistonPickPlaceG1GR00TN1d7DataConfig(),
}


# mixture_name -> [(dataset_folder_under_data_root_dir, weight, robot_type)].
# The dataset folder name is what sits under --datasets.vla_data.data_root_dir.
DATASET_NAMED_MIXTURES = {
    "piston_g1_inspire": [
        ("piston89_groot_lerobot_v2_multiview60_20260713", 1.0, "unitree_g1_inspire_piston"),
    ],
    "piston_g1_inspire_fullstate": [
        ("piston89_groot_lerobot_v2_multiview60_20260713", 1.0, "unitree_g1_inspire_piston_fullstate"),
    ],
    # HF: huggingface-cli download birbirll/g1-inspire-piston-pick-place \
    #       --repo-type dataset --local-dir <DATA_ROOT>/g1-inspire-piston-pick-place
    "piston_pickplace_g1": [
        ("g1-inspire-piston-pick-place", 1.0, "unitree_g1_inspire_piston_pickplace"),
    ],
    # QwenOFT recipe mixture (same dataset, horizon-30 + color-jitter config).
    # Mixture name matches birbirll/g1-inspire-piston-starvla-oft's config.yaml.
    "unitree_g1_piston_pnp": [
        ("g1-inspire-piston-pick-place", 1.0, "unitree_g1_piston"),
    ],
    # GR00T-N1.7 recipe mixture (same dataset; consumed with include_state:true).
    "unitree_g1_piston_pnp_n1d7": [
        ("g1-inspire-piston-pick-place", 1.0, "unitree_g1_piston_n1d7"),
    ],
}
