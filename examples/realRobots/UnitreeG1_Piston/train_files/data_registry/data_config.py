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
    VideoResize,
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


class PistonWristCamG1GR00TN1d7DataConfig(PistonPickPlaceG1GR00TN1d7DataConfig):
    """THREE-camera long-horizon variant — HF dataset
    ``birbirll/g1-inspire-piston-longhorizon-v3-wristcam`` (LeRobot v2.1, 67
    episodes / 45,943 frames @ 50 fps).

    Same 29-dim state / 30-dim action / instruction as
    ``g1-inspire-piston-longhorizon``; what is new are two wrist streams:

      ego_view          240x424   (head D435)
      left_wrist_view   480x640
      right_wrist_view  480x640

    The order below is a TRAIN/INFERENCE CONTRACT — the eval client must send
    images in exactly this order, because nothing downstream identifies a view
    by name: the dataloader concatenates them positionally into ``ex["image"]``
    and the Cosmos processor emits one 64-token block per image in list order.

    Resolution is NOT a contract: ``LeRobotSingleDataset._pack_sample`` resizes
    every view to 224x224, and the Cosmos processor's ``shortest_edge`` (65536
    px) floor then upscales that to a 16x16 patch grid — so each view costs 64
    image tokens regardless of its native size, and 3 views cost 192 (vs 64).
    """

    video_keys = [
        "video.ego_view",
        "video.left_wrist_view",
        "video.right_wrist_view",
    ]

    # Common size every view is resampled to before the views are stacked. It
    # matches what LeRobotSingleDataset._pack_sample resizes to anyway, so the
    # frame is only resampled once.
    view_resize_hw = (224, 224)

    def transform(self):
        """Same recipe as the single-view N1.7 config, restructured for views of
        DIFFERENT resolutions.

        ``VideoTransform.apply`` concatenates every key in ``apply_to`` along the
        frame axis before running the underlying op, so a shared transform
        requires all of its views to have identical H x W. Here they do not --
        ego_view is 240x424 while both wrist views are 480x640 -- which fails in
        ``np.concatenate`` with "all the input array dimensions except for the
        concatenation axis must match exactly".

        So the two DETERMINISTIC transforms get one instance per view (a
        single-key concat is a no-op) and bring every view to a common size;
        from VideoColorJitter onward the shapes match and the views share a
        transform again. Two ordering constraints are load-bearing:

          * Resize must come AFTER VideoToTensor -- ``VideoToTensor.check_input``
            asserts the incoming resolution still equals the dataset metadata's,
            so resizing first trips that assert.
          * VideoColorJitter must stay SHARED across the three views. The concat
            is what makes a single ``T.ColorJitter`` parameter draw cover every
            view; splitting it per view would give each camera an independently
            jittered colour, which is a different augmentation.
        """
        h, w = self.view_resize_hw
        per_view = []
        for key in self.video_keys:
            per_view.append(VideoToTensor(apply_to=[key]))
            per_view.append(VideoResize(apply_to=[key], height=h, width=w, interpolation="linear"))
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
    "unitree_g1_inspire_piston": PistonG1InspirePI05DataConfig(),
    "unitree_g1_inspire_piston_fullstate": PistonG1InspirePI05FullStateDataConfig(),
    "unitree_g1_inspire_piston_pickplace": PistonPickPlaceG1PI05DataConfig(),
    # QwenOFT recipe — name matches the reference HF checkpoint's config so
    # its config.yaml / data_mix resolve unchanged at train AND serve time.
    "unitree_g1_piston": PistonPickPlaceG1OFTDataConfig(),
    # GR00T-N1.7 recipe (Cosmos-Reason2 backbone + flow-matching DiT head).
    "unitree_g1_piston_n1d7": PistonPickPlaceG1GR00TN1d7DataConfig(),
    # Same, plus the two wrist views (3 camera streams into the VLM backbone).
    "unitree_g1_piston_wristcam_n1d7": PistonWristCamG1GR00TN1d7DataConfig(),
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
    # LONG-HORIZON piston task, GR00T-N1.7 recipe.
    #   HF: hf download birbirll/g1-inspire-piston-longhorizon --repo-type dataset \
    #         --local-dir <DATA_ROOT>/g1-inspire-piston-longhorizon
    # 67 episodes / 45,938 frames @ 50 fps, ego_view 240x424 — the SAME modality
    # layout as g1-inspire-piston-pick-place (29-dim state, 30-dim action, same
    # group order), so it reuses the `unitree_g1_piston_n1d7` DataConfig unchanged.
    # What differs is the TASK: a 3-stage instruction ("pick up the piston with the
    # right hand, inject it into the tube held by the left hand, then move it over
    # the hole plate") with ~687-frame (13.7 s) episodes vs ~266 (5.3 s) before.
    "unitree_g1_piston_longhorizon_n1d7": [
        ("g1-inspire-piston-longhorizon", 1.0, "unitree_g1_piston_n1d7"),
    ],
    # THREE-camera long-horizon (ego + both wrists) — see
    # PistonWristCamG1GR00TN1d7DataConfig for the view-order contract.
    #   HF: hf download birbirll/g1-inspire-piston-longhorizon-v3-wristcam \
    #         --repo-type dataset \
    #         --local-dir <DATA_ROOT>/g1-inspire-piston-longhorizon-v3-wristcam
    # DEPLOY COUPLING: a policy trained on this mixture REQUIRES all three
    # streams at eval. The Isaac piston env currently has its wrist cameras
    # commented out (they were disabled because rendering three views dropped
    # the sim to ~9% real-time), so they must be re-enabled before this
    # checkpoint can be served.
    "unitree_g1_piston_wristcam_n1d7": [
        ("g1-inspire-piston-longhorizon-v3-wristcam", 1.0, "unitree_g1_piston_wristcam_n1d7"),
    ],
}
