"""Unitree G1 (Inspire hands) — pipette tip attach/eject, GR00T-N1.7 data config.

Dataset: ``jren313/g1-pipette-tip-teleop`` (LeRobot v2.1, 64 episodes /
56,742 frames @ **60 fps**, single ego view 720x1280, converted from a hardware
teleop recording by VLAPolicyBridge's ``convert_g1_recording_to_lerobot.py``).

Task: "Attach a red pipette tip from the tip rack to the pipette, then eject the
used tip into the tip disposal box."

    hf download jren313/g1-pipette-tip-teleop --repo-type dataset \\
        --local-dir <DATA_ROOT>/g1-pipette-tip-teleop

Why this is NOT just a copy of the piston config
------------------------------------------------
The piston configs feed a 29-dim state / 30-dim action built from
``arms | hands | base_height | navigate_command``. Reusing that shape verbatim
here would train on channels that are provably dead in this recording, and —
worse — would feed several of them to the model **un-normalised**.

``StateActionTransform`` (transform/state_action.py) masks out any dim where
``q01 == q99`` and *passes the original value through*, because the q99 formula
is undefined there. That is harmless when the constant is small (the piston pnp
mix has one such dim, constant at -0.1). It is NOT harmless for Inspire hand
channels, whose raw register units are 0..1000: a constant hand dim arrives at
the flow-matching head as a literal 500 or 1000 while every other dim is in
[-1, 1], and the loss on that one channel then dwarfs the whole rest of the
action vector.

Measured over all 56,742 frames of this dataset (q01/q99, the exact statistics
the transform uses):

    state.waist / left_arm / right_arm     0 degenerate dims       -> clean q99
    state.hand_left                        1/6 degenerate @ 998    -> EXCLUDED
    state.hand_right                       1/6 degenerate @ 60     -> EXCLUDED
    action.left_arm / right_arm            0 degenerate            -> clean q99
    action.root_height / lin_vel / yaw     0 degenerate            -> clean q99
    action.hand_left                       6/6 degenerate @500,1000-> EXCLUDED
    action.hand_right                      4/6 degenerate @ 0      -> min_max

So:

* **action.hand_left is dropped entirely.** It is constant across every frame of
  every episode (the left hand was never commanded in this one-handed task), so
  there is nothing to learn and six raw-1000 channels to lose the loss to.
* **action.hand_right is normalised ``min_max``, not ``q99``.** Its four finger
  dims sit at 0 (closed — UPPER=OPEN on Inspire) for 99.8% of frames, so
  q01 == q99 == 0 and q99-normalisation would pass them through raw. min_max
  maps the true 0..1000 span onto [-1, 1] and nothing escapes un-normalised.
  The informative right-hand channels are ``thumb_bend`` (the ejector press)
  and ``thumb_rot``; the fingers fire in well under 1% of frames and are kept
  only so the action vector stays a complete hand command.
* **Both hand STATE groups are dropped.** Besides the degenerate dim, the
  recorder never refreshed right-hand feedback: ``state.hand_right`` is constant
  *within* each episode but differs *between* episodes, which is an episode
  fingerprint the policy could latch onto and which carries no information at
  serve time. See the dataset card's Caveats table.

Resulting spaces (both fully normalised, no passthrough):

    state  17 = waist 3 | left_arm 7 | right_arm 7
    action 24 = left_arm 7 | right_arm 7 | hand_right 6 | root_height 1 |
                root_lin_vel_xy 2 | root_yaw_rate 1

Action dim indices, for ``grasp_weight_dim`` and for the eval client:

    0:7   left_arm        14:20  hand_right  (18 = thumb_bend, 19 = thumb_rot)
    7:14  right_arm       20:21  root_height
                          21:23  root_lin_vel_xy   23:24  root_yaw_rate

``root_lin_vel_xy`` + ``root_yaw_rate`` together are the 3-dim navigate command
in the same (vx, vy, yaw_rate) order the piston action used, so the
``embodiment_id=25`` slot (``unitree_g1_full_body_with_waist_height_nav_cmd``)
still warm-starts the arms | hand | height | nav category rows.

Frame rate: this dataset is **60 fps** where the piston datasets are 50, so a
30-step chunk covers 0.5 s here vs 0.6 s there. Keep that in mind when setting
the eval client's replan cadence.
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


class PipetteTipG1GR00TN1d7DataConfig:
    """GR00T-N1.7 (framework ``CosmosGR00TN1d7``) fine-tune on the pipette task."""

    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT

    # Single head camera. The key is `rgb` because that is what this dataset's
    # meta/modality.json calls it (the piston datasets say `ego_view`); nothing
    # downstream identifies a view by name, views are consumed positionally.
    video_keys = ["video.rgb"]

    state_keys = [
        "state.waist",
        "state.left_arm",
        "state.right_arm",
    ]
    state_key_dims = {
        "state.waist": 3,
        "state.left_arm": 7,
        "state.right_arm": 7,
    }

    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.hand_right",
        "action.root_height",
        "action.root_lin_vel_xy",
        "action.root_yaw_rate",
    ]
    action_key_dims = {
        "action.left_arm": 7,
        "action.right_arm": 7,
        "action.hand_right": 6,
        "action.root_height": 1,
        "action.root_lin_vel_xy": 2,
        "action.root_yaw_rate": 1,
    }

    # q99 everywhere except the Inspire hand, which is bimodal 0/1000 and would
    # come out of q99 un-normalised (see the module docstring).
    action_normalization_modes = {
        "action.left_arm": "q99",
        "action.right_arm": "q99",
        "action.hand_right": "min_max",
        "action.root_height": "q99",
        "action.root_lin_vel_xy": "q99",
        "action.root_yaw_rate": "q99",
    }

    language_keys = ["annotation.human.task_description"]

    observation_indices = [0]
    state_indices = [0]
    # == framework.action_model.action_horizon.
    action_indices = list(range(30))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                # Jitter is bracketed by tensor/numpy conversions so the sample
                # re-emerges as numpy for the Cosmos image processor. It is a
                # no-op at eval (see VideoColorJitter.get_transform) and is here
                # for the same reason as the piston recipe: one fixed lab scene
                # under one lighting setup invites the policy to memorise exact
                # colour rather than shape.
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
                    normalization_modes=dict(self.action_normalization_modes),
                ),
            ]
        )


class PipetteTipG1ArmHandOnlyDataConfig(PipetteTipG1GR00TN1d7DataConfig):
    """20-dim action variant: arms + right hand, no base/nav channels.

    The four locomotion channels kept by the default config are, in this
    recording, jitter around a fixed stance -- q99-q01 is 0.0075 m for
    root_height and 0.06-0.08 m/s for root_lin_vel_xy. q99 normalisation divides
    by that tiny span, so pure sensor noise is stretched to |value| ~ 2.2 in the
    training target: the policy is asked to predict noise, and the flow-matching
    loss pays attention to it.

    Dropping them is the right call whenever the robot is meant to stand still
    (which it is for this task -- you do not want a VLA emitting base-velocity
    commands next to a tip rack). Keep the default 24-dim config instead if you
    want the action vector to stay shape-compatible with the bridge's canonical
    ``base_height`` / ``navigate_cmd`` keys; the eval client can otherwise fill
    base_height with the dataset constant and navigate_cmd with zeros.

    Action dims: 0:7 left_arm | 7:14 right_arm | 14:20 hand_right
    (so grasp_weight_dim is still 18 == thumb_bend).

    Launch with:
        DATA_MIX=unitree_g1_pipette_armhand_n1d7 \\
        ... --framework.action_model.action_dim 20
    """

    action_keys = [
        "action.left_arm",
        "action.right_arm",
        "action.hand_right",
    ]
    action_key_dims = {
        "action.left_arm": 7,
        "action.right_arm": 7,
        "action.hand_right": 6,
    }
    action_normalization_modes = {
        "action.left_arm": "q99",
        "action.right_arm": "q99",
        "action.hand_right": "min_max",
    }


ROBOT_TYPE_CONFIG_MAP = {
    "unitree_g1_pipette_n1d7": PipetteTipG1GR00TN1d7DataConfig(),
    "unitree_g1_pipette_armhand_n1d7": PipetteTipG1ArmHandOnlyDataConfig(),
}


DATASET_NAMED_MIXTURES = {
    #   hf download jren313/g1-pipette-tip-teleop --repo-type dataset \
    #     --local-dir <DATA_ROOT>/g1-pipette-tip-teleop
    "unitree_g1_pipette_n1d7": [
        ("g1-pipette-tip-teleop", 1.0, "unitree_g1_pipette_n1d7"),
    ],
    # Same data, 20-dim action (no base/nav). Needs action_dim 20 at launch.
    "unitree_g1_pipette_armhand_n1d7": [
        ("g1-pipette-tip-teleop", 1.0, "unitree_g1_pipette_armhand_n1d7"),
    ],
}
