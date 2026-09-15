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

import random

import numpy as np

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.transform.base import (
    ComposedModalityTransform,
    ModalityTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from starVLA.dataloader.gr00t_lerobot.transform.video import (
    VideoCenterCrop,
    VideoColorJitter,
    VideoColorTemperature,
    VideoCrop,
    VideoRandomRotation,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)



class VisionLagJitter(ModalityTransform):
    """Make the CAMERA lag the PROPRIOCEPTION by a random, mild amount.

    THE DEPLOY ASYMMETRY THIS MODELS. On the robot the two observation streams
    do not arrive together. State comes from the whole-body host over a local
    wire and is nearly current; the image is captured on the Orin, JPEG-encoded,
    pushed across the G1 network and decoded in the bridge. Training pairs them
    as simultaneous, which is a lie the policy has no way to detect.

    Worse, nobody knows the size of the lie. ``obs_latency_s`` in the bridge
    sounds like it measures frame age but is ``time.monotonic()`` around the
    ``get_action`` call -- INFERENCE time. The camera wire carries no capture
    timestamp at all, which is how a ~10 s backlog went unnoticed until
    2026-08-16.

    HOW. Video stays pinned at delta 0 and everything else slides FORWARD by a
    random k: state is read at ``[-36+k, -24+k, -12+k, k]`` and the action
    window at ``[k, k+30)``. Relative to the frame the model sees, state and
    actions are then k steps in the future -- which is the same thing as the
    frame being k steps stale, with state still current. Sliding the cheap
    modalities instead of the video is what keeps this free: state and actions
    are a few hundred floats, whereas randomising the VIDEO index would
    multiply the decode cost of every sample.

    k is 0 in eval mode, so open-loop scoring still compares against aligned
    ground truth and stays comparable with the earlier rounds.

    THE RANGE IS A GUESS AND IS DELIBERATELY MILD. 0-6 rows = 0-100 ms at the
    measured 59.6 Hz. Wide augmentation over an unmeasured latency would blur
    the timing of a task whose whole difficulty is landing a pipette on a
    few-millimetre tip. Put a capture timestamp in the streamer header, log the
    real distribution, and set this from its p95 instead of from this comment.
    """

    state_keys: list[str]
    state_picks: list[int]
    horizon: int
    max_lag: int

    def apply(self, data: dict) -> dict:
        lag = random.randint(0, self.max_lag) if self.training else 0
        # ONE k for state and actions alike. Drawing separately would put the
        # proprioception and the actions on different clocks and teach a
        # correspondence that exists on no robot.
        for key in self.state_keys:
            if key in data:
                data[key] = data[key][[i + lag for i in self.state_picks]]
        for key in self.apply_to:
            if key in data:
                data[key] = data[key][lag:lag + self.horizon]
        return data


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


class PipetteTipCropG1GR00TN1d7DataConfig(PipetteTipG1GR00TN1d7DataConfig):
    """TWO-view variant — HF dataset ``birbirll/g1-pipette-tip-teleop-tipcrop``.

    Same 64 episodes / 56,742 frames @ 60 fps and the SAME state/action parquet
    bytes as ``g1-pipette-tip-teleop``; the only delta is a second video stream.
    Verified against this dataset's own ``meta/stats.json``: the degenerate-dim
    census in the module docstring reproduces exactly (hand_left action 6/6
    constant, hand_right 4/6 at 0, every kept group 0/N), so the parent's
    key selection and ``action_normalization_modes`` carry over untouched and
    nothing reaches the head un-normalised.

    Why a second view at all
    ------------------------
    The thing being manipulated — a red pipette tip in a rack — is 2-4 px in the
    720x1280 ego frame. After the pipeline resizes that frame to 224x224 it is
    sub-pixel: the information the task turns on is simply not in the tensor. The
    crop stream is a detector-tracked 256x256 window (280 px source, ~4.6x zoom)
    centred on the rack and biased 10% upward for the descending pipette, so the
    tip survives the resize at usable scale.

        video.rgb        720x1280   ego, WITH a 2 px green crop-box overlay
        video.tip_crop   256x256    detector-tracked zoom on the tip rack

    DEPLOY CONTRACT — two halves, both easy to get wrong:

      1. Order is ``rgb`` then ``tip_crop``. View identity is positional (the
         dataloader concatenates in list order and the Cosmos processor emits one
         64-token block per image in that order); nothing downstream reads a
         name, so a swapped pair is misread SILENTLY.
      2. The ego stream is not the raw camera. Its frames carry the 2 px green
         rectangle marking where the crop was taken, so the eval client must draw
         the same box on the live ego frame AND run the same HSV rack detector /
         EMA smoothing to produce ``tip_crop``. Serving a clean ego frame, or a
         crop from a different tracker, is a distribution shift on both views at
         once. See the dataset card's "How the crop is chosen".

    ``meta/relative_stats.json`` ships empty (``{}``) here, and starVLA never
    reads it — that trap belongs to the GR00T-N1.6 trainer, not this path.
    """

    video_keys = [
        "video.rgb",
        "video.tip_crop",
    ]

    # Common size every view is resampled to before the views are stacked.
    # Matches what LeRobotSingleDataset._pack_sample resizes to anyway, so a
    # frame is only resampled once. The 256 -> 224 downscale on tip_crop costs
    # nothing in tokens: the Cosmos processor's shortest_edge floor maps 224 and
    # 256 alike onto a 16x16 patch grid, i.e. 64 tokens per view either way.
    view_resize_hw = (224, 224)

    #: Optional PER-VIEW override, {video_key: (h, w)}. Views absent here fall
    #: back to ``view_resize_hw``. The Cosmos processor emits an independent
    #: token block per image, so views do not have to share a size -- and on a
    #: rig where one camera carries the task-critical detail, spending the
    #: tokens only there is far cheaper than raising every view.
    view_resize_overrides: dict = {}

    def resize_for(self, key):
        """(h, w) this view is resampled to before the views are stacked."""
        return tuple(self.view_resize_overrides.get(key, self.view_resize_hw))

    def transform(self):
        """Parent recipe, restructured for views of DIFFERENT resolutions.

        ``VideoTransform.apply`` concatenates every key in ``apply_to`` along the
        frame axis before running the underlying op, so a shared transform needs
        its views to agree on H x W. Here they do not -- rgb is 720x1280 and
        tip_crop is 256x256 -- which fails inside ``np.concatenate``.

        So the DETERMINISTIC transforms get one instance per view (a single-key
        concat is a no-op) and bring both views to a common size; from
        VideoColorJitter onward the shapes match and the views share a transform
        again. Two ordering constraints are load-bearing:

          * Resize must come AFTER VideoToTensor -- ``VideoToTensor.check_input``
            asserts the incoming resolution still equals the dataset metadata's,
            so resizing first trips that assert.
          * VideoColorJitter must stay SHARED. The concat is what makes one
            ``T.ColorJitter`` parameter draw cover both views; splitting it would
            jitter the ego frame and its own crop to different colours, teaching
            the model that the two views disagree about the scene.
        """
        per_view = []
        for key in self.video_keys:
            h, w = self.resize_for(key)
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
                # WHITE BALANCE — the axis this bench actually drifts on, and
                # the one ColorJitter cannot model: its `hue` rotates the HSV
                # wheel uniformly, a camera applies per-channel GAIN.
                # Measured 2026-08-13, scene median hue was 68 in the training
                # data, 93 on the same bench hours later under changed room
                # lighting, and 13 with white balance pinned to 5800K — and in
                # that last case the policy drove nowhere near the rack.
                # strength 0.15 = up to 0.302 in log(R/B), against this
                # dataset's OWN spread of sd 0.047 / full range 0.222 across
                # its 49 episodes: comfortably wider than what the model has
                # already seen. Calibrated by tools/calibrate_wb_aug.py.
                # CHOSEN OVER widening ColorJitter's hue because the task is
                # "attach a RED tip": the rotation needed (±0.14) turns red
                # toward orange or magenta, while a channel gain leaves it red
                # — verified across t=±0.20, the median hue of red pixels
                # stays 8–12.
                # SHARED across views, same reason as ColorJitter: one scene,
                # one sensor, one draw.
                VideoColorTemperature(
                    apply_to=self.video_keys,
                    strength=0.15,
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


class PipetteTipCropWaistG1GR00TN1d7DataConfig(PipetteTipCropG1GR00TN1d7DataConfig):
    """Round-3: the two-view tipcrop action space PLUS the waist.

    Why the waist had to come back
    ------------------------------
    Round 2 (``unitree_g1_pipette_tipcrop_n1d7``) dropped ``action.waist``, and
    that turned out to be a deployment bug rather than a simplification. Arm
    actions are ABSOLUTE joint targets, so a waist the policy cannot command is
    a waist frozen at the ready pose while the arms replay a trajectory the
    operator performed with the torso turning. Measured over 20 episodes, the
    within-episode drift from each episode's own frame 0 is:

        yaw    median 6.6 deg   worst 13.8 deg
        roll   median 2.4 deg   worst  4.9 deg
        pitch  median 2.2 deg   worst  5.0 deg

    6.6 deg of yaw at a 0.45 m reach puts the hand ~52 mm off where the demo
    put it (110 mm in the worst episode). A pipette tip is a few mm. The ego
    camera is head-mounted too, so a frozen waist also shifts the VIEW away
    from anything in the training distribution -- and because the serving path
    runs ``--no_state``, the model cannot even sense that the waist is wrong.

    ``action.waist`` is APPENDED, not inserted, so every dim index documented
    against round 2 stays valid:

        0:7   left_arm      14:20  hand_right   21:23  root_lin_vel_xy
        7:14  right_arm     20:21  root_height  23:24  root_yaw_rate
                                                24:27  waist  (NEW)

    q99 like the arms: this dataset's ``action.waist`` has 0 degenerate dims
    (verified against meta/stats.json), so nothing reaches the head
    un-normalised.

    Launch with ``--framework.action_model.action_dim 27``.
    """

    action_keys = PipetteTipCropG1GR00TN1d7DataConfig.action_keys + ["action.waist"]
    action_key_dims = {**PipetteTipCropG1GR00TN1d7DataConfig.action_key_dims,
                       "action.waist": 3}
    action_normalization_modes = {
        **PipetteTipCropG1GR00TN1d7DataConfig.action_normalization_modes,
        "action.waist": "q99",
    }


class PipetteTipCropWaistHistG1GR00TN1d7DataConfig(
        PipetteTipCropWaistG1GR00TN1d7DataConfig):
    """Round 4: the 27-dim waist recipe + STATE HISTORY + mild vision lag.

    WHY. Rounds 1-3 are memoryless: video and state delta_indices are both [0]
    and the server runs --no_state, so the entire input is one ego frame, one
    tip crop and a sentence. The task is aliased under that input. Measured
    over 8 episodes / 1288 sampled frames (both model views, 48x48 normalised
    gray): a frame's nearest neighbour MORE THAN 2 s AWAY is typically closer
    (image distance p25 0.379) than two consecutive samples 83 ms apart (p50
    0.207), and among the closest quarter of those pairs 68% have OPPOSED
    next-chunk arm directions, median cosine -0.31. Descending onto the tip and
    rising after it look the same. The demonstrations disagree with each other,
    and flow matching averages the disagreement into a hold. Open loop hides it
    because dataset frames advance regardless of what the arm does; closed loop
    turns it into an absorbing state -- the observed "barely moves toward the
    tip".

    Joint POSITION alone does not break the tie: the arm is at nearly the same
    configuration going down and coming up. VELOCITY does, and that is what a
    state history is.

    NOTE THIS IS A DEPARTURE FROM UPSTREAM. nvidia/GR00T-N1.7-3B ships
    state_encoder.layer1.W of shape (32, 132, 1024) -- max_state_dim x ONE
    frame; there is no state_history_length field in its config at all, and
    starVLA's PI0/PI05 path explicitly drops all but the newest state frame
    (PI0.py: ``state_np = state_np[-1:]``). Neither reference design carries
    state history. The pretrained weights therefore offer nothing for the extra
    frames, which is exactly why the widening is warm-started rather than
    reinitialised (CosmosGR00T_N1d7._widen_state_encoder).

    WHAT TRAVELS WITH THIS CHECKPOINT AT DEPLOY TIME:

    1. SERVE WITHOUT --no_state, or the whole thing is wasted. That flag was
       never a judgement that state is useless -- upstream ships
       ``exclude_state = False``. It was the only safe option while
       PolicyNormProcessor had no ``apply_state`` and the server sent RAW
       radians to a head trained on q99-normalised ones. apply_state exists now
       and gr00t_obs_adapter calls it. (Worth knowing what --no_state actually
       does: it sends an all-ZERO state tensor, and normalised zero is a
       specific mid-range pose -- so the model is told the arm sits at a fixed
       posture on every tick, which is a confident lie rather than a shrug.)

    2. RUN AT REAL TIME: action_hz 60, not the lane's current 30. A memoryless
       policy is invariant to time scaling; a velocity-conditioned one is not.
       At action_hz 30 against a 60 fps recording the arm covers half the
       trained distance per unit wall time, so every history frame arrives at
       half its trained magnitude. For replay also set inference_hz 2.5, so
       inference_hz * advance_per_call stays equal to action_hz.

    3. The bridge buffer is indexed in STATE MESSAGE ticks (~50 Hz), not
       dataset rows, so the profile spans the same 0.6 s as
       ``[-30, -20, -10, 0]`` -- do not copy the numbers below.
    """

    # 0.6 s of history in 4 frames, measured against this dataset's true rate
    # (59.6 Hz from the t_img column, NOT the 60 in meta/info.json):
    # 36 rows = 604 ms, so the picks sit at -0.60, -0.40, -0.20, 0 s.
    # Over 36 rows the right arm moves a median 0.024 rad -- ~1.4 deg, well
    # clear of encoder noise, so direction of travel is recoverable.
    HISTORY_PICKS = [-36, -24, -12, 0]
    MAX_LAG = 6            # rows; 6 / 59.6 Hz = 101 ms of vision lag
    ACTION_HORIZON = 30    # == framework.action_model.action_horizon

    # SUPERSETS. The transform slides a window over these, so the loader must
    # fetch every row the window can land on: state from the oldest pick back
    # through MAX_LAG into the FUTURE, actions one horizon plus MAX_LAG ahead.
    # The model still receives exactly 4 state frames and 30 actions.
    state_indices = list(range(HISTORY_PICKS[0], MAX_LAG + 1))
    action_indices = list(range(ACTION_HORIZON + MAX_LAG))

    def transform(self):
        base = super().transform()
        offset = -self.HISTORY_PICKS[0]      # row 0 of the loaded state block
        return ComposedModalityTransform(
            transforms=list(base.transforms) + [
                # LAST in the pipeline. Normalisation is element-wise, so
                # slicing after it is identical to slicing before, and every
                # earlier transform still sees the full window it expects.
                VisionLagJitter(
                    apply_to=list(self.action_keys),
                    state_keys=list(self.state_keys),
                    state_picks=[i + offset for i in self.HISTORY_PICKS],
                    horizon=self.ACTION_HORIZON,
                    max_lag=self.MAX_LAG,
                ),
            ]
        )


class PipetteTipCropWaistHistPhaseG1GR00TN1d7DataConfig(
        PipetteTipCropWaistHistG1GR00TN1d7DataConfig):
    """Round 5: round 4 + PER-FRAME PHASE SENTENCES instead of one task string.

    Rounds 1-4 serve the same sentence on every tick of every episode, so
    language carries no information about WHERE in the task the robot is --
    the same blind spot as the single video frame and (before round 4) the
    single state frame. This config swaps the language key for
    ``annotation.human.phase``, a per-frame column written by
    phase_labels.label_episode:

        P1  Move the pipette left until it is above the tip rack.
        P2  Aim at the red tip in the rack and lower the pipette onto it.
        P3  The tip is attached. Lift the pipette clear of the rack, carry it
            to the clear box on the right, and press the ejector with the
            thumb.

    THREE phases, not four. The original P3 (lift) and P4 (carry + eject) were
    merged on 2026-08-18. The lift ran a median 0.95 s -- 6.4% of frames --
    and it was the one boundary set by an arbitrary knob rather than an
    extremum: `bottom` is an argmin and `eject` an argmax, but `lift` was
    "z climbed back through Z_LIFT=0.50 of its recovery", and sweeping that
    0.40->0.60 moves the cut a median 0.26 s, i.e. 27% of the phase it bounds.
    A label that uncertain over that few frames teaches little and costs a
    serving-side transition to detect. The merge also balances the vocabulary:
    14.6 / 41.8 / 43.5% of frames.

    P2 and P3 are still the pair the aliasing measurement is about: descending
    onto the tip and rising off it put the arm at nearly the same joint
    configuration in front of nearly the same picture, and 68% of look-alike
    frame pairs have OPPOSED next-chunk arm directions. The merge preserves
    that separation -- descending is P2 and rising is P3 either way. A sentence
    separates them with no extra tokens: the language block is already in every
    forward pass, it was just constant.

    DATASET: g1-pipette-tip-teleop-phase (73 episodes -- the 2026-08-13
    capture plus the 13-episode 2026-08-17 one, both carrying the
    detector-tracked tip_crop, less four whose crop locked onto the forearm).
    Phase sentences live at task rows 3-5;
    ``annotation.human.task_description`` is untouched at rows 0-2, so the
    round-4 recipe still trains from this same dataset by simply not naming
    the phase key. meta/splits.json carries train / eval / holdout / new.

    ONE THING TO CHECK BEFORE USING IT:

    - The two captures DISAGREE about the ejector. ``action.hand_right[5]``
      rests at 500 and the press drives it UP across
      all 49 episodes of the older set, and rests at ~956 and drives it DOWN
      across all 13 of the newer one. Training on the union makes that channel
      bimodal. Either hold the captures apart, or normalise the convention, or
      accept that the eject action is the one dim the model cannot learn
      cleanly.

    SERVING: the phase sentence has to come from somewhere at deploy time.
    The bridge already re-reads ``--instruction_file`` on mtime change, which
    is what the piston lane's phase machine writes to.
    """

    language_keys = ["annotation.human.phase"]


class Pipette3ViewWaistHistG1GR00TN1d7DataConfig(
        PipetteTipCropWaistHistG1GR00TN1d7DataConfig):
    """Round 6: round 4's action/state recipe on the THREE-CAMERA capture.

    WHAT CHANGES FROM ROUND 4/5
      * THREE views instead of two, and no manufactured crop. The 2026-08-19
        rig adds two 1920x1080 cameras beside the ego one; both are stored
        centre-cropped to 1080x1080, so every view is a square downscale at
        model input rather than a 1.78x horizontal squash.
      * view_resize_hw 224 -> 256. 224 was never a real setting: the Cosmos
        processor floors the shortest edge at 65536 px (256x256), so a 224
        frame is upscaled back to 256 before tokenising. Both spellings cost
        64 tokens per view; 224 just adds a resample that throws detail away.
      * LANGUAGE IS CONSTANT AGAIN. This capture carries no phase column --
        annotation.human.task_description is one sentence for every frame, as
        in rounds 1-4. Round 5's per-frame phase sentences do NOT apply here.
      * The tip is GREEN in this capture, and the task text says so. The
        2026-08-13/17 captures used red tips.

    TOKEN BUDGET: 3 views x 64 = 192 image tokens, up from 128 for two views.
    Measured VL-encode cost on an RTX 4080 (bf16, select_layer=16): 24 ms for
    three views at 256, against 578 ms if the frames were fed at native
    resolution -- 4960 tokens -- which no closed-loop rate could absorb.

    THE TIP IS SMALL IN THE WRIST VIEWS. Measured green-blob size at 256x256:
    about 6x10 px in the ego view but only 1-3 px in each wrist view, because
    those are 4.2x downscales of a 1080x1080 crop. If a policy trained here
    cannot find the tip, feeding the two wrist views at 384 costs ~10 ms and
    roughly doubles it; the DataConfig would need per-view sizes to do that.

    NO EVAL SPLIT. All 53 episodes train. Nothing is held out, so there is no
    in-distribution score for this run -- judge it on the robot.
    """

    video_keys = [
        "video.rgb",
        "video.wrist_left",
        "video.wrist_right",
    ]
    view_resize_hw = (256, 256)


class Pipette3ViewMixResG1GR00TN1d7DataConfig(
        Pipette3ViewWaistHistG1GR00TN1d7DataConfig):
    """Round 7: round 6 with the two wrist views at 448 instead of 256.

    WHY. Measured on the stored frames, the green tip is about 6x10 px in the
    ego view at 256 but 1-3 px in each wrist view -- small enough that a blob
    detector cannot find it at all. Those two are 4.2x downscales of a
    1080x1080 crop, and they are where the tip alignment actually reads. At 448
    the tip measures 6x11 (wrist_left) and 7x4 (wrist_right), the rack wells
    resolve individually, and "which well is the tip over" becomes legible.

    COST, measured on an RTX 4080 (bf16, select_layer=16):
        all three @ 256          192 tokens   24 ms
        ego 256 + two @ 448      456 tokens   44 ms
    20 ms against a ~56 ms action head, so the server keeps its 11-12 Hz.

    The ego view stays at 256 deliberately: it already carries the tip at a
    usable size, and it is the view whose framing matches the earlier captures.

    THE YAML MUST MATCH. datasets.vla_data.obs_image_size has to carry the same
    three sizes, in video_keys order -- the loader's _pack_sample and the
    server's predict_action both read it, and a mismatch there silently feeds
    the head views at the wrong scale.
    """

    view_resize_overrides = {
        "video.wrist_left": (448, 448),
        "video.wrist_right": (448, 448),
    }

    def transform(self):
        """Parent chain, but the views only diverge in size at the very END.

        VideoColorJitter and VideoColorTemperature are SHARED across views on
        purpose -- one scene, one sensor, one parameter draw -- and
        VideoTransform.apply implements sharing by concatenating the views
        along the frame axis. That concat needs every view at the same H x W,
        so resizing to the per-view targets first makes it raise
        "Sizes of tensors must match except in dimension 0".

        So: bring every view to a COMMON working size (the largest target),
        run the shared colour transforms there, and only then resize each view
        down to its own target. The ego view is resampled twice as a result
        (1280x720 -> 448 -> 256 rather than straight to 256); that is the price
        of keeping one colour draw across the three cameras, which matters
        more -- a model taught that its views disagree about the scene has a
        harder problem than one taught a slightly softer ego frame.
        """
        targets = {k: self.resize_for(k) for k in self.video_keys}
        common = max(max(hw) for hw in targets.values())

        saved = self.view_resize_overrides
        try:
            # parent builds its chain with every view at `common`
            self.view_resize_overrides = {k: (common, common) for k in self.video_keys}
            composed = super().transform()
        finally:
            self.view_resize_overrides = saved

        ts = list(composed.transforms)
        # insert the per-view down-resize just before the views leave tensor
        # form; VideoResize cannot operate on the numpy arrays after this.
        idx = next(i for i, t in enumerate(ts) if type(t).__name__ == "VideoToNumpy")
        tail = []
        for key, (h, w) in targets.items():
            if (h, w) != (common, common):
                tail.append(VideoResize(apply_to=[key], height=h, width=w,
                                        interpolation="linear"))
        # Every VideoTransform over MORE THAN ONE key concatenates them, so
        # once the sizes diverge the shared VideoToNumpy fails for the same
        # reason the colour transforms would. It carries no random state, so
        # splitting it per view costs nothing.
        for key in self.video_keys:
            tail.append(VideoToNumpy(apply_to=[key]))
        ts[idx:idx+1] = tail
        return ComposedModalityTransform(transforms=ts)


class Pipette3ViewMixResNoHistG1GR00TN1d7DataConfig(
        Pipette3ViewMixResG1GR00TN1d7DataConfig):
    """Round 11: round 7's views and data, MEMORYLESS state.

    WHAT CHANGES FROM ROUND 7 (the only three things)
      * STATE HISTORY REMOVED. Rounds 4-10 feed 4 state frames spanning
        0.6 s; this one feeds ONLY the current row. Pair with
        action_model.state_history_length 1 in the yaml -- the loader and the
        head must agree or the state_encoder sees 132*4 where 132 arrived.
      * state_dropout_prob 0.3 -> 0.8 (a yaml knob, not a DataConfig one).
      * Everything else is round 7 byte-for-byte: same p12 truncation, same
        ego 256 + wrists 448, same 27-dim action, same 30-step horizon.

    WHY THE ACTION WINDOW SHRINKS TOO. The parent chain loads a 43-row state
    window (HISTORY_PICKS[0] = -36 through MAX_LAG = +6) and one horizon plus
    MAX_LAG of actions, purely to give VisionLagJitter room to slide. With no
    history there is nothing to slide, so the state block collapses to the
    current row and the action block to exactly ACTION_HORIZON.

    VISIONLAGJITTER IS STRIPPED, NOT DISABLED. It indexes state at
    ``pick + 36`` against a block that is now ONE row tall, so leaving it in
    raises rather than degrading. Removing it also removes round 7's 0-100 ms
    vision-lag augmentation -- a real second difference from round 7, and one
    that cannot be avoided while keeping a single state frame.

    NOTE THIS IS THE MEMORYLESS CONFIGURATION THAT ALIASED BEFORE. The
    2026-08-17 diagnosis on the tipcrop rounds found 68% of near-duplicate
    frame pairs had opposed next-chunk directions with exactly this input
    (rgb_t, instruction, one state frame). The p12 truncation is what makes
    it defensible here: cut at insertion, the descend/lift collision does not
    exist in the data, so there is far less for a memoryless policy to
    confuse. That pairing -- truncated data AND no history -- is the
    experiment.
    """

    state_indices = [0]
    action_indices = list(range(
        Pipette3ViewMixResG1GR00TN1d7DataConfig.ACTION_HORIZON))

    def transform(self):
        composed = super().transform()
        kept = [t for t in composed.transforms
                if type(t).__name__ != "VisionLagJitter"]
        if len(kept) == len(composed.transforms):
            raise RuntimeError(
                "expected a VisionLagJitter in the parent chain to strip; "
                "the parent's transform() changed -- re-check this override")
        return ComposedModalityTransform(transforms=kept)


class Pipette3ViewOFTDataConfig(Pipette3ViewMixResNoHistG1GR00TN1d7DataConfig):
    """QwenOFT recipe on the pipette three-view capture (2026-09-11).

    The SAME data contract as round 11's memoryless GR00T config -- three
    views at ego 256 + wrists 448, the 27-dim action (arms, right hand,
    root height/vel/yaw, waist), a 30-step horizon, colour jitter + q99 --
    consumed by a different head: Qwen3-VL backbone with an MLP L1 head
    that regresses the whole chunk from 30 "🔍" action tokens (framework
    ``QwenOFT``, see the piston OFT config for the head's origin).

    Nothing in the transform chain changes. What differs is how the yaml
    consumes it: ``include_state: false`` (OFT is vision+language-only on
    purpose -- with proprio packed the head fits state and ignores the
    camera in closed loop), so the state keys stay registered here only so
    the q99 stats exist. ``action_model.action_dim`` must equal the 27 dims
    below and ``action_horizon`` the 30-row ``action_indices``.

    Registered under its OWN robot_type so a checkpoint's config resolves
    to this class at serve time (the norm processor reads the name back).
    """


class Pipette3ViewOFTEEDataConfig(PipetteTipG1GR00TN1d7DataConfig):
    """QwenOFT, round 2 (2026-09-12): the pi0.5-hil194 data recipe on hil293.

    Four things change against ``Pipette3ViewOFTDataConfig``, all borrowed
    from the pi0.5 full-BC run on hil194 (PI05_HIL194_TRAINING_SETTINGS.md)
    so the two heads can be compared on the same contract:

    1. ACTION = RIGHT-HAND CARTESIAN DELTA, 3 dims, not 27 joint targets.
       ``action.right_ee_dxyz[t] = FK(cmd[t+1]) - FK(cmd[t])`` in metres per
       60 Hz step, pelvis frame, ``right_rubber_hand`` of the stock Unitree
       URDF, FK of the COMMANDED 29 joints (waist + right arm move it). This
       is pi0.5's ``FK(ref[f+2]) - FK(ref[f])`` at 60 Hz instead of its
       30 Hz even-frame pairs. Precomputed into the ``-ee`` dataset
       (scratchpad build_hil293_ee.py, cross-checked against the bridge's
       UrdfKinematics to 1e-16 m), so the yaml stays ``action_mode: abs`` --
       the loader's own delta mode would make step 0 ``cmd - measured``,
       which is ~20 mm here against ~0.1 mm per-step increments and would
       own the q99. Zero-padded past the episode end (modality
       ``absolute: false``). Measured over the 293 episodes: |d| p50
       0.11 mm, q01/q99 about +-0.5 mm; 34-45% of steps are exact zeros
       because the command stream runs ~50 Hz under a 60 Hz recorder plus
       stationary holds. Kept raw, as pi0.5 kept its reference jumps.
       SERVING has to integrate: p_k = FK(last commanded q) + cumsum(d)[k],
       then IK -- the bridge's DLS ik_step -- and it must add to the
       COMMANDED hand position, not the measured one (they differ by
       ~20 mm of tracking offset in this data).
    2. STATE = 29 body joints + FK xyz of the MEASURED joints, 32 dims,
       in that order, q99-normalised and fed as pi0.5-style discrete
       tokens through the prompt (``include_state: true`` in the yaml;
       the no-state arm of the A/B leaves it false and never packs it).
    3. ALL THREE VIEWS AT 256 -- the ego view's mixres partner sizes are
       gone -- with 95% random crop + +-5 deg rotation on EVERY view (pi0.5
       did the ego view only; the operator asked for all three), then the
       shared ColorJitter(0.3, 0.4, 0.5) plus this bench's hue 0.08 and
       colour-temperature 0.15. The crop is implemented as resize-to-270
       -> RandomCrop(256) in train / CenterCrop(256) in eval, so the
       SERVING PATH MUST resize to 270 and centre-crop 256 -- a straight
       resize to 256 shows the model 5% more field of view than it ever
       trained on. obs_image_size stays [256, 256] (post-crop size).
       Geometric draws are PER VIEW (independent camera-pose jitter);
       colour draws are SHARED (one scene, one sensor), as every pipette
       recipe before it.
    4. 15% EPISODE HOLDOUT: ``-ee-train`` (249 eps) / ``-ee-eval`` (44
       eps), seed 1234, meta/splits.json in the ``-ee`` dir. The eval dir
       carries a COPY of the train dir's stats_gr00t.json so the holdout
       is normalised with train statistics (the loader globs its own dir
       otherwise). Trainer: ``datasets.vla_data.eval_data_mix``.

    Horizon stays 30 rows at 60 Hz (the bridge chunk contract), not
    pi0.5's 10 at 30 Hz. Registered as ``unitree_g1_pipette_3view_oft_ee``.
    """

    video_keys = [
        "video.rgb",
        "video.wrist_left",
        "video.wrist_right",
    ]
    VIEW_SIZE = 256
    CROP_FRAC = 0.95
    ROT_DEG = 5.0
    # int(270 * 0.95) == 256: resize here, crop to VIEW_SIZE.
    PRE_CROP = 270
    assert int(PRE_CROP * CROP_FRAC) == VIEW_SIZE

    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.right_ee_xyz",
    ]
    state_key_dims = {
        "state.left_leg": 6,
        "state.right_leg": 6,
        "state.waist": 3,
        "state.left_arm": 7,
        "state.right_arm": 7,
        "state.right_ee_xyz": 3,
    }

    action_keys = ["action.right_ee_dxyz"]
    action_key_dims = {"action.right_ee_dxyz": 3}
    action_normalization_modes = {"action.right_ee_dxyz": "q99"}

    observation_indices = [0]
    state_indices = [0]
    action_indices = list(range(30))

    def serve_view_preprocess(self, images):
        """Serve-time view pipeline == this config's EVAL transform chain.

        Training resized every view to PRE_CROP (270) and random-cropped
        256; eval (and therefore serving) resizes to 270 and CENTER-crops
        256. A server that resized straight to 256 would show the model 5%
        more field of view than it ever trained on — silently.

        Called by the policy server's obs adapter (gr00t_obs_adapter
        .obs_to_example) whenever the checkpoint's DataConfig defines it;
        takes/returns one uint8 HWC array per view.
        """
        from PIL import Image as _Image
        lo = (self.PRE_CROP - self.VIEW_SIZE) // 2
        hi = lo + self.VIEW_SIZE
        out = []
        for im in images:
            pil = _Image.fromarray(np.asarray(im))
            pil = pil.resize((self.PRE_CROP, self.PRE_CROP), _Image.BILINEAR)
            out.append(np.asarray(pil)[lo:hi, lo:hi])
        return out

    def transform(self):
        per_view = []
        for key in self.video_keys:
            per_view += [
                # check_input asserts the native resolution, so this is first
                VideoToTensor(apply_to=[key]),
                VideoResize(apply_to=[key], height=self.PRE_CROP, width=self.PRE_CROP,
                            interpolation="linear"),
                # train-only (get_transform returns None for eval -> pass-through)
                VideoRandomRotation(apply_to=[key], degrees=self.ROT_DEG,
                                    interpolation="linear"),
                # RandomCrop(256) in train, CenterCrop(256) in eval
                VideoCrop(apply_to=[key], height=self.PRE_CROP, width=self.PRE_CROP,
                          scale=self.CROP_FRAC),
            ]
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
                VideoColorTemperature(apply_to=self.video_keys, strength=0.15),
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


class Pipette3ViewOFTEEAbsDataConfig(Pipette3ViewOFTEEDataConfig):
    """Round 3 (2026-09-12): ABSOLUTE right-hand xyz, same everything else.

    The round-2 per-step delta head collapsed to exact zeros in closedloop
    at the hover-above-tip scene (flight 20260912_1917xx: 100% zero chunks
    from t=210 s on, both A/B arms): 34-45% of the delta labels are exact
    zeros (the ~50 Hz command stream under the 60 Hz recorder plus teleop
    micro-pauses that concentrate at fine alignment), so the conditional
    mean at that scene IS zero, and for a memoryless policy in closed loop
    zero output -> unchanged scene -> zero output is an absorbing state.

    Fix: action = ``action.right_ee_xyz`` -- FK(cmd[t+i]) of the COMMANDED
    joints, pelvis frame, metres, the column build_hil293_ee.py already
    wrote next to the deltas (modality carries no ``absolute: false`` for
    it; train q01/q99 span 116/156/73 mm per dim, and the eval dir's stats
    copy has the same entry). A demo pause now labels "hold THIS pose", a
    scene-dependent continuous value -- no probability spike shared across
    the dataset, so the collapse cannot be expressed. Views, augmentation,
    state block (unused by the no-state arm) and horizon are inherited
    unchanged. SERVING: rows are IK targets directly -- no integration, no
    last-command bookkeeping; cap per-step displacement AND the first-row
    jump. Holdout eval must run with ``eval_baseline: hold_first`` (a
    physical-zero action is the pelvis origin here, meaningless).
    Registered as ``unitree_g1_pipette_3view_oft_ee_abs``.
    """

    action_keys = ["action.right_ee_xyz"]
    action_key_dims = {"action.right_ee_xyz": 3}
    action_normalization_modes = {"action.right_ee_xyz": "q99"}


class Pipette3ViewOFTEERelDataConfig(Pipette3ViewOFTEEDataConfig):
    """Round 6 (2026-09-13): CHUNK-RELATIVE xyz, anchored at the last COMMAND.

    Round 3 (absolute xyz) stuck even at the start pose, and the reason is
    measurable: the absolute head's holdout error is 4.13/3.17/3.12 mm while
    the hand only moves 1.88/1.17/1.34 mm RMS inside one 0.5 s chunk, so in
    normalised units the motion-to-error ratio is 0.42/0.42/0.47 -- the head
    was never resolving its own signal. Worse, each absolute chunk re-anchors
    the trajectory at the model's own (non-advancing) position estimate, so
    the arm is pinned to a fixed point rather than accumulating motion. A
    2048-sample probe of the round-3 checkpoint confirmed the mechanism: 97.2%
    of its squared error is a per-chunk COMMON-MODE bias, and once that bias
    is differenced away the chunk shape carries real signal (SNR 1.26,
    episode-clustered 95% CI [1.20, 1.32], vs 1.01 for a vision-free canned
    ramp).

    So: keep the head, change what it is asked to predict. The label here is

        rel[i] = FK(cmd[t+i]) - FK(cmd[t]),   i = 0..29

    i.e. displacement from the pose the arm was last COMMANDED to, in metres,
    pelvis frame. Three things follow, all measured on the 249-episode train
    split (scratchpad anchor_cmp.py):

    1. RESOLUTION. The q99 span collapses 116.0/155.6/73.4 mm -> 13.3/10.4/6.9
       mm (5.2x / 8.6x / 2.0x... in fact 8.7x/14.9x/10.7x), and since q99
       normalisation maps [q01, q99] -> [-1, 1], that is a proportional gain in
       millimetre resolution. Motion-to-error goes 0.42/0.42/0.47 (round 3) ->
       3.63/6.29/5.05. Every axis clears 1.0 with margin.
    2. THE ANCHOR IS THE COMMAND, NOT THE MEASUREMENT. The loader's `rel` mode
       subtracts ``state_chunk[0]``, so the anchor is whatever state key it is
       pointed at. It MUST be pointed at the commanded column: on this robot
       FK(commanded) - FK(measured) is a ~15-18 mm quasi-static droop offset
       (mean -11.0/+12.0/+1.6 mm, and a +-40-frame lag scan improves it by
       0.16%, so it is steady-state tracking error, not latency). Anchoring at
       the measured pose would fold that offset into every label, force the
       head to predict a per-episode quantity that varies by +-20 mm in z, and
       turn any error in it into drift velocity (~3 mm/s in z). Hence the
       ``state.right_ee_cmd`` entry below, which reads the ACTION column.
       Anchor-at-measured scores 2.16/3.63/0.95 -- z barely at unity -- against
       3.63/6.29/5.05 here.
    3. THE PAUSE FILTER IS LOAD-BEARING, NOT HYGIENE. This parameterisation
       reintroduces a zero point mass that the absolute space did not have:
       conditioned on the arm already being stationary, 48.11% of chunks are
       bit-exact zero. QwenOFT trains with ``nn.L1Loss``, whose optimum is the
       conditional MEDIAN, and a median is pinned to exactly zero as soon as
       the zero mass straddles the 50th percentile -- which is precisely how
       round 2's delta head died (72.98% zero there). Dropping anchors whose
       in-chunk motion is under 1 mm removes the mass outright (48.11% -> 0%)
       at a cost of 17.4% of the anchors. The yaml must set it; training
       without it is predicted to stall.

    The dataset is ``g1-pipette-3view-hil293-eerel-{train,eval}``: the -ee
    dirs' data/ and videos/ symlinked, meta/ copied, and modality.json given
    the two aliases this config needs. A separate directory is mandatory
    because the statistics cache key is only ``{"mode": action_mode}`` and
    lives at ``meta/stats_gr00t.json`` inside the dataset dir, so a rel run
    and an abs run sharing a dir delete each other's cache on every launch.

    EVAL: ``eval_baseline: hold_first``. Row 0 of this label is identically
    zero, so repeating it IS the physical hold-still baseline, in raw or
    normalised units alike -- the metric is meaningful again without the
    round-3 caveat.

    SERVING: rows are displacements from the last commanded hand pose, so the
    bridge integrates exactly as the round-2 delta lane did (add to the
    COMMANDED pose, never the measured one) except that the rows are already
    cumulative -- no cumsum. Registered as
    ``unitree_g1_pipette_3view_oft_ee_rel``.
    """

    # state.right_ee_cmd aliases the COMMANDED hand column; the loader's rel
    # mode reads its row 0 as the anchor. It is never consumed by the model
    # (this arm runs include_state: false) -- it exists so that
    # _get_action_col_slices can resolve the pairing, which it does through
    # action_mode_state_map in the yaml.
    state_keys = Pipette3ViewOFTEEDataConfig.state_keys + ["state.right_ee_cmd"]
    state_key_dims = {**Pipette3ViewOFTEEDataConfig.state_key_dims,
                      "state.right_ee_cmd": 3}

    action_keys = ["action.right_ee_rel"]
    action_key_dims = {"action.right_ee_rel": 3}
    action_normalization_modes = {"action.right_ee_rel": "q99"}


class Pipette3ViewOFTWrist8DataConfig(Pipette3ViewOFTEEDataConfig):
    """Round 4 (2026-09-13): 8-D action = 6-D right-wrist POSE + 2-D thumb.

    Same recipe as round 3 (``Pipette3ViewOFTEEAbsDataConfig``) on a DIFFERENT
    dataset and with a richer action space:

      action.right_wrist_pose6  6 dims  [x, y, z, rx, ry, rz]
            FK(``action[:29]``) of link ``right_rubber_hand`` in the pelvis
            frame: absolute position in metres plus orientation as a ROTATION
            VECTOR (axis-angle). Round 3 predicted position only and HELD the
            wrist orientation, which meant the head could not rotate the wrist
            at all; this one can.
      action.right_thumb        2 dims  [thumb_bend, thumb_rot]
            A modality SLICE of the existing ``action.hand_right`` at [4:6]
            (the dataset's own feature names are pinky/ring/middle/index/
            thumb_bend/thumb_rot), in raw Inspire register units 0..1000. No new
            parquet column was needed for it. This is the first pipette head
            that commands the hand at all - every earlier one held the ready
            pose - so it can actually close the thumb on the tip.

    Axis-angle is safe on THIS data and that was checked, not assumed: the
    rotation angle spans 0.070..1.384 rad over all 53 episodes, leaving 1.757 rad
    (100.7 deg) of margin to the pi wrap where axis-angle goes discontinuous, and
    the largest within-episode step is 0.051 rad. Euler would have risked gimbal
    lock; a 6-D rotation representation would have made the pose 9-D.

    Dataset ``g1-pipette-3view-20260819-wrist8``, built by scratchpad
    build_20260819_wrist8.py from the HF canonical parquet of
    jren313/g1-pipette-3view-20260819 (sha 3f63167...); FK cross-checked against
    the bridge's UrdfKinematics to EXACTLY zero on both position and rotation.
    45 train / 8 eval episodes (59,273 / 9,875 frames), same seed-1234 15%
    episode holdout as round 3.

    NOTE the task is LONGER than hil293's: this capture also ejects the used tip
    into the disposal box, which is why the y span is 304 mm against round 3's
    156 mm.

    Views, augmentation, indices and ``serve_view_preprocess`` are inherited
    unchanged, so the only deliberate differences from round 3 are the dataset
    and the action space. Registered as ``unitree_g1_pipette_3view_oft_wrist8``.
    """

    # 35 dims; unused by the no-state arm (include_state false) but declared so
    # a with-state twin needs no rebuild. The 20260819 state vector is 34-wide
    # (29 joints + imu_ang_vel 3 + imu_rp 2); only the first 29 are sliced here.
    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.right_wrist_pose6",
    ]
    state_key_dims = {
        "state.left_leg": 6,
        "state.right_leg": 6,
        "state.waist": 3,
        "state.left_arm": 7,
        "state.right_arm": 7,
        "state.right_wrist_pose6": 6,
    }

    action_keys = ["action.right_wrist_pose6", "action.right_thumb"]
    action_key_dims = {"action.right_wrist_pose6": 6, "action.right_thumb": 2}
    action_normalization_modes = {
        "action.right_wrist_pose6": "q99",
        "action.right_thumb": "q99",
    }


def _insert_centre_crop(transforms, keys, size, replace_resize):
    """Put a VideoCenterCrop on `keys`, either before or instead of its resize.

    The parent (MixRes) chain holds exactly ONE VideoResize per view at the
    common working size, plus a second down-resize only for views whose target
    is smaller than common. The wrist views target 448 == common, so they carry
    exactly one -- asserted here, because silently cropping the wrong one would
    change the input scale with nothing to notice it.
    """
    out, hit = [], {k: 0 for k in keys}
    for t in transforms:
        applied = list(getattr(t, "apply_to", []))
        if type(t).__name__ == "VideoResize" and len(applied) == 1 and applied[0] in hit:
            key = applied[0]
            hit[key] += 1
            crop = VideoCenterCrop(apply_to=[key], size=size)
            if replace_resize:
                out.append(crop)          # crop REPLACES the resize
                continue
            out.append(crop)              # crop, then the parent's resize
        out.append(t)
    bad = {k: n for k, n in hit.items() if n != 1}
    if bad:
        raise RuntimeError(
            f"expected exactly one VideoResize per cropped view, got {bad}; "
            "the parent's transform() changed -- re-check this override")
    return out


class Pipette3ViewNoHistCrop720G1GR00TN1d7DataConfig(
        Pipette3ViewMixResNoHistG1GR00TN1d7DataConfig):
    """Round 12: round 11 with the two wrist views CROPPED 1080 -> 720 first.

    THE ONLY DIFFERENCE FROM ROUND 11 is what the wrist views show. Round 11
    resamples the stored 1080x1080 straight to 448 (a 2.41x downscale of the
    whole frame); round 12 takes the centre 720x720 and resamples THAT to 448
    (a 1.61x downscale of the middle 44% of the area). Same 448 input, same
    456 image tokens, same everything else -- so a difference in outcome is
    the field of view and the effective resolution on the rack, nothing else.

    CENTRE, NOT RANDOM. VideoCrop would give RandomCrop at train time; these
    are FIXED cameras and the bridge crops deterministically, so a random
    train-time crop invents a framing distribution the server never serves.
    See VideoCenterCrop's docstring.

    MEASURED BEFORE LAUNCH (12 episodes, 611 sampled frames per view):
      wrist_right  99.2% of green-tip detections fall inside the centred 720,
                   p1 margin +186 px. The rack sits squarely mid-frame here.
      wrist_left   78.4% inside, p1 margin -175 px -- AND rendering the frames
                   shows this view barely sees the rack at all: it looks along
                   the pipette at the bench, with the rack clipped at the top
                   edge and the detector often locking onto shelf clutter on
                   the right. The 22% "outside" is mostly false positives, but
                   the underlying fact stands -- cropping wrist_left tighter
                   removes what little of the rack it had.
    That asymmetry is why round 13 exists. Round 12 crops BOTH views because
    the request was to crop both; if only one round can run, 13 is the better
    motivated one.

    SERVING NEEDS A BRIDGE CHANGE. view_crop only implements "center_square"
    (cameras/base.py:110). A checkpoint from this round needs a centred 720x720
    of the live 1920x1080 frame -- centre-of-centre is centre, so it is one new
    crop mode, not a re-derivation. Train it, but do not expect to serve it
    until that lands.
    """

    WRIST_KEYS = ["video.wrist_left", "video.wrist_right"]
    CROP_SIZE = 720

    def transform(self):
        composed = super().transform()
        return ComposedModalityTransform(transforms=_insert_centre_crop(
            composed.transforms, self.WRIST_KEYS, self.CROP_SIZE,
            replace_resize=False))


class Pipette2ViewNoHistCrop448G1GR00TN1d7DataConfig(
        Pipette3ViewMixResNoHistG1GR00TN1d7DataConfig):
    """Round 13: round 11 with TWO views, and wrist_right cropped 1080 -> 448.

    WHAT CHANGES FROM ROUND 11
      * wrist_left is DROPPED. Rendering the stored frames (2026-08-27) shows
        it looking along the pipette at the bench with the tip rack clipped at
        the top edge; the green-blob detector's hits there are largely shelf
        clutter, and its centroid spans the full frame width (p1 x=23,
        p99 x=1069) where wrist_right's spans 392..604. It is the view that
        carries the least about "which well is the tip over".
      * wrist_right is CENTRE-CROPPED to 448 AND NOT RESIZED. The crop is the
        whole resampling step: 448 px of the original sensor grid reach the
        model at 1:1, so the rack is at native resolution for the first time
        in this project. 99.2% of tip detections fall inside that window
        (p1 margin +50 px) -- tight, but measured, not assumed.

    TOKEN BUDGET DROPS: ego 64 + one wrist 196 = 260 image tokens against
    round 11's 456. That should shorten the VL encode noticeably, which is
    worth watching given the served rate has been the binding constraint.

    THE RISK IS THE 50 px MARGIN. Round 12's 720 window has +186 px of slack
    on the same view; this one has +50. If the rack shifts on the bench, or an
    episode is staged a few cm off, the crop can lose it -- and unlike a
    resize, a crop cannot be undone downstream. Re-run the crop survey against
    any NEW capture before reusing this DataConfig on one.

    SERVING NEEDS A BRIDGE CHANGE, as round 12 -- a centred 448x448 of the
    live 1920x1080 frame, plus a 2-view profile.
    """

    video_keys = ["video.rgb", "video.wrist_right"]
    view_resize_overrides = {"video.wrist_right": (448, 448)}
    WRIST_KEYS = ["video.wrist_right"]
    CROP_SIZE = 448

    def transform(self):
        composed = super().transform()
        return ComposedModalityTransform(transforms=_insert_centre_crop(
            composed.transforms, self.WRIST_KEYS, self.CROP_SIZE,
            replace_resize=True))


class Pipette3ViewOFTEERelCropDataConfig(Pipette3ViewOFTEERelDataConfig):
    """Round 7 (2026-09-14): round 6 with the views CROPPED BEFORE the resize.

    THE ONLY DIFFERENCE FROM ROUND 6 is how many native pixels land on the
    target. Action space, pause filter, state block, horizon, augmentation,
    dataset and holdout are inherited byte-for-byte, so a difference in outcome
    is the input geometry and nothing else.

    WHY. Round 6 learns x (holdout ratio 0.749) and z (0.769) and learns nothing
    at all on y (0.918-1.009, i.e. no better than predicting the hand does not
    move laterally), and on the robot it descends cleanly but misses in y. The
    target is the ONE well that has a tip in it, so y is decided by finding that
    tip. Measured, with an annulus-context colour classifier that was checked by
    eye (two earlier detectors silently tracked a green marker on the WRIST and
    had to be thrown away), the tip subtends in ROUND 6's INPUT:

        rgb          8 x  8 native ->  1.7 x 3.0 px      2.0% of a 16x16 patch
        wrist_left  20 x 20 native ->  5.0 x 5.0 px      9.8%
        wrist_right 21 x 22 native ->  5.2 x 5.5 px     11.3%

    Under 2 px wide in the side view -- and the side view is the ONLY one whose
    y sensitivity is real (y -> dx = -0.425 px/mm; wrist_right has y almost
    exactly along its optical axis at dx +0.010, dy -0.087). Worse, that view is
    1280x720 resized to a SQUARE 270, so the horizontal axis -- the one carrying
    y -- is squeezed 4.74x against the vertical's 2.67x. The one clue that says
    which well to aim at is 1.7 px wide in the one direction that matters.

    WHAT THIS CHANGES.
        video.rgb          centre 720x720 of 1280x720 (i.e. center_square)
                           scale 0.211 x / 0.375 y  ->  0.375 / 0.375
                           tip 1.7 x 3.0  ->  3.0 x 3.0 px   (1.78x wider)
                           aspect distortion gone; the cost is the outer 44% of
                           the horizontal field of view.
        video.wrist_left   centre 448x448 of 1080x1080
                           scale 0.250  ->  0.603
                           tip 5.0  ->  12.1 px            (2.41x)
        video.wrist_right  UNCHANGED. It is blind to y by geometry, so pixels
                           spent on it cannot buy y; it is already the sharpest
                           view for x and z (2.1 px of image motion per chunk).

    COVERAGE, CHECKED BEFORE LAUNCH because a crop that drops the target cannot
    be recovered after training (25 episodes x 6 offsets across the approach):
        rgb centre square  tip inside 100.0%, worst single offset 100.0%
        wrist_left 448     tip inside  93.8%, worst single offset  83.3%
    wrist_left's 448 box is the deliberate trade in this round: 2.41x the tip,
    against losing it in ~6% of frames (17% at the worst phase) because its tip
    position scatters with sd 127 px vertically. rgb still has the tip in those
    frames. Nothing between 448 and 864 improves wrist_left's coverage -- the
    misses are far outside, not marginal -- so 448 costs no more than 720 would.

    SERVING NEEDS NO NEW BRIDGE CODE, which is why these two crops and not some
    better-placed box: both are already-implemented view_crop modes.
        "rgb": "center_square"      largest centred square; the live side camera
                                    is 1280x720, so this is the same 720x720.
        "wrist_left": "center:448"  the stored 1080x1080 IS the centred square of
                                    the live 1920x1080, and centre-of-a-centred-
                                    square is the centre, so one mode is correct
                                    for both.
        "wrist_right": "center_square"   as round 6.
    ``serve_view_preprocess`` is inherited UNCHANGED and must stay that way: the
    bridge crops on receive, before the frame reaches the policy server, so the
    server still sees a frame it only has to resize to 270 and centre-crop 256.

    VideoCenterCrop, not VideoCrop: these are fixed cameras and the bridge crops
    deterministically, so a random train-time crop would invent a framing
    distribution the server never serves. It is inserted BEFORE each view's
    resize (replace_resize=False), which is also after VideoToTensor, the order
    its docstring requires. The 95% RandomCrop(256) augmentation downstream is
    untouched.

    DATASET IS SHARED WITH ROUND 6 ON PURPOSE. The crop changes images only, so
    the action/state statistics are identical, and the stats cache key is
    {"mode": action_mode} -- both rounds are "rel" on the same dirs and read the
    same meta/stats_gr00t.json. No rebuild, and no cache fight of the kind that
    forced the -eerel- dirs to exist in the first place.

    KNOWN LIMIT, stated so it is not read as a prediction of success: this round
    fixes the INPUT side only. QwenOFT trains with nn.L1Loss, whose optimum is
    the conditional median, and the 1.0 mm pause filter thresholds the 3-D norm,
    so it never cleared y's per-axis zero mass -- 65.2% of surviving train chunks
    still have |net y| < 0.25 mm (x 7.1%, z 9.3%) and the training median of
    net y is +0.050 mm. A controlled loss A/B on exactly these labels collapses
    the y output 14x under L1 against L2 while leaving x and z alone. So a y
    that is visible may still be predicted as zero. Registered as
    ``unitree_g1_pipette_3view_oft_ee_rel_crop``.
    """

    RGB_KEY = ["video.rgb"]
    RGB_CROP = 720          # 1280x720 -> the largest centred square
    WRIST_LEFT_KEY = ["video.wrist_left"]
    WRIST_LEFT_CROP = 448   # 1080x1080 -> centred 448

    def transform(self):
        composed = super().transform()
        t = _insert_centre_crop(composed.transforms, self.RGB_KEY,
                                self.RGB_CROP, replace_resize=False)
        t = _insert_centre_crop(t, self.WRIST_LEFT_KEY,
                                self.WRIST_LEFT_CROP, replace_resize=False)
        return ComposedModalityTransform(transforms=t)


class Pipette3ViewOFTEERelCrop3DataConfig(Pipette3ViewOFTEERelCropDataConfig):
    """Round 8: round 7 with wrist_right ALSO centre-cropped, 1080 -> 720.

    SINGLE VARIABLE AGAINST ROUND 7. rgb (centre 720 of 1280x720) and
    wrist_left (centre 448 of 1080) are inherited untouched, as are the action
    space, state block, pause filter, dataset and horizon. The one change:

        video.wrist_right   1080x1080 -> centre 720x720 -> 270 -> 256
                            scale 0.237 -> 0.356,  tip 5.3 -> 7.6 px (1.42x)

    WHY 720 AND NOT 448. Two separate measurements, pulling the same way.
    (a) A gray-frame ablation of round 7 @ steps_2000 on 96 holdout chunks
        showed wrist_right had gone IDLE -- blanking it moved x +4%, z -1%,
        and y -6% (i.e. slightly better without it) -- while wrist_left
        carried everything (blanking it: y +18%, z +107%). Round 7's crops
        moved all the reliance onto one view, which is also the one whose
        448 box loses the tip in ~6% of frames. Giving wrist_right real
        resolution is meant to restore a second, INDEPENDENT source for the
        same evidence rather than to add y sensitivity it cannot have.
    (b) wrist_right is the best "WHICH well" view -- fixed, top-down, the
        rack centred and static in frame -- and at 720 the whole rack plus
        the pipette and the green tip stay inside the box for the entire
        approach, at 1.42x the magnification. At 448 the rack's edges leave
        the frame, which is exactly the context the task's one cue (the
        single well holding a tip) needs.

    WHAT THIS CANNOT DO, stated so the result is read correctly:
    wrist_right's y sensitivity is ~0.01 px/mm (y runs down its optical
    axis); 1.42x magnification leaves it ~0.014. This crop cannot let the
    model watch itself move in y. What it can do is make the TARGET's
    identity easier to read, which combined with state (the commanded hand
    pose, already in the 35-dim vector) is a different route to the same
    number. The L1-median / per-axis zero-mass mechanism that keeps y's aim
    slope at 0.18 is untouched here too -- that is the loss-side fix, kept
    out of this round to preserve the single variable.

    SERVING NEEDS NO NEW BRIDGE CODE: "wrist_right": "center:720" is an
    already-implemented view_crop mode (rounds 12/13 use center:N), and
    centre-of-a-centred-square is the centre, so the one mode is correct for
    both the stored 1080x1080 mp4s and the live 1920x1080 stream.

    DATASET SHARED WITH ROUNDS 6 AND 7 for the same reason as round 7: the
    crop touches images only, the action/state statistics are identical, and
    the stats cache key is {"mode": "rel"} for all three. Registered as
    ``unitree_g1_pipette_3view_oft_ee_rel_crop3`` ("3" = all three views now
    carry a centre crop).
    """

    WRIST_RIGHT_KEY = ["video.wrist_right"]
    WRIST_RIGHT_CROP = 720   # 1080x1080 -> centred 720

    def transform(self):
        composed = super().transform()      # round 7's rgb + wrist_left crops
        t = _insert_centre_crop(composed.transforms, self.WRIST_RIGHT_KEY,
                                self.WRIST_RIGHT_CROP, replace_resize=False)
        return ComposedModalityTransform(transforms=t)


class Pipette3ViewOFTWrist8RelCrop3DataConfig(Pipette3ViewOFTEERelCrop3DataConfig):
    """Round 9 (2026-09-15): round 8's recipe with an 8-D CHUNK-RELATIVE action
    (6-D wrist pose + 2-D thumb) on hil293 POOLED WITH the plus21 capture.

    Views, crops (rgb 720 / wrist_left 448 / wrist_right 720 -> 270 -> 256),
    augmentation, horizon, pause filter and the discretised-state prompt are
    round 8's, inherited unchanged. What changes:

    ACTION  action.right_wrist_rel6 (6) + action.right_thumb_rel (2) = 8.
        pose: rel[i] = pose(cmd[t+i]) (-) pose(cmd[t]) with the position part
              subtracted (metres, pelvis frame, as round 8) and the rotation
              part the EXACT body-frame relative rotation
              rotvec(R_cmd[t]^T R_cmd[t+i]) -- NOT an elementwise rotation-
              vector difference, which was measured on plus21 to be wrong by
              p50 0.30 deg / p99 2.2 deg / max 5.6 deg against relative
              rotations of p50 1.0 deg / p99 14 deg (scratchpad
              build_plus21_wrist8.py). The loader does this through
              `action_mode_rotvec_dims` (datasets.py, round-9 patch), in the
              label AND in the q99 statistics.
        thumb: rel[i] = hand_right[t+i][4:6] - hand_right[t][4:6], raw Inspire
              registers, anchored at the last COMMANDED thumb like the pose.
        Registered under its own name so the serve-time config resolves.

    STATE  43 dims = 29 body joints + measured wrist pose6 + COMMANDED wrist
        pose6 + COMMANDED thumb2. The two commanded blocks are the chunk's own
        anchors, read through aliases of BYTE-COPY columns
        (action.right_wrist_cmd6, action.right_thumb_cmd), never of the label
        columns: `action_mode: rel` overwrites the statistics of the column it
        is applied to, and rounds 6-8 read their commanded-xyz state block
        through such a shared column -- its three state dims were normalised
        with the +-6.5 mm REL span and saturated to a constant (verified in
        round 8's dataset_statistics.json). Round 9 keeps them absolute.

    DATA  hil293-wrist8rel-{train,eval} (249/44 eps, the seed-1234 split every
        OFT round shares) + 20260819-plus21-wrist8rel-{train,eval} (66/12 eps:
        the 8 episodes rounds 4/5 held out of the first 53, plus 4 of the 25
        new ones). ONE union rel-mode statistics file is written into all four
        dirs (scratchpad build_wrist8rel_r9.py) so both halves share a q99 box;
        mixture weights are the post-pause-filter anchor counts. The two
        captures carry DIFFERENT task sentences (hil293: attach only; plus21:
        attach then eject into the disposal box) and the thumb only works in
        the latter -- language is what disambiguates them, as in round 5.
    """

    state_keys = [
        "state.left_leg",
        "state.right_leg",
        "state.waist",
        "state.left_arm",
        "state.right_arm",
        "state.right_wrist_pose6",   # FK(measured), absolute
        "state.right_wrist_cmd6",    # FK(commanded) = the pose anchor, absolute
        "state.right_thumb_cmd",     # commanded thumb = the thumb anchor
    ]
    state_key_dims = {
        "state.left_leg": 6,
        "state.right_leg": 6,
        "state.waist": 3,
        "state.left_arm": 7,
        "state.right_arm": 7,
        "state.right_wrist_pose6": 6,
        "state.right_wrist_cmd6": 6,
        "state.right_thumb_cmd": 2,
    }

    action_keys = ["action.right_wrist_rel6", "action.right_thumb_rel"]
    action_key_dims = {"action.right_wrist_rel6": 6, "action.right_thumb_rel": 2}
    action_normalization_modes = {
        "action.right_wrist_rel6": "q99",
        "action.right_thumb_rel": "q99",
    }


ROBOT_TYPE_CONFIG_MAP = {
    "unitree_g1_pipette_n1d7": PipetteTipG1GR00TN1d7DataConfig(),
    "unitree_g1_pipette_armhand_n1d7": PipetteTipG1ArmHandOnlyDataConfig(),
    # Same task/action space, plus the detector-tracked tip crop (2 views).
    "unitree_g1_pipette_tipcrop_n1d7": PipetteTipCropG1GR00TN1d7DataConfig(),
    # Round 3: same two views, action space + waist (27 dims).
    "unitree_g1_pipette_tipcrop_waist_n1d7": PipetteTipCropWaistG1GR00TN1d7DataConfig(),
    # Round 4: same two views and 27-dim action, plus a 4-frame state history.
    # Pair with state_history_length 4 and serve WITHOUT --no_state.
    "unitree_g1_pipette_tipcrop_waist_hist_n1d7":
        PipetteTipCropWaistHistG1GR00TN1d7DataConfig(),
    # Round 6: the three-camera 2026-08-19 capture, 256x256 per view.
    "unitree_g1_pipette_3view_waist_hist_n1d7":
        Pipette3ViewWaistHistG1GR00TN1d7DataConfig(),
    # Round 7: round 6 with the two wrist views at 448.
    "unitree_g1_pipette_3view_mixres_n1d7":
        Pipette3ViewMixResG1GR00TN1d7DataConfig(),
    # Round 11: round 7 with NO state history (1 frame, not 4). Pair with
    # state_history_length 1 and state_dropout_prob 0.8 in the yaml.
    "unitree_g1_pipette_3view_mixres_nohist_n1d7":
        Pipette3ViewMixResNoHistG1GR00TN1d7DataConfig(),
    # QwenOFT (2026-09-11): round 11's views/action/horizon, MLP L1 head,
    # vision+language-only. Own name so the serve-time config resolves.
    "unitree_g1_pipette_3view_oft": Pipette3ViewOFTDataConfig(),
    # QwenOFT round 2 (2026-09-12): pi0.5-hil194 recipe -- 3-dim right-hand
    # FK delta action, 32-dim state (29 joints + FK xyz), three 256 views
    # with crop/rotation augmentation, 15% episode holdout.
    "unitree_g1_pipette_3view_oft_ee": Pipette3ViewOFTEEDataConfig(),
    "unitree_g1_pipette_3view_oft_ee_abs": Pipette3ViewOFTEEAbsDataConfig(),
    "unitree_g1_pipette_3view_oft_wrist8": Pipette3ViewOFTWrist8DataConfig(),
    # Round 6 (2026-09-13): chunk-relative xyz anchored at the last COMMANDED
    # hand pose, via the loader's own action_mode: rel pointed at an action
    # column. Pairs with pause_filter_mm: 1.0 in the yaml -- see the class.
    "unitree_g1_pipette_3view_oft_ee_rel": Pipette3ViewOFTEERelDataConfig(),
    # Round 7: round 6 with rgb centre-cropped 1280x720 -> 720x720 and
    # wrist_left 1080 -> 448 BEFORE their resize. Same action space, same
    # dataset, same token count -- only how many native pixels reach the tip.
    "unitree_g1_pipette_3view_oft_ee_rel_crop":
        Pipette3ViewOFTEERelCropDataConfig(),
    # Round 8: round 7 plus wrist_right centre-cropped 1080 -> 720. Same
    # action space and dataset again; the single variable is that the view
    # round 7's own ablation found IDLE now gets 1.42x the resolution.
    "unitree_g1_pipette_3view_oft_ee_rel_crop3":
        Pipette3ViewOFTEERelCrop3DataConfig(),
    # Round 9 (2026-09-15): round 8's views/crops/state prompt with an 8-D
    # chunk-relative action (wrist pose6 + thumb2) on hil293 + plus21.
    "unitree_g1_pipette_3view_oft_wrist8_rel_crop3":
        Pipette3ViewOFTWrist8RelCrop3DataConfig(),
    # Round 12: round 11 with the wrist views centre-cropped 1080 -> 720
    # BEFORE the 448 resize. Same three views, same token count.
    "unitree_g1_pipette_3view_nohist_crop720_n1d7":
        Pipette3ViewNoHistCrop720G1GR00TN1d7DataConfig(),
    # Round 13: round 11 with wrist_left dropped and wrist_right centre-cropped
    # 1080 -> 448 with NO resize (native pixels). Two views, 260 image tokens.
    "unitree_g1_pipette_2view_nohist_crop448_n1d7":
        Pipette2ViewNoHistCrop448G1GR00TN1d7DataConfig(),
    # Round 5: as above, with per-frame phase sentences as the language input.
    "unitree_g1_pipette_tipcrop_waist_hist_phase_n1d7":
        PipetteTipCropWaistHistPhaseG1GR00TN1d7DataConfig(),
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
    # TWO views (ego + detector-tracked tip crop). Different DATASET DIR to the
    # two mixes above -- the tipcrop repo re-encodes rgb with the crop-box
    # overlay, so the streams are not interchangeable with the source dataset's.
    #   hf download birbirll/g1-pipette-tip-teleop-tipcrop --repo-type dataset \
    #     --local-dir <DATA_ROOT>/g1-pipette-tip-teleop-tipcrop
    "unitree_g1_pipette_tipcrop_n1d7": [
        ("g1-pipette-tip-teleop-tipcrop", 1.0, "unitree_g1_pipette_tipcrop_n1d7"),
    ],
    # Same dataset, 27-dim action (arms | hand_right | height | nav | WAIST).
    # Needs --framework.action_model.action_dim 27 at launch.
    "unitree_g1_pipette_tipcrop_waist_n1d7": [
        ("g1-pipette-tip-teleop-tipcrop", 1.0, "unitree_g1_pipette_tipcrop_waist_n1d7"),
    ],
    # SPLIT MIXES (build_splits.py). Same DataConfig as the mix above -- the
    # modality layout and transforms are identical; only the EPISODE SET
    # differs -- pointed at a separate dataset DIRECTORY, which is the whole
    # point: the loader computes normalisation statistics by globbing
    # data/*/*.parquet over the dataset dir, not from meta/episodes.jsonl, so
    # a split sharing one parquet tree would normalise train using eval's
    # statistics no matter what the manifest said.
    #   49 train / 10 eval / ep 0 held out for bench testing, seed 1234;
    #   eps 40,44,45,55 dropped (crop locked on the forearm, no rack in
    #   frame -- verified by eye across each episode 2026-08-13).
    #   Manifest: <DATA_ROOT>/g1-pipette-tip-teleop-tipcrop-split.json
    "unitree_g1_pipette_tipcrop_train_n1d7": [
        ("g1-pipette-tip-teleop-tipcrop-train", 1.0,
         "unitree_g1_pipette_tipcrop_n1d7"),
    ],
    "unitree_g1_pipette_tipcrop_train_waist_n1d7": [
        ("g1-pipette-tip-teleop-tipcrop-train", 1.0,
         "unitree_g1_pipette_tipcrop_waist_n1d7"),
    ],
    # Round 4: the same train split, read through the STATE-HISTORY DataConfig.
    # Only state_indices differs, so the dataset directory is unchanged and the
    # round-3 normalisation statistics still describe it exactly.
    "unitree_g1_pipette_tipcrop_train_waist_hist_n1d7": [
        ("g1-pipette-tip-teleop-tipcrop-train", 1.0,
         "unitree_g1_pipette_tipcrop_waist_hist_n1d7"),
    ],
    # Round 5: the merged 73-episode phase-labelled set, WHOLE. Includes the
    # eval and holdout episodes -- use it for statistics or inspection, not for
    # a training run you intend to score.
    "unitree_g1_pipette_phase_n1d7": [
        ("g1-pipette-tip-teleop-phase", 1.0,
         "unitree_g1_pipette_tipcrop_waist_hist_phase_n1d7"),
    ],
    # Round 5 TRAINING mixture: splits.train + splits.new = 62 episodes /
    # 57,875 frames, built by
    #   build_splits.py --src .../g1-pipette-tip-teleop-phase --from-splits
    # --from-splits matters. The phase set already carries the 2026-08-13
    # assignment in meta/splits.json, and re-randomising would move round 4's
    # eval episodes into round 5's training set -- the two rounds would then no
    # longer be comparable on the one split they are both supposed to be
    # scored on. The split needs its OWN directory tree, not a manifest,
    # because normalisation statistics are globbed off the parquet tree.
    # Round 6: the whole 53-episode three-camera capture (no split exists).
    "unitree_g1_pipette_3view_n1d7": [
        ("g1-pipette-3view-20260819", 1.0,
         "unitree_g1_pipette_3view_waist_hist_n1d7"),
    ],
    # Round 7: same 53-episode capture, ego 256 + both wrist views at 448.
    "unitree_g1_pipette_3view_mixres_n1d7_mix": [
        ("g1-pipette-3view-20260819", 1.0,
         "unitree_g1_pipette_3view_mixres_n1d7"),
    ],
    # ROUND 10: the same mixres recipe on the MERGED capture -- the 53
    # episodes of 20260819_1211 plus 25 more recorded 20260821_0911, rebuilt
    # into one dataset. Episodes 0..52 are byte-identical to
    # g1-pipette-3view-20260819 (same session, same converter, same flags);
    # 53..77 are the new ones. Same rig, same green tips, same three cameras,
    # same task sentence -- the DataConfig is reused unchanged because nothing
    # about the observation or action space differs, only the episode count.
    #
    # NOTE THE EPOCH COUNT, AND HOW IT WAS HELD. At 10000 steps and batch 128
    # this dataset gives ~12.5 epochs against round 9's 18.5, because the frame
    # count grew and the step budget would not have. That would have changed
    # TWO things at once -- more data AND a third less training per frame -- so
    # a worse result could not have been pinned on either. Round 10 WAS RUN AT
    # 15000 steps instead: 15000 x 128 / 102,474 = 18.7 epochs, matching round
    # 9's 18.5, which is what isolates 'more data' as the single variable.
    # MAX_TRAIN_STEPS must be exported by the LAUNCHER to take effect -- the
    # run script passes its own default on the command line, which beats the
    # yaml. The first attempt was launched with only the yaml edited and came
    # up as 0/10000.
    "unitree_g1_pipette_3view_mixres_plus21_n1d7_mix": [
        ("g1-pipette-3view-20260819-plus21", 1.0,
         "unitree_g1_pipette_3view_mixres_n1d7"),
    ],
    # Round 7 AS RUN: the same mixres recipe on the P1+P2 TRUNCATION.
    #
    # Every episode STOPS when phase 2 finishes -- the frame the tip seats in
    # the well -- so the model is trained on approach + insertion only and
    # never sees the lift, the carry, or the ejection. 37,943 of 69,148
    # frames (54.9%), median 11.4 s per episode against 19.4 s.
    #
    # WHY A DIFFERENT DIRECTORY and not a filter: normalisation statistics are
    # globbed off the parquet tree, so a mixture that pointed at the full
    # capture and dropped rows later would normalise the actions against a
    # distribution containing the carry and the ejector press -- motions this
    # round deliberately excludes. The q99 the head trains against has to be
    # the q99 of what it actually sees.
    #
    # LANGUAGE IS ONE CONSTANT SENTENCE, rewritten for the shorter task
    # ("Attach a green pipette tip from the tip rack to the pipette."). The
    # source capture's sentence continues "...then eject the used tip into
    # the tip disposal box", which the truncated data never shows; carrying
    # it over would train the model on an instruction it cannot complete.
    # The phase column survives in the parquet but nothing here reads it.
    "unitree_g1_pipette_3view_mixres_p12_n1d7_mix": [
        ("g1-pipette-3view-20260819-p12", 1.0,
         "unitree_g1_pipette_3view_mixres_n1d7"),
    ],
    # ROUND 11: the SAME p12 truncation round 7 trained on, read through the
    # memoryless DataConfig. Same dataset dir on purpose -- round 11 vs round
    # 7 differs in the state input and nothing about the data, which is what
    # makes it a one-variable test of "does state history help or alias?".
    "unitree_g1_pipette_3view_mixres_nohist_n1d7_mix": [
        ("g1-pipette-3view-20260819-p12", 1.0,
         "unitree_g1_pipette_3view_mixres_nohist_n1d7"),
    ],
    # ROUNDS 12 AND 13: the SAME p12 dataset directory as rounds 7 and 11.
    # Nothing about the data changes across 11/12/13 -- only which pixels of
    # the wrist views reach the model -- which is what makes the three a
    # one-variable field-of-view sweep rather than three separate experiments.
    "unitree_g1_pipette_3view_nohist_crop720_n1d7_mix": [
        ("g1-pipette-3view-20260819-p12", 1.0,
         "unitree_g1_pipette_3view_nohist_crop720_n1d7"),
    ],
    "unitree_g1_pipette_2view_nohist_crop448_n1d7_mix": [
        ("g1-pipette-3view-20260819-p12", 1.0,
         "unitree_g1_pipette_2view_nohist_crop448_n1d7"),
    ],
    # ROUND 14: round 11's exact recipe (same DataConfig object, same state/
    # action/dropout settings) on the PLUS21 capture's phase-1+2 cut -- 78
    # episodes / 58,371 frames vs p12's 53 / 37,943. Boundaries: eps 0..52
    # are the p12 gold boundaries verbatim; eps 53..77 come from the FK
    # bottom-dwell detector (validated against all 53 gold boundaries:
    # median -10 / p90 32 / max 100 frames) plus a 40-frame late margin, and
    # every new cut frame was visually checked. One variable vs round 11:
    # +25 episodes. See <dataset>/meta/cut_provenance.json.
    "unitree_g1_pipette_3view_mixres_nohist_plus21p12_n1d7_mix": [
        ("g1-pipette-3view-20260819-plus21-p12", 1.0,
         "unitree_g1_pipette_3view_mixres_nohist_n1d7"),
    ],
    # ROUND 15: round 14's 78 teleop episodes + the 46 HG-DAgger HIL
    # correction episodes (collected 2026-09-04 on the R11/R14 rollout
    # stack, hf jren313/g1-pipette-3view-hgdagger-20260904), merged into
    # ONE directory (meta/build_script.py archived there) so the q99
    # stats are computed over the teleop+HIL union. Same DataConfig as
    # rounds 11/14 — the HIL episodes carry the identical channel set.
    "unitree_g1_pipette_3view_mixres_nohist_plus21p12_hil46_n1d7_mix": [
        ("g1-pipette-3view-plus21p12-hil46", 1.0,
         "unitree_g1_pipette_3view_mixres_nohist_n1d7"),
    ],
    # R15 STRUCTURE, NEW HIL BLOCK (2026-09-10): round 14's 78 teleop
    # episodes + 30 fresh joystick-HIL episodes recorded after the raw
    # capture dir was reset (episode_0001-0030, all successful, hand
    # actions live via the adapter mirror, wrist slots audited against
    # hil293 references). Merged into ONE directory by the same
    # build_merge_r15.py so q99 covers the union; fine-tuned from the
    # R14 checkpoint exactly like plus21p12_hil46_ft14 — the single
    # variable vs that arm is the replacement HIL data.
    "unitree_g1_pipette_3view_mixres_nohist_plus21p12_hil30_n1d7_mix": [
        ("g1-pipette-3view-plus21p12-hil30", 1.0,
         "unitree_g1_pipette_3view_mixres_nohist_n1d7"),
    ],
    # QwenOFT FROM SCRATCH on the full HIL union (2026-09-11): 293 episodes
    # / 363,006 frames (hf jren313/g1-pipette-3view-hil293). Same three
    # views and 27-dim action as every N1.7 round; the head is the MLP L1
    # regressor over Qwen3-VL-4B. 60000 steps at batch 128 = 21.2 epochs.
    "unitree_g1_pipette_3view_oft_hil293_mix": [
        ("g1-pipette-3view-hil293", 1.0, "unitree_g1_pipette_3view_oft"),
    ],
    # QwenOFT ROUND 2 (2026-09-12): hil293 with the right-hand FK columns
    # added (``-ee``, built by build_hil293_ee.py from the hil293 parquet;
    # videos are symlinks into hil293), split 249 / 44 episodes by seed
    # 1234 (meta/splits.json). SEPARATE DIRS because q99 statistics are
    # globbed off the dir: the eval dir ships a copy of the train dir's
    # stats_gr00t.json so holdout targets are normalised with train stats.
    # Two runs share both mixes and differ only in the yaml's include_state.
    "unitree_g1_pipette_3view_oft_ee_train_mix": [
        ("g1-pipette-3view-hil293-ee-train", 1.0, "unitree_g1_pipette_3view_oft_ee"),
    ],
    # Round 5 (2026-09-13): the two captures pooled under ONE action space and
    # ONE q99 box. Weights are the TRAIN FRAME COUNTS, so sampling matches the
    # natural pooled distribution -- LeRobotMixtureDataset.sample_step draws the
    # dataset by these weights (it does NOT concatenate), so the default 1.0/1.0
    # would have oversampled the 45-episode capture about 6x.
    # The two dirs carry a byte-identical meta/stats_gr00t.json computed over the
    # union of both train splits (scratchpad union_stats_wrist8.py) -- without
    # that each half normalises with its own quantiles and the same physical pose
    # gets different labels (spans 122/157/75 mm vs 68/304/134 mm).
    # hil293's split is the SAME episodes round 3 held out, so the shared half is
    # directly comparable between the two runs.
    "unitree_g1_pipette_3view_oft_wrist8_merged_train_mix": [
        ("g1-pipette-3view-hil293-wrist8-train", 306891.0,
         "unitree_g1_pipette_3view_oft_wrist8"),
        ("g1-pipette-3view-20260819-wrist8-train", 59273.0,
         "unitree_g1_pipette_3view_oft_wrist8"),
    ],
    "unitree_g1_pipette_3view_oft_wrist8_merged_eval_mix": [
        ("g1-pipette-3view-hil293-wrist8-eval", 56115.0,
         "unitree_g1_pipette_3view_oft_wrist8"),
        ("g1-pipette-3view-20260819-wrist8-eval", 9875.0,
         "unitree_g1_pipette_3view_oft_wrist8"),
    ],
    "unitree_g1_pipette_3view_oft_wrist8_train_mix": [
        ("g1-pipette-3view-20260819-wrist8-train", 1.0,
         "unitree_g1_pipette_3view_oft_wrist8"),
    ],
    "unitree_g1_pipette_3view_oft_wrist8_eval_mix": [
        ("g1-pipette-3view-20260819-wrist8-eval", 1.0,
         "unitree_g1_pipette_3view_oft_wrist8"),
    ],
    "unitree_g1_pipette_3view_oft_ee_abs_train_mix": [
        ("g1-pipette-3view-hil293-ee-train", 1.0, "unitree_g1_pipette_3view_oft_ee_abs"),
    ],
    "unitree_g1_pipette_3view_oft_ee_abs_eval_mix": [
        ("g1-pipette-3view-hil293-ee-eval", 1.0, "unitree_g1_pipette_3view_oft_ee_abs"),
    ],
    # Round 6. SEPARATE dataset dirs, not a flag: the statistics cache key is
    # only {"mode": action_mode} and lives at meta/stats_gr00t.json inside the
    # dataset dir, so a rel run and an abs run sharing a dir would delete and
    # recompute each other's cache on every launch. data/ and videos/ are
    # symlinks into the -ee dirs; only meta/ is real (scratchpad
    # build_eerel_dirs.py).
    "unitree_g1_pipette_3view_oft_ee_rel_train_mix": [
        ("g1-pipette-3view-hil293-eerel-train", 1.0, "unitree_g1_pipette_3view_oft_ee_rel"),
    ],
    "unitree_g1_pipette_3view_oft_ee_rel_eval_mix": [
        ("g1-pipette-3view-hil293-eerel-eval", 1.0, "unitree_g1_pipette_3view_oft_ee_rel"),
    ],
    # Round 7 reuses round 6's dirs deliberately. The crop touches images only,
    # so the action/state statistics are identical and both rounds are
    # action_mode "rel" -- they read the SAME meta/stats_gr00t.json rather than
    # invalidating each other, which is exactly the failure the separate -eerel-
    # dirs above exist to avoid. Round 6 stays reproducible while this runs.
    "unitree_g1_pipette_3view_oft_ee_rel_crop_train_mix": [
        ("g1-pipette-3view-hil293-eerel-train", 1.0,
         "unitree_g1_pipette_3view_oft_ee_rel_crop"),
    ],
    "unitree_g1_pipette_3view_oft_ee_rel_crop_eval_mix": [
        ("g1-pipette-3view-hil293-eerel-eval", 1.0,
         "unitree_g1_pipette_3view_oft_ee_rel_crop"),
    ],
    # Round 8 shares the dirs with rounds 6 and 7 for the same reason: its
    # crop is images-only, so the stats (and the {"mode": "rel"} cache key)
    # are identical and nothing is invalidated.
    "unitree_g1_pipette_3view_oft_ee_rel_crop3_train_mix": [
        ("g1-pipette-3view-hil293-eerel-train", 1.0,
         "unitree_g1_pipette_3view_oft_ee_rel_crop3"),
    ],
    "unitree_g1_pipette_3view_oft_ee_rel_crop3_eval_mix": [
        ("g1-pipette-3view-hil293-eerel-eval", 1.0,
         "unitree_g1_pipette_3view_oft_ee_rel_crop3"),
    ],
    "unitree_g1_pipette_3view_oft_ee_eval_mix": [
        ("g1-pipette-3view-hil293-ee-eval", 1.0, "unitree_g1_pipette_3view_oft_ee"),
    ],
    # ROUND 9 (2026-09-15): hil293 + plus21 under ONE 8-D chunk-relative action
    # space and ONE union q99 box (scratchpad build_wrist8rel_r9.py writes the
    # same rel-mode stats_gr00t.json into all four dirs). SEPARATE -wrist8rel-
    # dirs, not the -wrist8 ones: the stats cache key is {"mode": action_mode},
    # and rounds 4/5 still need the abs cache in the -wrist8 dirs intact.
    # WEIGHTS ARE POST-PAUSE-FILTER ANCHOR COUNTS (train) / frame counts (eval):
    # LeRobotMixtureDataset.sample_step draws the DATASET by these weights, so
    # they must be the natural pooled proportions, not 1.0/1.0.
    "unitree_g1_pipette_3view_oft_wrist8_rel_crop3_train_mix": [
        ("g1-pipette-3view-hil293-wrist8rel-train", 253594.0,
         "unitree_g1_pipette_3view_oft_wrist8_rel_crop3"),
        # 83,519 of 87,641 frames survive the 1.0 mm pause filter (95.30%,
        # measured by the round-9 smoke test); hil293's 253,594 is round 8's
        # exact count, which also cross-checks that wrist8's pose xyz equals
        # the -ee columns bit-for-bit.
        ("g1-pipette-3view-20260819-plus21-wrist8rel-train", 83519.0,
         "unitree_g1_pipette_3view_oft_wrist8_rel_crop3"),
    ],
    "unitree_g1_pipette_3view_oft_wrist8_rel_crop3_eval_mix": [
        ("g1-pipette-3view-hil293-wrist8rel-eval", 56115.0,
         "unitree_g1_pipette_3view_oft_wrist8_rel_crop3"),
        ("g1-pipette-3view-20260819-plus21-wrist8rel-eval", 14833.0,
         "unitree_g1_pipette_3view_oft_wrist8_rel_crop3"),
    ],
    # PURE-HIL arm of the R15 A/B: ONLY the 46 HG-DAgger correction
    # episodes (eps 78-123 of the merged R15 set, re-extracted to their
    # own directory so q99 stats are HIL-only). Fine-tuned from the R14
    # checkpoint like ft14 — the single variable vs ft14 is dropping the
    # 78 teleop episodes from the adaptation data.
    "unitree_g1_pipette_3view_mixres_nohist_hilonly46_n1d7_mix": [
        ("g1-pipette-3view-hilonly46", 1.0,
         "unitree_g1_pipette_3view_mixres_nohist_n1d7"),
    ],
    # HIL-194 (2026-09-10): ALL human-in-the-loop episodes to date — the 46
    # HG-DAgger corrections (hand action RECONSTRUCTED, see the dataset's
    # source_notes) + the 148 joystick-HIL full runs of 2026-09-09 (hand
    # action measured). 245,898 frames, green-tip no-eject text throughout,
    # wrist slots verified per episode against the R14 teleop reference.
    "unitree_g1_pipette_3view_mixres_nohist_hil194_n1d7_mix": [
        ("g1-pipette-3view-hil194", 1.0,
         "unitree_g1_pipette_3view_mixres_nohist_n1d7"),
    ],
    "unitree_g1_pipette_phase_train_n1d7": [
        ("g1-pipette-tip-teleop-phase-train", 1.0,
         "unitree_g1_pipette_tipcrop_waist_hist_phase_n1d7"),
    ],
}
