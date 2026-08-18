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
    VideoColorJitter,
    VideoColorTemperature,
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
    # Round 5: the merged 77-episode phase-labelled set. Point at a split of it
    # once one exists; this entry is the whole thing.
    "unitree_g1_pipette_phase_n1d7": [
        ("g1-pipette-tip-teleop-phase", 1.0,
         "unitree_g1_pipette_tipcrop_waist_hist_phase_n1d7"),
    ],
}
