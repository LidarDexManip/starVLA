#!/usr/bin/env python3
"""Diagnose closed-loop stalls: compare the LIVE Isaac camera frames against
training-dataset frames, visually and through the policy.

Does three things:
  1. Grabs one frame per view from the Isaac camera ZMQ stream (:5556), on
     whichever wire the scene publishes (bare JPEG for a 1-camera task, keyed
     msgpack for a -WristCam- one).
  2. Saves them beside dataset frames (early/mid episode) as a PNG — eyeball
     viewpoint, lighting, colours, flip, scene content, PER VIEW.
  3. Sends BOTH frame sets through the live policy server (:5555) and prints
     what each produces, for the right ARM and the right HAND. If the dataset
     frames yield task-consistent actions while the sim frames yield a stuck
     or degenerate pose, the visual domain gap is PROVEN, not suspected.

WHY THE HAND IS REPORTED SEPARATELY: an arm that reaches correctly while the
hand never closes is the signature failure of a visually-triggered grasp — the
reach is driven by the whole scene and survives a domain shift the close-up
grasp cue does not. Reporting only the arm hides exactly that case.

Run while the Isaac loop and the policy server are BOTH up:
    python deployment/model_server/tools/sim_vs_dataset_frame_check.py \
        --dataset ~/Datasets/g1-inspire-piston-longhorizon-v3-wristcam \
        --camera_format msgpack
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from deployment.model_server.tools.replay_openloop_eval import (  # noqa: E402
    ACTION_GROUPS, ZmqClient, build_obs, load_episode,
)


def grab_isaac_frames(host: str, port: int, video_keys, camera_format: str,
                      timeout_s: float = 15.0) -> dict:
    """{view: RGB} from the live sim, on either camera wire."""
    import base64

    import cv2
    ctx = zmq.Context()
    s = ctx.socket(zmq.SUB)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.setsockopt(zmq.CONFLATE, 1)
    s.connect(f"tcp://{host}:{port}")
    if not s.poll(timeout=int(timeout_s * 1000)):
        raise SystemExit(f"[check] no camera frame in {timeout_s:.0f}s — is the "
                         f"Isaac loop up on :{port}?")
    payload = s.recv()

    def _dec(buf):
        img = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit("[check] camera payload did not decode as JPEG")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # policy expects RGB

    if camera_format == "raw_jpeg":
        return {video_keys[0]: _dec(payload)}

    import msgpack
    # NO ndarray object_hook, deliberately. ImageMessageSchema carries base64
    # JPEG strings and float timestamps — nothing that needs one — and asking
    # for `msgpack_numpy` here resolves to the openpi-derived STUB that sits
    # beside this file (tools/msgpack_numpy.py, which exposes pack_array/
    # unpack_array, not the upstream encode/decode) because a script's own
    # directory leads sys.path. Plain unpackb has neither problem.
    msg = msgpack.unpackb(payload, raw=False)
    images = msg.get("images") or {}
    out = {}
    for key in video_keys:
        raw = images.get(key)
        if raw is None:
            continue
        out[key] = _dec(base64.b64decode(raw) if isinstance(raw, str) else bytes(raw))
    missing = [k for k in video_keys if k not in out]
    if missing:
        raise SystemExit(
            f"[check] sim published {sorted(images)} but the server wants "
            f"{list(video_keys)} — missing {missing}. That alone would explain a "
            f"stalled policy; fix the camera wire before reading anything else.")
    return out


def predict(client: ZmqClient, frames: dict, state_row: np.ndarray,
            instruction: str) -> dict:
    reply = client.call("get_action",
                        {"observation": build_obs(frames, state_row, instruction)})
    action_dict = reply[0] if isinstance(reply, (list, tuple)) else reply
    return {k: np.asarray(action_dict[k])[0] for k in ("right_arm", "right_hand")}


#: right_arm slice in the sim's 41-DoF canonical q vector
#: (vla_policy_bridge/robots/g1_inspire.py::GROUPS).
Q_RIGHT_ARM = slice(28, 35)


def _as_array(obj) -> np.ndarray:
    """Decode a msgpack-numpy encoded ndarray, or pass a plain list through.

    The sim packs its state with ``default=mnp.encode``, so ``q`` arrives as
    ``{b'nd': True, b'type': '<f4', b'shape': [41], b'data': b'...'}`` rather
    than a list. Decoded by hand on purpose: importing ``msgpack_numpy`` from
    this directory resolves to the openpi-derived STUB next to this file, whose
    encoding uses different keys entirely (see grab_isaac_frames).
    """
    if isinstance(obj, dict):
        k = {(kk.decode() if isinstance(kk, bytes) else kk): vv for kk, vv in obj.items()}
        if k.get("nd"):  # msgpack-numpy
            return np.frombuffer(k["data"], dtype=np.dtype(k["type"])).reshape(k["shape"])
        if k.get("__ndarray__"):  # openpi stub spelling
            return np.frombuffer(k["data"], dtype=np.dtype(k["dtype"])).reshape(k["shape"])
        raise ValueError(f"unrecognised encoded array: {sorted(k)[:5]}")
    return np.asarray(obj, dtype=np.float32)


def grab_sim_right_arm(host: str, port: int, timeout_s: float = 10.0):
    """The robot's CURRENT right-arm joints from the sim state stream (:5557)."""
    import msgpack
    ctx = zmq.Context()
    s = ctx.socket(zmq.SUB)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.setsockopt(zmq.CONFLATE, 1)
    s.connect(f"tcp://{host}:{port}")
    if not s.poll(timeout=int(timeout_s * 1000)):
        return None
    q = _as_array(msgpack.unpackb(s.recv(), raw=False)["q"]).astype(np.float32)
    return q[Q_RIGHT_ARM] if q.size >= 35 else None


def match_dataset_frame(A: np.ndarray, arm: np.ndarray) -> int:
    """Index of the dataset frame whose COMMANDED right arm is nearest ``arm``.

    Comparing sim and dataset frames at t=mid confounds two things: the sim is
    early in the task and the dataset frame shows a completed grasp, so a swap
    moves both the RENDERING and the PHASE. Matching on arm pose holds the
    phase roughly fixed, leaving the rendering as the only difference — which
    is the thing being tested.
    """
    return int(np.argmin(np.linalg.norm(A[:, 7:14] - arm[None, :], axis=1)))


def _fingers(pred) -> float:
    """Mean of right_hand[0..3] at step 0 — 0 = open, 1.3 = full grasp."""
    return float(np.mean(pred["right_hand"][0][:4]))


def _ablate(client, sim, ds, state_row, gt_row, video_keys, args, tag="t=mid"):
    """Swap ONE view at a time between sim and dataset; report the grasp.

    Isolates WHICH camera breaks the policy. Each combination is sampled
    ``--ablate_repeats`` times and averaged, because this head is a
    flow-matching sampler seeded from fresh noise on every call
    (GR00TN1d7ActionHead.predict_action starts at torch.randn), so a single
    draw is a sample and not the model's answer.

    THE CONFOUND, stated because it changes how the result reads: the sim and
    dataset frames show DIFFERENT task progress, so a mixed set is not a scene
    that ever existed. What the numbers still tell you is which view the grasp
    decision is SENSITIVE to — if replacing just one sim view with its dataset
    counterpart recovers the grasp, that view is carrying the cue and is the
    one whose sim rendering to distrust.
    """
    ha, hb = ACTION_GROUPS["right_hand"]
    n = args.ablate_repeats
    gt_grasp = float(np.mean(gt_row[ha:hb][:4]))
    print(f"\nVIEW ABLATION — mean right_hand[0..3] over {n} samples "
          f"(0 = open, {gt_grasp:.2f} = this frame's GT)")
    if gt_grasp < 0.2:
        # The metric is "how much grasp does this view recover". At a phase
        # where the demo hand is OPEN there is no grasp to recover, so every
        # row sits at ~0 and the block reads like "no view matters" when it
        # actually means "nothing was measured". Say so.
        print("  ** GT hand is OPEN at this phase — this ablation CANNOT")
        print("     detect a grasp difference. Every row will read ~0 whatever")
        print("     the rendering does. Re-run while the robot is IN the grasp")
        print("     or press phase (the demo closes at ~2.2s and holds), or")
        print("     pass --ablate_frame to pin a grasped reference frame.")

    def run(label, frames):
        vals = [_fingers(predict(client, frames, state_row, args.instruction))
                for _ in range(n)]
        print(f"  {label:<44}{np.mean(vals):+7.3f}  (sd {np.std(vals):.3f})")
        return float(np.mean(vals))

    base_sim = run("all SIM", sim)
    base_ds = run(f"all DATASET ({tag})", ds)
    print("  " + "-" * 52)
    for key in video_keys:
        run(f"SIM, {key} <- dataset", {**sim, key: ds[key]})
    print("  " + "-" * 52)
    for key in video_keys:
        run(f"DATASET, {key} <- sim", {**ds, key: sim[key]})
    print(f"\n  baseline gap: all-SIM {base_sim:+.3f} vs all-DATASET {base_ds:+.3f}")
    print("  A row in the FIRST block near the dataset baseline names the view")
    print("  that was suppressing the grasp. A row in the SECOND block that")
    print("  collapses toward 0 names the same view from the other direction —")
    print("  the two should agree, and agreeing is what makes it a finding.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path,
                   default=Path.home() / "Datasets/g1-inspire-piston-longhorizon-v3-wristcam")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--server_port", type=int, default=5555)
    p.add_argument("--camera_port", type=int, default=5556)
    p.add_argument("--state_port", type=int, default=5557,
                   help="Sim state stream, used to phase-match the "
                        "dataset frame to the robot's current arm.")
    p.add_argument("--camera_format", choices=("msgpack", "raw_jpeg"),
                   default="msgpack")
    p.add_argument("--instruction",
                   default="pick up the piston with the right hand, inject it "
                           "into the tube held by the left hand, then move it "
                           "over the hole plate.")
    p.add_argument("--out", type=Path,
                   default=Path("outputs/sim_vs_dataset_frames.png"))
    p.add_argument("--ablate", action="store_true",
                   help="Swap one view at a time between sim and dataset to "
                        "find WHICH camera suppresses the grasp.")
    p.add_argument("--ablate_frame", type=int, default=None,
                   help="Pin the dataset reference to this frame index instead "
                        "of phase-matching on arm pose. Use a GRASPED frame "
                        "(the demo closes at ~2.2s = frame 110+) when you want "
                        "the ablation to measure the grasp cue.")
    p.add_argument("--ablate_repeats", type=int, default=8,
                   help="Samples per combination. >1 because the flow-matching "
                        "head reseeds from noise on every call.")
    args = p.parse_args()

    client = ZmqClient("localhost", args.server_port)
    contract = client.call("get_modality_config")
    video_keys = list(contract.get("video_keys") or ["ego_view"])
    print(f"[check] server wants views {video_keys}")

    sim = grab_isaac_frames("localhost", args.camera_port, video_keys,
                            args.camera_format)
    for k, v in sim.items():
        print(f"[check] Isaac {k}: {v.shape}")

    S, A, readers = load_episode(args.dataset, args.episode, video_keys)
    T = min(len(A), min(len(r) for r in readers.values()))
    t_mid = T // 2
    ds0 = {k: r[0].asnumpy() for k, r in readers.items()}
    dsm = {k: r[t_mid].asnumpy() for k, r in readers.items()}
    for k, v in ds0.items():
        print(f"[check] dataset {k}: {v.shape}")

    # -- side-by-side PNG: one ROW per view, one COLUMN per source ------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [("ISAAC LIVE (what the policy sees now)", sim),
            (f"DATASET ep{args.episode} t=0", ds0),
            (f"DATASET ep{args.episode} t=mid", dsm)]
    fig, axes = plt.subplots(len(video_keys), 3,
                             figsize=(15, 3.6 * len(video_keys)), squeeze=False)
    for r, key in enumerate(video_keys):
        for c, (title, src) in enumerate(cols):
            ax = axes[r][c]
            ax.imshow(src[key])
            ax.set_title(f"{key}\n{title}" if r == 0 else key, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=110)
    print(f"[check] side-by-side saved: {args.out}")

    # -- policy reaction to each frame set ------------------------------------
    zero_state = np.zeros(63, dtype=np.float32)  # --no_state ckpt ignores it
    np.set_printoptions(precision=3, suppress=True)
    got = [("sim frames", predict(client, sim, zero_state, args.instruction), None),
           ("dataset t=0", predict(client, ds0, S[0], args.instruction), 0),
           ("dataset t=mid", predict(client, dsm, S[t_mid], args.instruction), t_mid)]

    ha, hb = ACTION_GROUPS["right_hand"]
    print("\npredicted right_arm step-0 (rad):")
    for name, pred, t in got:
        gt = "" if t is None else f"  | GT: {A[t, 7:14]}"
        print(f"  {name:<14}: {pred['right_arm'][0]}{gt}")

    print("\npredicted right_hand step-0 (rad)   [fingers 0..3: 0=open, 1.3=grasp]:")
    for name, pred, t in got:
        gt = "" if t is None else f"  | GT: {A[t, ha:hb]}"
        print(f"  {name:<14}: {pred['right_hand'][0]}{gt}")

    print("\nmotion WITHIN the predicted chunk (|last-first|, rad):")
    print(f"  {'source':<14}{'right_arm':>12}{'right_hand':>12}")
    for name, pred, _ in got:
        arm = float(np.abs(pred["right_arm"][-1] - pred["right_arm"][0]).sum())
        hand = float(np.abs(pred["right_hand"][-1] - pred["right_hand"][0]).sum())
        print(f"  {name:<14}{arm:>12.4f}{hand:>12.4f}")

    # -- PHASE-MATCHED comparison: same arm pose, only the rendering differs --
    arm = grab_sim_right_arm("localhost", args.state_port)
    if arm is None:
        print(f"\n[check] no sim state on :{args.state_port} — skipping the "
              f"phase-matched comparison (the t=mid numbers above still carry "
              f"the phase confound; read them as a ranking, not a magnitude).")
        t_match = None
    else:
        t_match = match_dataset_frame(A[:T], arm)
        dsq = {k: r[t_match].asnumpy() for k, r in readers.items()}
        print(f"\nPHASE-MATCHED — sim right_arm {np.round(arm, 3)}")
        print(f"  nearest dataset frame: {t_match} (t={t_match / 50:.2f}s), "
              f"arm {np.round(A[t_match, 7:14], 3)}, "
              f"residual {np.linalg.norm(A[t_match, 7:14] - arm):.3f} rad")
        print(f"  GT right_hand there: {np.round(A[t_match, ha:hb], 3)}")
        pm = predict(client, dsq, S[t_match], args.instruction)
        print(f"  dataset@match -> right_hand {np.round(pm['right_hand'][0], 3)}")
        print(f"  sim           -> right_hand "
              f"{np.round(got[0][1]['right_hand'][0], 3)}")
        print(f"  dataset@match -> right_arm  {np.round(pm['right_arm'][0], 3)}")
        print(f"  sim           -> right_arm  "
              f"{np.round(got[0][1]['right_arm'][0], 3)}")
        print("  Same phase, different rendering: if the dataset frame ADVANCES")
        print("  the arm and the sim frame returns ~the current pose, the stall")
        print("  is the rendering and not the phase.")

    if args.ablate:
        # Ablate against the phase-matched frame when we have one — otherwise
        # the swap moves rendering AND phase together and cannot separate them.
        if args.ablate_frame is not None:
            tf = max(0, min(int(args.ablate_frame), T - 1))
            ref = {k: r[tf].asnumpy() for k, r in readers.items()}
            ref_state, ref_gt, tag = S[tf], A[tf], f"t={tf} (pinned)"
        elif t_match is not None:
            ref, ref_state, ref_gt, tag = dsq, S[t_match], A[t_match], f"t={t_match}"
        else:
            ref, ref_state, ref_gt, tag = dsm, S[t_mid], A[t_mid], "t=mid"
        _ablate(client, sim, ref, ref_state, ref_gt, video_keys, args, tag)

    print("\nReading:")
    print("  * sim motion ~0 while dataset frames move -> the policy is frozen by")
    print("    the visual gap; the PNG shows which view differs and how.")
    print("  * sim ARM moves but sim HAND fingers stay near 0 while the dataset")
    print("    frames drive them toward 1.3 -> the reach survives the domain shift")
    print("    and the GRASP CUE does not. Compare the WRIST rows in the PNG")
    print("    first: that is where a close-up grasp cue lives.")


if __name__ == "__main__":
    main()
