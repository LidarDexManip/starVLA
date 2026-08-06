#!/usr/bin/env python3
"""Open-loop replay evaluation against a LIVE starVLA GR00T-compat ZMQ server.

Feeds recorded dataset observations (ego video frame + state + language) to the
policy server over the real wire protocol and compares the predicted action
chunk against the dataset's ground-truth actions — no sim, no robot. This
isolates "did the policy learn the task?" from every sim-transfer question
(camera pose, lighting, physics).

Metrics reported per action group and overall:
  - chunk MAE  : mean |pred - gt| over the whole predicted horizon
  - step0 MAE  : mean |pred[0] - gt[0]| (the action the robot would execute
                 first in a receding-horizon deployment)
  - hold MAE   : baseline that repeats the current GT action across the
                 horizon. A useful policy must beat this on moving joints —
                 otherwise it only learned to copy the present, not predict.

Usage (server must be running, e.g. on :5555):
    python deployment/model_server/tools/replay_openloop_eval.py \
        --dataset ~/Datasets/pickplace_hf/g1-inspire-piston-pick-place \
        --episodes 0 5 10 --stride 5 --plot outputs/openloop_eval.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import zmq

# Reuse the server's own msgpack ndarray codec so the wire format is identical.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from deployment.model_server.tools.zmq_policy_server import pack, unpack  # noqa: E402

LANG_KEY = "annotation.human.task_description"

# Dataset modality layout (meta/modality.json): [start, end) into the flat rows.
STATE_GROUPS = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 20),
                "right_hand": (20, 26), "waist": (26, 29)}
ACTION_GROUPS = {"left_arm": (0, 7), "right_arm": (7, 14), "left_hand": (14, 20),
                 "right_hand": (20, 26), "base_height": (26, 27),
                 "navigate_command": (27, 30)}


class ZmqClient:
    def __init__(self, host: str, port: int, timeout_s: float = 30.0):
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
        self._sock.setsockopt(zmq.SNDTIMEO, int(timeout_s * 1000))
        self._sock.connect(f"tcp://{host}:{port}")

    def call(self, endpoint: str, data: dict | None = None):
        req = {"endpoint": endpoint}
        if data is not None:
            req["data"] = data
        self._sock.send(pack(req))
        rep = unpack(self._sock.recv())
        if isinstance(rep, dict) and "error" in rep:
            raise RuntimeError(f"server error: {rep['error']}")
        return rep


def load_episode(root: Path, ep: int, video_keys: list[str]):
    """Return (state[T,·], action[T,30], {view_key: VideoReader}).

    One reader per view. ``meta/modality.json`` maps each short view name to its
    dataset column ("ego_view" -> "observation.images.ego_view"), and that column
    is the video subdirectory, so the short names the server declares are enough
    to find every stream.
    """
    pq = root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    df = pd.read_parquet(pq)
    S = np.stack(df["observation.state"].values).astype(np.float32)
    A = np.stack(df["action"].values).astype(np.float32)             # (T, 30)
    import decord
    readers = {}
    for key in video_keys:
        mp4 = root / "videos" / "chunk-000" / f"observation.images.{key}" / f"episode_{ep:06d}.mp4"
        if not mp4.exists():
            raise FileNotFoundError(
                f"view {key!r} has no video at {mp4}. The server declares views "
                f"{video_keys}; this dataset does not carry all of them, so the "
                f"replay would feed the policy a different camera set than it was "
                f"trained on."
            )
        readers[key] = decord.VideoReader(str(mp4))
    return S, A, readers


def build_obs(frames: dict, state_row: np.ndarray, instruction: str) -> dict:
    """``frames``: {view_key: (H, W, 3) uint8}, already in the server's order.

    Server-side the view order is positional, so this dict is built by iterating
    the server's declared ``video_keys`` — see main().
    """
    return {
        "video": {k: f[None, None] for k, f in frames.items()},  # (1,1,H,W,3) uint8
        "state": {k: state_row[a:b][None, None] for k, (a, b) in STATE_GROUPS.items()},
        "language": {LANG_KEY: [[instruction]]},
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path,
                   default=Path.home() / "Datasets/pickplace_hf/g1-inspire-piston-pick-place")
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--episodes", type=int, nargs="+", default=[0])
    p.add_argument("--stride", type=int, default=5,
                   help="Evaluate every Nth frame (50 fps data; 5 = 10 Hz).")
    p.add_argument("--instruction", default="pick up the piston.")
    p.add_argument("--plot", type=Path, default=None,
                   help="Save a GT-vs-pred trajectory plot (PNG).")
    p.add_argument("--plot_group", default="right_arm", choices=sorted(ACTION_GROUPS),
                   help="Action group to plot. right_hand is where the plunger "
                        "press lives (dim 25 == right_hand[5], the thumb): the "
                        "arm traces look identical whether or not the press was "
                        "learned, so check that group before trusting a clean "
                        "arm plot.")
    p.add_argument("--video_keys", nargs="+", default=None,
                   help="Camera views to replay, IN ORDER. Default: whatever the "
                        "server declares (the checkpoint's training order), which "
                        "is what you want — pass this only to probe a mismatch.")
    args = p.parse_args()

    client = ZmqClient(args.host, args.port)
    assert client.call("ping").get("status") == "ok" or True  # any reply = alive
    contract = client.call("get_modality_config")
    action_keys = [k.split(".", 1)[-1] for k in contract["action_keys"]]
    # The server is the authority on view order: identity is positional, so the
    # replay must lay views out exactly as the checkpoint was trained.
    served = list(contract.get("video_keys") or [])
    video_keys = args.video_keys or served or ["ego_view"]
    if served and list(video_keys) != served:
        print(f"[eval] WARNING: replaying views {video_keys} but the server "
              f"declares {served}. Results are only meaningful if you are "
              f"deliberately testing a mismatch.")
    horizon = None
    print(f"[eval] server contract: video_keys={video_keys} action_keys={action_keys}")

    # accumulators: {group: [abs errors]}
    chunk_err, step0_err, hold_err = ({k: [] for k in ACTION_GROUPS} for _ in range(3))
    plot_rows = []  # (t_abs, gt_right_arm(7), pred_right_arm(7)) for episode[0]

    for ei, ep in enumerate(args.episodes):
        S, A, readers = load_episode(args.dataset, ep, video_keys)
        T = min([len(A)] + [len(r) for r in readers.values()])
        n_req = 0
        for t in range(0, T, args.stride):
            # dict comprehension over video_keys preserves the declared order
            frames = {k: readers[k][t].asnumpy() for k in video_keys}  # (H,W,3) uint8 RGB
            reply = client.call("get_action",
                                {"observation": build_obs(frames, S[t], args.instruction)})
            action_dict = reply[0] if isinstance(reply, (list, tuple)) else reply
            pred = np.concatenate(
                [np.asarray(action_dict[k])[0] for k in action_keys], axis=-1)  # (H, 30)
            if horizon is None:
                horizon = pred.shape[0]
                print(f"[eval] horizon={horizon} steps per chunk")
            h = min(horizon, T - t)
            gt = A[t:t + h]                      # (h, 30)
            hold = np.repeat(A[t][None], h, 0)   # hold current action
            for k, (a, b) in ACTION_GROUPS.items():
                chunk_err[k].append(np.abs(pred[:h, a:b] - gt[:, a:b]).mean())
                step0_err[k].append(np.abs(pred[0, a:b] - gt[0, a:b]).mean())
                hold_err[k].append(np.abs(hold[:, a:b] - gt[:, a:b]).mean())
            if ei == 0:
                pa, pb = ACTION_GROUPS[args.plot_group]
                plot_rows.append((t, gt[0, pa:pb], pred[0, pa:pb]))
            n_req += 1
        print(f"[eval] episode {ep}: {n_req} requests done (T={T})")

    print(f"\n{'group':<18}{'chunk MAE':>12}{'step0 MAE':>12}{'hold MAE':>12}  verdict")
    print("-" * 68)
    for k in ACTION_GROUPS:
        c, s0, hd = (float(np.mean(v[k])) for v in (chunk_err, step0_err, hold_err))
        verdict = "beats hold" if c < hd else ("~hold" if c < hd * 1.15 else "WORSE than hold")
        print(f"{k:<18}{c:>12.4f}{s0:>12.4f}{hd:>12.4f}  {verdict}")
    call = float(np.mean([np.mean(chunk_err[k]) for k in ACTION_GROUPS]))
    hall = float(np.mean([np.mean(hold_err[k]) for k in ACTION_GROUPS]))
    print("-" * 68)
    print(f"{'ALL (rad)':<18}{call:>12.4f}{'':>12}{hall:>12.4f}")

    if args.plot and plot_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = np.array([r[0] for r in plot_rows]) / 50.0  # seconds
        gt = np.stack([r[1] for r in plot_rows])
        pr = np.stack([r[2] for r in plot_rows])
        group = args.plot_group
        ndim = gt.shape[1]
        fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
        axes = np.atleast_1d(axes)
        for j, ax in enumerate(axes):
            ax.plot(ts, gt[:, j], label="ground truth", lw=1.5)
            ax.plot(ts, pr[:, j], label="predicted (step0)", lw=1.2, ls="--")
            ax.set_ylabel(f"{group}[{j}] rad")
            # The press-frame threshold, drawn where it actually applies: a
            # frame counts as pressing when the right thumb is lifted off its
            # -0.1 rest past 0.05 raw (= -0.143 normalised, the
            # grasp_weight_thresh the 3x loss weighting keys on).
            if group == "right_hand" and j == 5:
                ax.axhline(0.05, color="crimson", lw=1.0, ls=":",
                           label="press threshold (raw 0.05)")
                ax.legend(loc="upper right", fontsize=8)
        axes[0].set_title(f"{group} — episode {args.episodes[0]} "
                          f"(open-loop, step-0 of each chunk)")
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("time (s)")
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=110)
        print(f"[eval] plot saved: {args.plot}")


if __name__ == "__main__":
    main()
