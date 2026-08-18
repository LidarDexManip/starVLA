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

def resolve_groups(root: Path) -> tuple[dict, dict]:
    """Read ``meta/modality.json`` -> (state_groups, action_groups).

    Each maps a group's SHORT name to ``(column, start, end)``. Two layouts
    exist in the wild and both appear in this repo's datasets, which is why
    this is read rather than hard-coded:

      * a slice of the flat column -- ``{"left_arm": {"start": 15, "end": 22}}``
        indexes ``observation.state`` / ``action``. The piston datasets use
        this for every group.
      * a slice of a SEPARATE column -- the same record plus
        ``"original_key": "action.hand_right"``, which is its own parquet
        column with its own 0-based indices. The pipette datasets use this for
        the hand groups, whose registers were recorded out of band.

    Hard-coding the piston's flat 29/30 layout silently mis-sliced the pipette
    data (its flat ``action`` is 35-dim low-level joints, and the hands are not
    in it at all), so the groups now come from the dataset that is being
    replayed.
    """
    mod = json.loads((root / "meta" / "modality.json").read_text())
    resolved = []
    for section, flat_col in (("state", "observation.state"), ("action", "action")):
        groups = {}
        for name, rec in mod.get(section, {}).items():
            groups[name] = (rec.get("original_key", flat_col),
                            int(rec["start"]), int(rec["end"]))
        resolved.append(groups)
    return resolved[0], resolved[1]


def stack_groups(df, groups: dict) -> dict:
    """``{group: (T, dim) float32}`` for one episode, one read per column."""
    cols: dict[str, np.ndarray] = {}
    out = {}
    for name, (col, a, b) in groups.items():
        if col not in cols:
            if col not in df.columns:
                raise KeyError(
                    f"modality.json group {name!r} wants column {col!r}, absent "
                    f"from the parquet (columns: {sorted(df.columns)[:12]}...)")
            cols[col] = np.stack(df[col].values).astype(np.float32)
        arr = cols[col]
        out[name] = arr[:, a:b] if arr.ndim == 2 else arr[:, None][:, a:b]
    return out


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
    """Return (dataframe, {view_key: VideoReader}).

    One reader per view. ``meta/modality.json`` maps each short view name to its
    dataset column ("ego_view" -> "observation.images.ego_view"), and that column
    is the video subdirectory, so the short names the server declares are enough
    to find every stream.

    State and action are NOT unpacked here: their grouping is per-dataset (see
    ``resolve_groups``), so the caller slices with the layout it read.
    """
    pq = root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    df = pd.read_parquet(pq).sort_values("frame_index")
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
    return df, readers


def build_obs(frames: dict, state_t: dict, instruction: str) -> dict:
    """``frames``: {view_key: (H, W, 3) uint8}, already in the server's order.
    ``state_t``: {group: (n_hist, dim) float32} — the state HISTORY window this
    timestep, oldest row first. A single frame is ``(1, dim)``.

    Server-side the view order is positional, so this dict is built by iterating
    the server's declared ``video_keys`` — see main(). Only the groups the
    SERVER asks for are sent: this checkpoint family varies (the piston models
    take hands in state, the pipette ones do not, because those channels were a
    per-episode constant), and sending a group the server does not consume is
    silently ignored while omitting one it needs fails loudly.
    """
    return {
        "video": {k: f[None, None] for k, f in frames.items()},  # (1,1,H,W,3) uint8
        # (1, n_hist, dim): the shape the bridge puts on the wire. Sending a
        # single frame to a checkpoint whose head was BUILT for history is not
        # neutral -- the server pads by repeating the oldest row, so all frames
        # come out identical and the head reads velocity = 0 on every sample.
        # That understates exactly the feature a state-history round added, so
        # the window has to be replayed, not synthesised. See --state_history.
        "state": {k: np.asarray(v, np.float32)[None] for k, v in state_t.items()},
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
    p.add_argument("--plot_group", default="right_arm",
                   help="Action group to plot; validated against the dataset's "
                        "own modality.json (not a fixed list — layouts differ "
                        "per dataset). Plot the HAND group before trusting a "
                        "clean arm plot: arm traces look identical whether or "
                        "not the press/insert was learned. Piston: right_hand[5] "
                        "is the plunger thumb. Pipette: hand_right[4] is "
                        "thumb_bend, the tip ejector.")
    p.add_argument("--fps", type=float, default=None,
                   help="Frame rate for the plot's time axis. Default: read "
                        "from meta/info.json (piston 50, pipette 60).")
    p.add_argument("--press_dim", type=int, default=None,
                   help="Index WITHIN --plot_group to draw a threshold line on.")
    p.add_argument("--press_thresh", type=float, default=None,
                   help="Raw value for that line (piston plunger: 0.05).")
    p.add_argument("--state_history", type=int, nargs="+", default=[0],
                   help="state delta indices in DATASET ROWS, oldest first, "
                        "matching the checkpoint's training DataConfig. Default "
                        "[0] is a single current frame. The 0.6 s / 4-frame "
                        "pipette recipe is: --state_history -36 -24 -12 0")
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
    state_keys = [k.split(".", 1)[-1] for k in contract["state_keys"]]
    horizon = None
    print(f"[eval] server contract: video_keys={video_keys} "
          f"state_keys={state_keys} action_keys={action_keys}")

    # The DATASET's layout, not a compiled-in one.
    ds_state, ds_action = resolve_groups(args.dataset)
    missing_s = [k for k in state_keys if k not in ds_state]
    missing_a = [k for k in action_keys if k not in ds_action]
    if missing_s or missing_a:
        raise SystemExit(
            f"[eval] the server asks for groups this dataset does not define — "
            f"state {missing_s}, action {missing_a}. Available: "
            f"state={sorted(ds_state)} action={sorted(ds_action)}. "
            f"This is a checkpoint/dataset mismatch, not a flag to work around.")
    if args.plot_group not in ds_action:
        raise SystemExit(f"[eval] --plot_group {args.plot_group!r} not in "
                         f"{sorted(ds_action)}")
    fps = args.fps
    if fps is None:
        try:
            fps = float(json.loads(
                (args.dataset / "meta" / "info.json").read_text())["fps"])
        except Exception:
            fps = 50.0
    print(f"[eval] dataset layout: {len(ds_state)} state / {len(ds_action)} "
          f"action groups, {fps:g} fps")

    # accumulators: {group: [abs errors]} — only the groups the server emits
    chunk_err, step0_err, hold_err = ({k: [] for k in action_keys} for _ in range(3))
    plot_rows = []  # (t_abs, gt[plot_group], pred[plot_group]) for episode[0]

    for ei, ep in enumerate(args.episodes):
        df, readers = load_episode(args.dataset, ep, video_keys)
        S = stack_groups(df, {k: ds_state[k] for k in state_keys})
        A = stack_groups(df, {k: ds_action[k] for k in action_keys})
        T = min([len(df)] + [len(r) for r in readers.values()])
        n_req = 0
        for t in range(0, T, args.stride):
            # dict comprehension over video_keys preserves the declared order
            frames = {k: readers[k][t].asnumpy() for k in video_keys}  # (H,W,3) uint8 RGB
            # Clamp at the episode start rather than zero-padding: repeating
            # the oldest AVAILABLE row means "stationary", which is true at a
            # standstill, where a zero row would claim a lunge from the origin.
            # Same rule the dataloader and the bridge's ring buffer both use.
            hist = [max(t + d, 0) for d in args.state_history]
            state_t = {k: S[k][hist] for k in state_keys}
            reply = client.call("get_action",
                                {"observation": build_obs(frames, state_t, args.instruction)})
            action_dict = reply[0] if isinstance(reply, (list, tuple)) else reply
            if horizon is None:
                horizon = np.asarray(action_dict[action_keys[0]])[0].shape[0]
                print(f"[eval] horizon={horizon} steps per chunk")
            h = min(horizon, T - t)
            # Compare PER GROUP against the server's own keys rather than
            # concatenating: the flat order only matches when the dataset's
            # group order equals the checkpoint's, which is not guaranteed
            # across datasets and is exactly the kind of mismatch that reads
            # as a merely-bad policy.
            for k in action_keys:
                pred = np.asarray(action_dict[k])[0][:h]     # (h, dim)
                gt = A[k][t:t + h]                           # (h, dim)
                hold = np.repeat(A[k][t][None], h, 0)
                chunk_err[k].append(np.abs(pred - gt).mean())
                step0_err[k].append(np.abs(pred[0] - gt[0]).mean())
                hold_err[k].append(np.abs(hold - gt).mean())
                if ei == 0 and k == args.plot_group:
                    plot_rows.append((t, gt[0].copy(), pred[0].copy()))
            n_req += 1
        print(f"[eval] episode {ep}: {n_req} requests done (T={T})")

    print(f"\n{'group':<18}{'chunk MAE':>12}{'step0 MAE':>12}{'hold MAE':>12}  verdict")
    print("-" * 68)
    for k in action_keys:
        c, s0, hd = (float(np.mean(v[k])) for v in (chunk_err, step0_err, hold_err))
        if hd < 1e-9:                     # a group that never moves in this data
            verdict = "constant in GT"
        elif c < hd:
            verdict = "beats hold"
        elif c < hd * 1.15:
            verdict = "~hold"
        else:
            verdict = "WORSE than hold"
        print(f"{k:<18}{c:>12.4f}{s0:>12.4f}{hd:>12.4f}  {verdict}")
    call = float(np.mean([np.mean(chunk_err[k]) for k in action_keys]))
    hall = float(np.mean([np.mean(hold_err[k]) for k in action_keys]))
    print("-" * 68)
    print(f"{'ALL':<18}{call:>12.4f}{'':>12}{hall:>12.4f}")

    if args.plot and plot_rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ts = np.array([r[0] for r in plot_rows]) / fps  # seconds
        gt = np.stack([r[1] for r in plot_rows])
        pr = np.stack([r[2] for r in plot_rows])
        group = args.plot_group
        ndim = gt.shape[1]
        fig, axes = plt.subplots(ndim, 1, figsize=(10, 2 * ndim), sharex=True)
        axes = np.atleast_1d(axes)
        for j, ax in enumerate(axes):
            ax.plot(ts, gt[:, j], label="ground truth", lw=1.5)
            ax.plot(ts, pr[:, j], label="predicted (step0)", lw=1.2, ls="--")
            ax.set_ylabel(f"{group}[{j}]")
            # Threshold line on the channel that decides the task. It is
            # explicit rather than inferred from the group name: "the press
            # channel" is per-checkpoint (piston right_hand[5] at raw 0.05,
            # pipette hand_right[4] thumb_bend at raw 520), and a line drawn on
            # the wrong dim is worse than no line.
            if args.press_dim is not None and j == args.press_dim \
                    and args.press_thresh is not None:
                ax.axhline(args.press_thresh, color="crimson", lw=1.0, ls=":",
                           label=f"press threshold (raw {args.press_thresh:g})")
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
