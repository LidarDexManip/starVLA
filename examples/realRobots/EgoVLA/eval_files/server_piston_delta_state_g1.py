"""Serve the deltav2 state-conditioned EgoVLA piston model (DELTA joints + pi0.5-style
discretised state token) on :5555.

Deployment contract (per trainer, differs from absv2):
  * action_mode = DELTA: model outputs increments; absolute = current_qpos + cumsum(delta).
  * state token fed (pi0.5, num_bins=256); inference dropout=0 (always injected — the
    framework only drops during training).
  * state normalized with THIS package's dataset_statistics.json (q99, clip ±2.2).

Combines server_piston_state_g1 (state feeding: leg-fill, Dex3-7->inspire-6, q99+clip)
with server_piston_delta_g1 (delta integration). `current_state26` is read BEFORE
obs_to_example mutates the hand state.

usage:
  python examples/realRobots/EgoVLA/eval_files/server_piston_delta_state_g1.py \
      --ckpt_path <...>/final_model/pytorch_model.pt --port 5555 --use_bf16 \
      --unnorm_key new_embodiment \
      --state_stats /home/dhy/Projects/egovla_piston_deltav2/dataset_statistics.json
"""
import argparse
import json
import logging

import numpy as np

try:  # standalone script: bootstrap the repo root onto sys.path (repo isn't pip-installed)
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # imported as a package module: root already on sys.path
    pass


def _inspire6_to_dex7(q: np.ndarray) -> np.ndarray:
    pinky, ring, middle, index, thumb_pitch, thumb_yaw = q
    return np.array([index, index * 0.7, middle, middle * 0.7, thumb_yaw, thumb_pitch, thumb_pitch * 0.5],
                    dtype=np.float32)


def _dex7_to_inspire6(d: np.ndarray) -> np.ndarray:
    idx0, _idx1, mid0, _mid1, th0, th1, _th2 = d
    avg = (idx0 + mid0) / 2.0
    return np.array([avg, avg, mid0, idx0, th1, th0], dtype=np.float32)


def main(args) -> None:
    from deployment.model_server.gr00t_obs_adapter import Gr00tCompatPolicy
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.tools.zmq_policy_server import ZmqGr00tPolicyServer

    stats = json.load(open(args.state_stats))["new_embodiment"]["state"]
    q01 = np.asarray(stats["q01"], np.float32)
    q99 = np.asarray(stats["q99"], np.float32)
    mean = np.asarray(stats["mean"], np.float32)
    span = q99 - q01
    nz = span != 0
    _LEG_FILL = {"left_leg": mean[0:6], "right_leg": mean[6:12]}  # N1.7 bridge omits legs

    # Optional clamp on integrated absolute targets (default OFF: state should self-correct
    # the drift that plagued the stateless delta model).
    clamp_lo = clamp_hi = None
    if args.clamp_stats:
        cs = json.load(open(args.clamp_stats))["new_embodiment"]["action"]
        clamp_lo = np.asarray(cs["min"], np.float32)
        clamp_hi = np.asarray(cs["max"], np.float32)
        logging.warning("[clamp] integrated targets clamped to %s action range", args.clamp_stats)

    logging.warning("[state] q99-normalizing %d-dim state (clip ±2.2); %d constant dims pass-through",
                    q01.shape[0], int((~nz).sum()))

    class PistonDeltaStatePolicy(Gr00tCompatPolicy):
        def _current_state26(self, obs):
            st = obs["state"]
            la = np.asarray(st["left_arm"]).reshape(-1)[:7]
            ra = np.asarray(st["right_arm"]).reshape(-1)[:7]
            lh = _dex7_to_inspire6(np.asarray(st["left_hand"]).reshape(-1)[:7])
            rh = _dex7_to_inspire6(np.asarray(st["right_hand"]).reshape(-1)[:7])
            return np.concatenate([la, ra, lh, rh]).astype(np.float32)  # (26,) current qpos

        def obs_to_example(self, observation):
            st_in = observation.get("state")
            if isinstance(st_in, dict):
                for key, fill in _LEG_FILL.items():
                    if key not in st_in:
                        st_in[key] = fill.reshape(1, 1, -1)
                for hk in ("left_hand", "right_hand"):
                    h = np.asarray(st_in.get(hk)).reshape(-1)
                    if h.shape[0] == 7:
                        st_in[hk] = _dex7_to_inspire6(h).reshape(1, 1, 6)
            example = super().obs_to_example(observation)
            st = example.get("state")
            if st is not None:
                x = np.asarray(st, np.float32)
                norm = x.copy()
                norm[..., nz] = 2 * (x[..., nz] - q01[nz]) / span[nz] - 1
                norm[..., ~nz] = x[..., ~nz]
                example["state"] = np.clip(norm, -2.2, 2.2).astype(np.float32)
            return example

        def get_action(self, observation, options=None):
            cur = self._current_state26(observation)          # BEFORE obs_to_example mutates hands
            example = self.obs_to_example(observation)          # image + lang + [STATE] token
            delta = np.asarray(self._wrapper.predict_action(
                examples=[example], unnorm_key=self._unnorm_key)["actions"])[0]  # (T, 26) delta
            absolute = cur[None, :] + np.cumsum(delta, axis=0)  # integrate
            if clamp_lo is not None:
                absolute = np.clip(absolute, clamp_lo[None, :], clamp_hi[None, :])
            T = absolute.shape[0]
            la = absolute[:, 0:7][None].astype(np.float32)
            ra = absolute[:, 7:14][None].astype(np.float32)
            lh = np.stack([_inspire6_to_dex7(absolute[t, 14:20]) for t in range(T)])[None].astype(np.float32)
            rh = np.stack([_inspire6_to_dex7(absolute[t, 20:26]) for t in range(T)])[None].astype(np.float32)
            return {
                "left_arm": la, "right_arm": ra, "left_hand": lh, "right_hand": rh,
                "base_height_command": np.zeros((1, T, 1), np.float32),
                "navigate_command": np.zeros((1, T, 3), np.float32),
            }, {"mode": "delta+state"}

    wrapper = PolicyServerWrapper(ckpt_path=args.ckpt_path, device="cuda",
                                  use_bf16=args.use_bf16, unnorm_key=args.unnorm_key)
    policy = PistonDeltaStatePolicy(wrapper, unnorm_key=args.unnorm_key, send_state=True,
                                    fallback_instruction=args.fallback_instruction)
    logging.warning("[EgoVLA piston DELTA+STATE server] :%d — delta integration + pi0.5 state token.", args.port)
    server = ZmqGr00tPolicyServer(policy, host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        server.stop()


def build_argparser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--use_bf16", action="store_true")
    p.add_argument("--unnorm_key", type=str, default="new_embodiment")
    p.add_argument("--fallback_instruction", type=str, default="pick up the piston and place it in the pot")
    p.add_argument("--state_stats", type=str,
                   default="/home/dhy/Projects/egovla_piston_deltav2/dataset_statistics.json")
    p.add_argument("--clamp_stats", type=str, default="",
                   help="optional dataset_statistics.json with ABSOLUTE action min/max to clamp "
                        "integrated targets (default off).")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
