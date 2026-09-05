"""Serve a DELTA-action finetuned EgoVLA piston model (Option A + action_mode=delta) on :5555.

The model predicts joint DELTAS (action_type=delta_qpos): delta[0]=action[0]-current_state,
delta[t]=action[t]-action[t-1]. So absolute targets = current_state + cumsum(delta). Unlike
the absolute-target model, this can't collapse to a fixed point — each step advances.

This server:
  1. reads the robot's CURRENT joint state from the bridge obs (arms + hands);
  2. runs the model (image+language) -> un-normalized 16x26 deltas;
  3. absolute = current_state + cumsum(delta);
  4. CLAMPS the absolute targets to the demo action range (per-dim min/max from the
     dataset's ABSOLUTE action statistics) — a state-less delta policy integrates
     open-loop and can run away into joint-limit poses never seen in the data;
     the clamp bounds the failure to "stops at the demo envelope" instead;
  5. maps to the bridge contract (hands Inspire-6 <-> Dex3-7, base/navigate neutral).

usage:
  python examples/realRobots/EgoVLA/eval_files/server_piston_delta_g1.py \
      --ckpt_path <...>/final_model/pytorch_model.pt --port 5555 --use_bf16 --unnorm_key new_embodiment
"""
import argparse
import json
import logging

import numpy as np

try:  # standalone script: bootstrap the repo root onto sys.path (repo isn't pip-installed)
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:  # imported as a package module: root already on sys.path
    pass


def _inspire6_to_dex7(q):  # [pinky,ring,middle,index,thumb_pitch,thumb_yaw] -> Dex3-7
    pinky, ring, middle, index, tp, ty = q
    return np.array([index, index * 0.7, middle, middle * 0.7, ty, tp, tp * 0.5], dtype=np.float32)


def _dex7_to_inspire6(d):  # inverse used by the bridge (Dex3-7 -> Inspire-6 primary)
    idx0, _idx1, mid0, _mid1, th0, th1, _th2 = d
    avg = (idx0 + mid0) / 2.0
    return np.array([avg, avg, mid0, idx0, th1, th0], dtype=np.float32)  # pinky,ring,middle,index,tp,ty


def main(args) -> None:
    from deployment.model_server.gr00t_obs_adapter import Gr00tCompatPolicy
    from deployment.model_server.policy_wrapper import PolicyServerWrapper
    from deployment.model_server.tools.zmq_policy_server import ZmqGr00tPolicyServer

    # Demo-envelope clamp: per-dim [min, max] of the dataset's ABSOLUTE actions.
    clamp_lo = clamp_hi = None
    if args.clamp_stats:
        stats = json.load(open(args.clamp_stats))["new_embodiment"]["action"]
        clamp_lo = np.asarray(stats["min"], dtype=np.float32)
        clamp_hi = np.asarray(stats["max"], dtype=np.float32)
        assert clamp_lo.shape == (26,) and clamp_hi.shape == (26,)
        logging.warning("[clamp] absolute targets clamped to demo action range "
                        "(e.g. right_arm[0] in [%.3f, %.3f])", clamp_lo[7], clamp_hi[7])

    class PistonDeltaPolicy(Gr00tCompatPolicy):
        def _current_state26(self, obs):
            st = obs["state"]
            la = np.asarray(st["left_arm"]).reshape(-1)[:7]
            ra = np.asarray(st["right_arm"]).reshape(-1)[:7]
            lh = _dex7_to_inspire6(np.asarray(st["left_hand"]).reshape(-1)[:7])
            rh = _dex7_to_inspire6(np.asarray(st["right_hand"]).reshape(-1)[:7])
            return np.concatenate([la, ra, lh, rh]).astype(np.float32)  # (26,)

        def get_action(self, observation, options=None):
            cur = self._current_state26(observation)                      # (26,)
            example = self.obs_to_example(observation)                    # image + language
            delta = np.asarray(self._wrapper.predict_action(
                examples=[example], unnorm_key=self._unnorm_key)["actions"])[0]  # (16, 26) un-normalized delta
            absolute = cur[None, :] + np.cumsum(delta, axis=0)            # (16, 26) integrate
            if clamp_lo is not None:
                absolute = np.clip(absolute, clamp_lo[None, :], clamp_hi[None, :])
            # collection-time wrist offset (dataset README: roll=0.35, pitch=-0.1, yaw=0),
            # applied downstream of the (clamped) model actions, matching data collection.
            ro, po, yo = args.wrist_offset
            for base in (0, 7):
                absolute[:, base + 4] += ro
                absolute[:, base + 5] += po
                absolute[:, base + 6] += yo
            T = absolute.shape[0]
            la = absolute[:, 0:7][None].astype(np.float32)
            ra = absolute[:, 7:14][None].astype(np.float32)
            lh = np.stack([_inspire6_to_dex7(absolute[t, 14:20]) for t in range(T)])[None].astype(np.float32)
            rh = np.stack([_inspire6_to_dex7(absolute[t, 20:26]) for t in range(T)])[None].astype(np.float32)
            action = {
                "left_arm": la, "right_arm": ra, "left_hand": lh, "right_hand": rh,
                "base_height_command": np.zeros((1, T, 1), np.float32),
                "navigate_command": np.zeros((1, T, 3), np.float32),
            }
            return action, {"mode": "delta"}

    wrapper = PolicyServerWrapper(ckpt_path=args.ckpt_path, device="cuda",
                                  use_bf16=args.use_bf16, unnorm_key=args.unnorm_key)
    policy = PistonDeltaPolicy(wrapper, unnorm_key=args.unnorm_key, send_state=False,
                               fallback_instruction=args.fallback_instruction)
    logging.warning("[EgoVLA piston DELTA server] :%d — absolute = current_state + cumsum(delta). Bridge unchanged.",
                    args.port)
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
    p.add_argument("--clamp_stats", type=str,
                   default="/home/dhy/Projects/egovla_piston_ckpt/dataset_statistics.json",
                   help="dataset_statistics.json with ABSOLUTE action stats; clamps integrated "
                        "targets to its per-dim [min,max]. Pass '' to disable.")
    p.add_argument("--wrist_offset", type=float, nargs=3, default=[0.35, -0.1, 0.0],
                   metavar=("ROLL", "PITCH", "YAW"),
                   help="collection-time wrist offset re-applied at deployment (0 0 0 to disable).")
    return p


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
