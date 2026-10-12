# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""GR00T-protocol policy server (ZMQ) for starVLA checkpoints.

Drop-in replacement for Isaac-GR00T's ``gr00t/eval/run_gr00t_server.py``:
binds a ZMQ REP socket speaking the GR00T N1.6 msgpack protocol, so existing
GR00T clients — in particular the GR00T-WBC-Bridge deploy script
(``gr00t_n1_wbc_bridge_deploy.py`` with its default ``--server_codec custom``)
— work against a starVLA checkpoint with zero client changes.

The observation/action contract is derived from the checkpoint's training
DataConfig (state key order + per-key dims, action key split), so serving a
new embodiment only requires registering its DataConfig and training with it.

Example::

    python deployment/model_server/server_policy_gr00t_zmq.py \
        --ckpt_path /path/to/steps_N_pytorch_model.pt \
        --port 5555 --use_bf16
"""

import argparse
import logging

from deployment.model_server.gr00t_obs_adapter import Gr00tCompatPolicy
from deployment.model_server.policy_wrapper import PolicyServerWrapper
from deployment.model_server.tools.zmq_policy_server import ZmqGr00tPolicyServer


def main(args) -> None:
    wrapper = PolicyServerWrapper(
        ckpt_path=args.ckpt_path,
        device="cuda",
        use_bf16=args.use_bf16,
        unnorm_key=args.unnorm_key,
    )
    focus = None
    if args.focus_view != "off":
        from deployment.model_server.focus_view import FocusViewSynth
        focus = FocusViewSynth(sam=args.focus_view, device="cuda",
                               camera=args.focus_camera or None, rate_hz=args.focus_rate_hz,
                               tap_port=args.focus_tap_port)
    policy = Gr00tCompatPolicy(
        wrapper,
        unnorm_key=args.unnorm_key,
        send_state=not args.no_state,
        fallback_instruction=args.fallback_instruction,
        focus_synth=focus,
        focus_source=args.focus_source,
        predict_kwargs=({"num_steps": args.num_inference_steps}
                        if args.num_inference_steps else None),
    )

    contract = policy.get_modality_config()
    logging.warning(
        "[TRAIN/TEST CONSISTENCY CHECK] serving ckpt=%s over the GR00T ZMQ protocol. "
        "Clients must send camera views %s (IN THIS ORDER — view identity is "
        "positional, so a reordered or partial set is misread silently) and state "
        "keys %s (flattened in this order, dims %s), and will receive action keys %s "
        "(dims %s), chunk_size=%s. Cross-check against the client's bridge profile.",
        args.ckpt_path,
        contract["video_keys"] or "<unspecified>",
        contract["state_keys"],
        contract["state_key_dims"],
        contract["action_keys"],
        contract["action_key_dims"],
        wrapper.metadata.get("action_chunk_size"),
    )

    server = ZmqGr00tPolicyServer(policy, host=args.host, port=args.port)
    try:
        server.run()
    except KeyboardInterrupt:
        logging.info("Shutting down server...")
        server.stop()


def build_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument(
        "--unnorm_key", type=str, default=None,
        help="Dataset statistics key; required for multi-dataset checkpoints.",
    )
    parser.add_argument(
        "--no_state", action="store_true",
        help="Do not forward proprioceptive state (for state-less checkpoints).",
    )
    parser.add_argument(
        "--fallback_instruction", type=str, default="",
        help="Language instruction used when the observation carries none.",
    )
    parser.add_argument(
        "--focus_view", default="off", choices=["off", "tiny", "small", "base_plus", "large"],
        help="Make the checkpoint's `focus` view on the SERVER (focus_view.py): a causal "
             "OWLv2 + SAM 2.1-<size> tracker cuts the phase target's 300 px window from the "
             "full-size ego frame, so the client sends only the ego camera (five-step v6).",
    )
    parser.add_argument(
        "--focus_camera", default="",
        help="host:port of the EGO camera's binary ZMQ stream (the robot: 192.168.123.164:5555). "
             "Set: a background thread tracks on it at --focus_rate_hz and yields to inference. "
             "Empty: the tracker steps on the frames the requests carry (openloop).",
    )
    parser.add_argument("--focus_rate_hz", type=float, default=10.0)
    parser.add_argument(
        "--focus_tap_port", type=int, default=0,
        help="PUB port for the operator feed: every request's served focus crop + the ego frame "
             "with its window, as keyed JPEG tiles (the bridge frame tap's VIEWJPG1 wire; the "
             "console shows them as policy-eye tiles). 0 = off.",
    )
    parser.add_argument(
        "--focus_source", default="rgb",
        help="The view the focus window is cut from (default rgb = the ego camera).",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=None,
        help="Override the flow-matching denoise steps per call (PI0 / PI05; default: the "
             "checkpoint's, 10). 5 halves the action-expert time; measured on the five-step "
             "v5 PI0.5: 140 -> 108 ms on an RTX 4080, normalised action change 0.00006 vs 10.",
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(build_argparser().parse_args())
