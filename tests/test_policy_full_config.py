"""Serving must use the same explicit config for the model and its normalizer."""
from unittest.mock import MagicMock, patch

from deployment.model_server import policy_wrapper
from deployment.model_server.server_policy_gr00t_zmq import build_argparser


def test_full_config_reaches_model_metadata_and_normalizer():
    cfg = {
        "framework": {"action_model": {"action_horizon": 30}},
        "datasets": {"vla_data": {"include_state": True}},
    }
    model = MagicMock()
    model.to.return_value = model
    model.eval.return_value = model
    processor = MagicMock(unnorm_key="new_embodiment")
    with (
        patch.object(policy_wrapper.baseframework, "from_pretrained", return_value=model) as load,
        patch.object(policy_wrapper, "read_mode_config", return_value=(cfg, {"new_embodiment": {}})) as read,
        patch.object(policy_wrapper, "PolicyNormProcessor", return_value=processor) as normalize,
    ):
        wrapper = policy_wrapper.PolicyServerWrapper(
            "checkpoints/model.pt", device="cpu", config_path="config.full.yaml")
        load.assert_called_once_with("checkpoints/model.pt", config_path="config.full.yaml")
        assert read.call_count == 2
        for call in read.call_args_list:
            assert call.kwargs["config_path"] == "config.full.yaml"
        normalize.assert_called_once_with(
            "checkpoints/model.pt", unnorm_key=None, config_path="config.full.yaml")
        assert wrapper.metadata["action_chunk_size"] == 30
    args = build_argparser().parse_args([
        "--ckpt_path", "checkpoints/model.pt", "--config_path", "config.full.yaml"])
    assert args.config_path == "config.full.yaml"


def test_existing_cli_keeps_default_config_selection():
    args = build_argparser().parse_args(["--ckpt_path", "checkpoints/model.pt"])
    assert args.config_path is None
