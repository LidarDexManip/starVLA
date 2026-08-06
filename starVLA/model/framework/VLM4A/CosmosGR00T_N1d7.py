# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
CosmosGR00T-N1.7 Framework — a starVLA-native reproduction of NVIDIA GR00T-N1.7
that DIRECTLY LOADS the public ``nvidia/GR00T-N1.7-3B`` pretrained parameters.

Architecture (matches the checkpoint 1:1):
  - Backbone : Cosmos-Reason2-2B (Qwen3-VL), LLM truncated to ``select_layer`` (16)
  - Head     : "alternate VL" flow-matching DiT (32 self-attn+adaLN blocks, even
               blocks cross-attend to VL, alternating text/image tokens) +
               a 4-layer VL self-attention adapter + VL LayerNorm +
               multi-embodiment (category-specific) state/action projectors.

Unlike the older ``CosmosGR00T`` (single-embodiment cross-attention DiT, no
pretrained weights), this framework's module names/shapes are chosen so the
checkpoint's ``backbone.*`` and ``action_head.*`` tensors load without surgery.
Set ``framework.pretrained_gr00t_n1d7`` to the N1.7 snapshot dir (or leave the
default HF cache path) to warm-start; training then fine-tunes on starVLA data.

The data contract is starVLA's usual one: ``examples`` = list of
``{image, lang, action, state}`` (actions/states already normalised by the
dataloader). We pad action/state to the model's ``max_*_dim`` (132), tag every
sample with a fixed ``embodiment_id`` (default 25 = the pretrained
``unitree_g1_full_body_with_waist_height_nav_cmd`` slot, whose action layout
matches the G1 piston robot), and mask the loss to the real action dims.
"""

import glob
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_workspace_root = Path(__file__).parent.parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config
from starVLA.model.modules.action_model.GR00T_N1d7_ActionHeader import get_action_model_n1d7
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

# Default N1.7 snapshot in the HF cache (resolved lazily; only used if the path exists).
_DEFAULT_N1D7 = str(
    Path.home()
    / ".cache/huggingface/hub/models--nvidia--GR00T-N1.7-3B/snapshots"
)


@dataclass
class CosmosGR00TN1d7DefaultConfig:
    """Defaults mirror ``nvidia/GR00T-N1.7-3B``'s config.json. YAML overrides win."""

    name: str = "CosmosGR00TN1d7"

    # Path to the N1.7 checkpoint dir (contains model-*.safetensors). If None or
    # missing, the model is built with a base-Cosmos backbone + randomly-init head
    # (still trainable, just no GR00T pretraining).
    pretrained_gr00t_n1d7: Optional[str] = None

    # Fixed embodiment slot used for every sample (see module docstring).
    embodiment_id: int = 25

    # Backbone fine-tuning knobs. If BOTH tune_llm and tune_visual are False the
    # backbone is frozen AND its features are detached (no backbone activation
    # graph is stored) — the "freeze VLM, train action head only" recipe, which
    # is the memory-light default for small-data fine-tunes.
    tune_llm: bool = True
    tune_visual: bool = True
    enable_gradient_checkpointing: bool = True

    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "nvidia/Cosmos-Reason2-2B",
            "attn_implementation": "sdpa",
            "select_layer": 16,
        }
    )

    action_model: dict = field(
        default_factory=lambda: {
            # real robot dims (unpadded); the head runs at max_*_dim (132)
            "action_dim": 30,
            "state_dim": 29,
            "action_horizon": 30,
            # padded head dims (must stay 132 to match the pretrained projectors)
            "max_action_dim": 132,
            "max_state_dim": 132,
            "state_history_length": 1,
            "max_num_embodiments": 32,
            "backbone_embedding_dim": 2048,
            "input_embedding_dim": 1536,
            "hidden_size": 1024,
            "add_pos_embed": True,
            "max_seq_len": 1024,
            "use_vlln": True,
            "use_alternate_vl_dit": True,
            "attend_text_every_n_blocks": 2,
            "num_inference_timesteps": 4,
            "num_timestep_buckets": 1000,
            "noise_beta_alpha": 1.5,
            "noise_beta_beta": 1.0,
            "noise_s": 0.999,
            "state_dropout_prob": 0.2,
            # Press-phase frame weighting (see `_frame_weights`). 1.0 == disabled.
            "grasp_loss_weight": 1.0,
            "grasp_weight_dim": 25,
            "grasp_weight_thresh": -0.143,
            "diffusion_model_cfg": {
                "num_attention_heads": 32,
                "attention_head_dim": 48,
                "output_dim": 1024,
                "num_layers": 32,
                "dropout": 0.2,
                "final_dropout": True,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
            },
            "vl_self_attention_cfg": {
                "num_attention_heads": 32,
                "attention_head_dim": 64,
                "num_layers": 4,
                "dropout": 0.2,
                "final_dropout": True,
                "positional_embeddings": None,
            },
        }
    )


@FRAMEWORK_REGISTRY.register("CosmosGR00TN1d7")
class CosmosGR00T_N1d7(baseframework):
    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(CosmosGR00TN1d7DefaultConfig, config)
        am = self.config.framework.action_model

        # --- backbone (Cosmos-Reason2 / Qwen3-VL) ---
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.select_layer = int(self.config.framework.qwenvl.get("select_layer", 16))
        self._truncate_backbone_layers(self.select_layer)
        self._image_token_id = self._resolve_image_token_id()

        # --- N1.7 flow-matching head ---
        self.action_head = get_action_model_n1d7(config=self.config)

        # real (unpadded) robot dims + head dims
        self.action_dim = int(am.action_dim)
        self.state_dim = int(am.state_dim)
        self.action_horizon = int(am.action_horizon)
        self.max_action_dim = int(am.max_action_dim)
        self.max_state_dim = int(am.max_state_dim)
        self.state_history_length = int(am.get("state_history_length", 1))
        self.embodiment_id_value = int(self.config.framework.get("embodiment_id", 25))

        # press-phase frame weighting (training only; see `_frame_weights`)
        self.grasp_loss_weight = float(am.get("grasp_loss_weight", 1.0))
        self.grasp_weight_dim = int(am.get("grasp_weight_dim", 25))
        self.grasp_weight_thresh = float(am.get("grasp_weight_thresh", -0.143))
        self._grasp_weight_logged = False

        # --- load N1.7 pretrained weights (backbone + head) ---
        ckpt = self._resolve_n1d7_path(self.config.framework.get("pretrained_gr00t_n1d7", None))
        if ckpt is not None:
            self._load_n1d7_pretrained(ckpt)
        else:
            logger.warning(
                "No GR00T-N1.7 checkpoint found; built with base-Cosmos backbone "
                "and randomly-initialised head (set framework.pretrained_gr00t_n1d7)."
            )

        # --- backbone fine-tuning knobs (applied AFTER weight load) ---
        self.tune_llm = bool(self.config.framework.get("tune_llm", True))
        self.tune_visual = bool(self.config.framework.get("tune_visual", True))
        self._freeze_backbone = not (self.tune_llm or self.tune_visual)
        self._apply_backbone_trainability()
        if bool(self.config.framework.get("enable_gradient_checkpointing", True)):
            self._enable_gradient_checkpointing()

    # ------------------------------------------------------------------ setup
    def _backbone_llm_layers(self):
        # Qwen3VLForConditionalGeneration -> .model (Qwen3VLModel) -> .language_model -> .layers
        return self.qwen_vl_interface.model.model.language_model.layers

    def _truncate_backbone_layers(self, select_layer: int) -> None:
        layers = self._backbone_llm_layers()
        removed = 0
        while len(layers) > select_layer:
            layers.pop(-1)
            removed += 1
        # keep config in sync so HF forward doesn't index past the kept layers
        try:
            self.qwen_vl_interface.model.config.text_config.num_hidden_layers = len(layers)
        except Exception:
            pass
        logger.info(f"Backbone LLM truncated to {len(layers)} layers (select_layer={select_layer}, removed {removed}).")

    def _apply_backbone_trainability(self) -> None:
        vlm = self.qwen_vl_interface.model
        try:
            llm = vlm.model.language_model
            visual = vlm.model.visual
        except AttributeError:
            logger.warning("Could not resolve backbone submodules for trainability; leaving as-is.")
            return
        llm.requires_grad_(self.tune_llm)
        visual.requires_grad_(self.tune_visual)
        if self._freeze_backbone:
            logger.info("Backbone FROZEN (tune_llm=tune_visual=False): VL features detached — training action head only.")
        else:
            logger.info(f"Backbone trainable: tune_llm={self.tune_llm} tune_visual={self.tune_visual}")

    def _enable_gradient_checkpointing(self) -> None:
        # Backbone (HF PreTrainedModel) — only meaningful if it is being tuned.
        if not self._freeze_backbone:
            try:
                self.qwen_vl_interface.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                logger.info("Enabled gradient checkpointing on the backbone.")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Backbone gradient checkpointing not enabled: {e}")

    def _resolve_image_token_id(self) -> int:
        cfg = self.qwen_vl_interface.model.config
        for obj in (cfg, getattr(cfg, "text_config", None)):
            tid = getattr(obj, "image_token_id", None)
            if tid is not None:
                return int(tid)
        return 151655  # Cosmos-Reason2 default

    @staticmethod
    def _resolve_n1d7_path(path: Optional[str]) -> Optional[str]:
        candidates = []
        if path:
            candidates.append(path)
        candidates.append(_DEFAULT_N1D7)
        for c in candidates:
            p = Path(c)
            if p.is_dir():
                # if a snapshots parent was given, descend into the single snapshot
                if p.name == "snapshots" or (p / "snapshots").is_dir():
                    base = p if p.name == "snapshots" else p / "snapshots"
                    snaps = [d for d in base.iterdir() if d.is_dir()]
                    if snaps:
                        p = snaps[0]
                if list(p.glob("*.safetensors")):
                    return str(p)
        return None

    # -------------------------------------------------------------- weight load
    def _load_n1d7_pretrained(self, ckpt_dir: str) -> None:
        from safetensors.torch import load_file

        state = {}
        for shard in sorted(glob.glob(str(Path(ckpt_dir) / "*.safetensors"))):
            state.update(load_file(shard))

        remapped = {}
        for k, v in state.items():
            if k.startswith("backbone.model."):
                remapped["qwen_vl_interface.model." + k[len("backbone.model."):]] = v
            elif k.startswith("action_head."):
                remapped[k] = v  # self.action_head.* aligns directly
            # (anything else is ignored on purpose)

        model_keys = set(self.state_dict().keys())
        ckpt_keys = set(remapped.keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        incompatible = self.load_state_dict(remapped, strict=False)

        # summarise (only warn on genuinely concerning gaps)
        logger.info(
            f"Loaded GR00T-N1.7 weights from {ckpt_dir}: "
            f"{len(ckpt_keys & model_keys)}/{len(model_keys)} tensors matched."
        )
        # backbone truncation intentionally drops LLM layers >= select_layer; those
        # never appear in `missing`. Any other missing/unexpected keys are real.
        real_missing = [k for k in missing if not k.startswith("qwen_vl_interface.model.model.language_model.layers.")]
        if real_missing:
            logger.warning(f"[N1.7 load] {len(real_missing)} missing keys, e.g. {real_missing[:6]}")
        if unexpected:
            logger.warning(f"[N1.7 load] {len(unexpected)} unexpected keys, e.g. {unexpected[:6]}")

    # ---------------------------------------------------------------- backbone
    def _encode_vl(self, images, instructions):
        """Return (vl_embeds[B,S,2048] bf16, image_mask[B,S] bool, attn_mask[B,S] bool)."""
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=images, instructions=instructions)
        keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: qwen_inputs[k] for k in keys if k in qwen_inputs}
        fwd_ctx = torch.no_grad() if getattr(self, "_freeze_backbone", False) else torch.enable_grad()
        with fwd_ctx, torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = self.qwen_vl_interface.model(
                **vl_input, output_hidden_states=True, use_cache=False, return_dict=True
            )
        vl_embeds = outputs.hidden_states[-1]
        if getattr(self, "_freeze_backbone", False):
            vl_embeds = vl_embeds.detach()
        input_ids = vl_input["input_ids"]
        image_mask = input_ids == self._image_token_id
        attn_mask = vl_input["attention_mask"] == 1
        return vl_embeds, image_mask, attn_mask

    # ------------------------------------------------------------ data padding
    def _prep_actions(self, actions_list, device, dtype):
        """list[np [T, adim]] -> (actions[B,H,132], action_mask[B,H,132])."""
        H, D = self.action_horizon, self.max_action_dim
        B = len(actions_list)
        actions = torch.zeros((B, H, D), device=device, dtype=dtype)
        mask = torch.zeros((B, H, D), device=device, dtype=dtype)
        for i, a in enumerate(actions_list):
            a = torch.as_tensor(np.asarray(a), device=device, dtype=dtype)
            a = a[-H:]  # last H steps
            t, d = a.shape
            actions[i, H - t:, :d] = a
            mask[i, H - t:, :d] = 1.0
        if self.grasp_loss_weight != 1.0:
            mask = mask * self._frame_weights(actions, mask)
        return actions, mask

    def _frame_weights(self, actions, mask):
        """[B,H,132] -> [B,H,1] per-frame loss weight, folded into ``action_mask``.

        The plunger press is a SHORT phase (~21% of frames on the long-horizon
        piston set) so a uniform mean averages it away and the press is under-fit.
        A frame is flagged when the right-hand thumb channel (``grasp_weight_dim``,
        25 == ``action.right_hand[5]`` in the 30-dim
        ``left_arm7|right_arm7|left_hand6|right_hand6|base_height1|navigate3``
        layout) sits above ``grasp_weight_thresh``, i.e. lifted off its rest value.
        Flagged frames get ``grasp_loss_weight`` on ALL of their action dims.

        Folding the weight into the mask is exact rather than approximate: the head
        computes ``(per_elem * action_mask).sum() / action_mask.sum()``, so a scaled
        mask yields precisely the proposed weighted mean ``L = Σ wₜ·err / Σ wₜ`` —
        and, being a weighted MEAN, it leaves the loss on the same scale as the
        unweighted run, so LR / grad-clip carry over unchanged.
        """
        d = self.grasp_weight_dim
        valid = mask[:, :, d] > 0  # excludes the zero-pad rows of short chunks
        press = (actions[:, :, d] > self.grasp_weight_thresh) & valid
        if not self._grasp_weight_logged:
            self._grasp_weight_logged = True
            frac = (press.sum() / valid.sum().clamp(min=1)).item()
            logger.info(
                f"[press-frame weighting] dim={d} thresh={self.grasp_weight_thresh} "
                f"weight={self.grasp_loss_weight} -> {frac * 100:.1f}% of frames "
                f"flagged in the first batch."
            )
            if frac < 0.01 or frac > 0.99:
                logger.warning(
                    "[press-frame weighting] the flag fires on ~all or ~no frames, so "
                    "the weighted mean collapses back to the plain mean (a silent "
                    f"no-op). Check that action dim {d} actually VARIES in this "
                    "dataset: when q01 == q99 StateActionTransform passes the raw "
                    "value through un-normalised, so the threshold is compared "
                    "against a raw number."
                )
        w = 1.0 + (self.grasp_loss_weight - 1.0) * press.to(actions.dtype)
        return w.unsqueeze(-1)  # [B,H,1] -> broadcasts over all action dims

    def _prep_state(self, state_list, batch_size, device, dtype):
        """list[np [T,sdim] or [sdim]] (or None) -> state[B, state_history_length, 132].

        ``state_list`` is None/empty when the checkpoint is served with
        ``--no_state`` (or trained state-less); return zeros in that case.
        """
        T, D = self.state_history_length, self.max_state_dim
        state = torch.zeros((batch_size, T, D), device=device, dtype=dtype)
        if not state_list:
            return state
        for i, s in enumerate(state_list):
            s = torch.as_tensor(np.asarray(s), device=device, dtype=dtype)
            if s.dim() == 1:
                s = s[None, :]
            s = s[-T:]
            t, d = s.shape
            state[i, T - t:, :d] = s
        return state

    def _embodiment_ids(self, B, device):
        return torch.full((B,), self.embodiment_id_value, dtype=torch.long, device=device)

    # ------------------------------------------------------------------ train
    def forward(self, examples: List[dict] = None, **kwargs):
        images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None

        vl_embeds, image_mask, attn_mask = self._encode_vl(images, instructions)
        device = vl_embeds.device

        with torch.autocast("cuda", dtype=torch.float32):
            vl_embeds = vl_embeds.float()
            act, act_mask = self._prep_actions(actions, device, torch.float32)
            st = self._prep_state(state, len(examples), device, torch.float32)
            emb = self._embodiment_ids(len(examples), device)
            loss = self.action_head(
                vl_embeds=vl_embeds,
                actions=act,
                state=st,
                embodiment_id=emb,
                image_mask=image_mask,
                backbone_attention_mask=attn_mask,
                action_mask=act_mask,
            )
        return {"action_loss": loss}

    # ---------------------------------------------------------------- inference
    def _check_num_views(self, images) -> None:
        """Warn once if the eval client sends a different camera count than training.

        Nothing downstream identifies a view by name — the processor just emits one
        image-token block per PIL image in list order — so a client that sends the
        ego view alone to a 3-view checkpoint produces a valid forward pass on a
        third of the expected tokens, i.e. a SILENT out-of-distribution failure that
        looks like a merely bad policy. ``datasets.vla_data.expected_num_views``
        (set it to ``len(video_keys)`` of the training DataConfig) turns that into a
        loud message. Leave it unset to skip the check.
        """
        if getattr(self, "_num_views_checked", False):
            return
        self._num_views_checked = True
        expected = getattr(self.config.datasets.vla_data, "expected_num_views", None)
        if not expected:
            return
        got = len(images[0]) if images and isinstance(images[0], (list, tuple)) else 1
        if got != int(expected):
            logger.error(
                f"[camera contract] this checkpoint was trained on {expected} camera "
                f"view(s) but the eval client sent {got}. The forward pass will still "
                "succeed — with the wrong number of image tokens — so expect "
                "confidently wrong actions. Fix the client, not this check."
            )
        else:
            logger.info(f"[camera contract] {got} camera view(s), matches training.")

    @torch.inference_mode()
    def predict_action(self, examples: List[dict], **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        state = [ex["state"] for ex in examples] if "state" in examples[0] else None
        self._check_num_views(images)

        train_obs_image_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        if train_obs_image_size:
            images = resize_images(images, target_size=train_obs_image_size)

        vl_embeds, image_mask, attn_mask = self._encode_vl(images, instructions)
        device = vl_embeds.device

        with torch.autocast("cuda", dtype=torch.float32):
            vl_embeds = vl_embeds.float()
            st = self._prep_state(state, len(examples), device, torch.float32)
            emb = self._embodiment_ids(len(examples), device)
            pred = self.action_head.predict_action(
                vl_embeds=vl_embeds,
                state=st,
                embodiment_id=emb,
                image_mask=image_mask,
                backbone_attention_mask=attn_mask,
            )
        pred = pred[:, :, : self.action_dim]  # strip 132-pad back to real dims
        return {"normalized_actions": pred.detach().cpu().float().numpy()}
