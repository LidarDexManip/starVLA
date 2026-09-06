# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# EgoVLA framework — adds EgoVLA (VILA-based bimanual VLA) as a selectable
# starVLA backbone, alongside QwenOFT / QwenGR00T / PI0 / PI05, in 2025.
"""EgoVLA framework (``framework.name: EgoVLA``).

EgoVLA is a VILA-based bimanual VLA: SigLIP-384 vision tower + Qwen2-1.5B LLM +
an ``mlp_downsample`` projector + a transformer trajectory decoder that predicts
a 48-dim per-step MANO / camera-frame action (per hand: 3 wrist-translation +
6 rot6d + 15 MANO pose) over a chunk of future steps.

Native to starVLA — no dependency on the EgoVLA_Release / ``llava`` package:
  * vision tower  : ``SiglipVisionModel``           (transformers, native)
  * LLM           : ``Qwen2ForCausalLM``            (transformers, native)
  * projector     : ``EgoVLAMMProjector``           (vendored, modules/vlm/vila_egovla)
  * action head   : ``EgoVLATrajDecoder``           (vendored, action_model/EgoVLA_ActionHeader)

The whole pipeline (encode_images -> multimodal splice -> LLM -> action-query
latent -> traj decoder) was verified end-to-end with the public
``ego_vla_checkpoint`` weights.

Contract (starVLA framework):
  forward(examples)        -> {"action_loss": scalar}          # L1 on 48-dim action
  predict_action(examples) -> {"normalized_actions": (B,T,48)} # inference

``examples`` items: ``image`` (List[PIL]), ``lang`` (str), ``action`` ([T,48]),
and camera-frame proprio (``state`` / ``proprio*``). Action-query positions are
marked by placeholder token ids in ``[input_placeholder_end_token_idx,
input_placeholder_start_token_idx]`` (EgoVLA: 151195..151375), matched by the
decoder's ``output_mask`` — identical to EgoVLA's own forward.
"""

import os
import random
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.framework.share_tools import merge_framework_config, state2str_transform
from starVLA.model.modules.action_model.EgoVLA_ActionHeader import get_action_model
from starVLA.model.modules.vlm.VILA import _EgoVLA_VILA_Interface
from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

logger = initialize_overwatch(__name__)

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100


@dataclass
class EgoVLADefaultConfig:
    """EgoVLA framework defaults (YAML ``framework:`` overrides these)."""

    name: str = "EgoVLA"

    # VILA backbone: base_vlm points at a VILA-style checkpoint dir with
    # {llm, vision_tower, mm_projector} sub-folders.
    qwenvl: dict = field(
        default_factory=lambda: {
            "base_vlm": "./playground/Pretrained_models/ego_vla_checkpoint/ckpt-6720",
            "attn_implementation": "sdpa",
            "use_bf16": True,
        }
    )

    # Trajectory-decoder action head (see EgoVLA_ActionHeader.get_action_model).
    action_model: dict = field(
        default_factory=lambda: {
            "action_model_type": "EgoVLATrajDecoder",
            "action_dim": 48,           # 2 hands x (3 trans + 6 rot6d + 15 MANO)
            "action_hidden_dim": 1536,  # Qwen2-1.5B hidden (aligned at runtime)
            "action_horizon": 30,       # future steps per chunk
            "proprio_size": 16,
            "use_proprio": True,
            "sep_proprio": True,
            "traj_decoder_type": "transformer_split_action_v2",
        }
    )

    # Action-query placeholder token id range (EgoVLA action_tokenizer).
    input_placeholder_end_token_idx: int = 151195
    input_placeholder_start_token_idx: int = 151375
    # Prompt scaffold; the action-query block is appended as `action_query_id` * (2*horizon).
    action_query_token_id: int = 151300


@FRAMEWORK_REGISTRY.register("EgoVLA")
class EgoVLA(baseframework):
    """EgoVLA VILA backbone + trajectory-decoder head as a starVLA framework."""

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = merge_framework_config(EgoVLADefaultConfig, config)

        # VLM backbone (native SigLIP + Qwen2 + vendored projector).
        self.vlm = _EgoVLA_VILA_Interface(self.config)
        # Align the decoder hidden dim to the actual LLM hidden size.
        self.config.framework.action_model.action_hidden_dim = self.vlm.hidden_size

        # Action head. Two modes:
        #  - "EgoVLATrajDecoder" (default): EgoVLA's native 48-dim MANO/camera head
        #    (checkpoint-compatible; needs 2*horizon interleaved action queries).
        #  - "MLP"/"joint": a plain L1 regression head that predicts `action_dim`
        #    joint targets directly (Option A: finetune the VILA backbone on G1 joint
        #    data; one action query per future step, no MANO/proprio decode).
        self.action_horizon = int(self.config.framework.action_model.action_horizon)
        self._head_type = str(getattr(self.config.framework.action_model, "action_model_type", "EgoVLATrajDecoder"))
        self._is_joint_head = self._head_type in ("MLP", "joint", "L1", "L1Regression")
        self._is_fm_head = self._head_type in ("DiT-B", "DiT-L", "DiT-XL", "GR00T", "FlowMatching", "flow_matching")
        if self._is_joint_head:
            from starVLA.model.modules.action_model.MLP_ActionHeader import L1RegressionActionHead
            self.action_model = L1RegressionActionHead(
                input_dim=self.vlm.hidden_size,
                hidden_dim=int(getattr(self.config.framework.action_model, "action_hidden_dim", self.vlm.hidden_size)),
                action_dim=int(self.config.framework.action_model.action_dim),
                NUM_ACTIONS_CHUNK=self.action_horizon,
            ).to(self.vlm.model_dtype)
            self._n_query = self.action_horizon
        elif self._is_fm_head:
            # GR00T-style flow-matching DiT head. It cross-attends to the full VILA
            # hidden sequence (like VLM4A/QwenGR00T, but with VILA as the backbone),
            # so it needs only `hidden` + attention mask + (state, action) — no MANO
            # proprio and no action-query latent gather. cross_attention_dim is aligned
            # to the VILA hidden size (1536) at runtime, exactly like QwenGR00T. The head
            # is left in fp32 (its forward/predict_action autocast to fp32 internally).
            from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
            self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = self.vlm.hidden_size
            self.action_model = FlowmatchingActionHead(self.config)
            self._n_query = self.action_horizon  # query block still injected but unused by this head
        else:
            # Match the backbone dtype so bf16 hidden states / proprio flow through it.
            self.action_model = get_action_model(config=self.config).to(self.vlm.model_dtype)
            self._n_query = 2 * self.action_horizon
        fw = self.config.framework
        self._q_lo = int(getattr(fw, "input_placeholder_end_token_idx", 151195))
        self._q_hi = int(getattr(fw, "input_placeholder_start_token_idx", 151375))
        self._q_id = int(getattr(fw, "action_query_token_id", 151300))
        self.tokenizer = self.vlm.tokenizer
        self.l1_loss = nn.L1Loss()

    # ------------------------------------------------------------------
    # Input construction (StarVLA examples -> EgoVLA model inputs)
    # ------------------------------------------------------------------
    def _build_inputs(self, examples):
        """Tokenize instruction + <image> + action-query block; assemble proprio.

        Produces (input_ids, attention_mask, labels, images, proprio_dict). The
        action-query block is ``2*action_horizon`` placeholder tokens (left/right
        interleaved) whose label ids sit in the query range so the decoder's
        ``output_mask`` selects exactly those hidden states.
        """
        device = next(self.parameters()).device
        n_query = self._n_query
        ids_list, labels_list, imgs_list = [], [], []

        state_dropout_prob = float(self.config.framework.get("state_dropout_prob", 0.8))
        for ex in examples:
            instr = ex["lang"]
            # Fold discretised proprioceptive state into the instruction (π0.5-style,
            # share_tools.state2str_transform), matching absv2 training. The dropout is
            # training-only, so at inference state is always injected when present.
            # The flow-matching head instead consumes the continuous state through its
            # own state_encoder (GR00T-style), so we do NOT inject a [STATE] token there.
            state = ex.get("state")
            if state is not None and not self._is_fm_head and not (self.training and random.random() < state_dropout_prob):
                state_str = state2str_transform(np.asarray(state)[0], num_bins=256)
                instr = f"{instr} [STATE] {state_str} [ACTION]"
            if os.environ.get("EGOVLA_LOG_PROMPT") and not getattr(self, "_prompt_logged", False):
                logger.warning("[deploy-prompt] %s", instr[:400])
                self._prompt_logged = True
            tok = self.tokenizer(instr, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
            # One <image> placeholder PER input view. VILA's
            # prepare_inputs_labels_for_multimodal (vila_egovla/arch.py) counts the
            # IMAGE_TOKEN slots and stuffs image_features[i] into the i-th slot in
            # order, so the number of placeholders MUST equal the number of images
            # this example carries: 1 for single-ego, 3 for the multi-view checkpoint
            # (ego_view, left_wrist_view, right_wrist_view — the modality.json order,
            # which the obs adapter preserves via video.values()). A single hardcoded
            # token silently drops every view past the first.
            _img = ex["image"]
            n_img_ex = len(_img) if isinstance(_img, (list, tuple)) else 1
            # [text tokens] <image>*n_img [text?] then N_QUERY action-query tokens.
            seq = torch.cat([
                tok,
                torch.full((n_img_ex,), IMAGE_TOKEN_INDEX, dtype=torch.long),
                torch.full((n_query,), self._q_id, dtype=torch.long),
            ])
            lab = torch.full_like(seq, IGNORE_INDEX)
            lab[-n_query:] = self._q_id  # query positions carry their id
            ids_list.append(seq)
            labels_list.append(lab)
            imgs_list.append(ex["image"])

        max_len = max(len(s) for s in ids_list)
        B = len(ids_list)
        input_ids = torch.full((B, max_len), self.tokenizer.pad_token_id or 0, dtype=torch.long)
        labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
        attn = torch.zeros((B, max_len), dtype=torch.bool)
        for i, (s, l) in enumerate(zip(ids_list, labels_list)):
            input_ids[i, : len(s)] = s
            labels[i, : len(l)] = l
            attn[i, : len(s)] = True

        # Images -> (B, n_img, 3, 384, 384) via SigLIP processor. Deployment sends
        # numpy frames (from the bridge obs); to_pil_preserve normalizes them to PIL.
        flat = [im for sub in imgs_list for im in (sub if isinstance(sub, (list, tuple)) else [sub])]
        flat = [to_pil_preserve(im) for im in flat]
        px = self.vlm.image_processor(images=flat, return_tensors="pt")["pixel_values"]
        n_img = len(flat) // B
        images = px.view(B, n_img, *px.shape[1:])

        proprio = self._build_proprio(examples, device)
        return (input_ids.to(device), attn.to(device), labels.to(device),
                images.to(device, next(self.vlm.parameters()).dtype), proprio)

    def _build_proprio(self, examples, device):
        """Assemble the sep-proprio dict (camera-frame) from the example.

        Reads explicit ``proprio_3d`` / ``proprio_rot`` / ``proprio_hand_finger_tip``
        if the DataConfig provides them; otherwise zeros (model still runs).
        """
        dt = next(self.vlm.parameters()).dtype
        B = len(examples)

        def stack(key, dim):
            if key in examples[0] and examples[0][key] is not None:
                return torch.tensor(np.array([np.asarray(e[key]).reshape(-1) for e in examples]),
                                    device=device, dtype=dt)
            return torch.zeros((B, dim), device=device, dtype=dt)

        return {
            "proprio": stack("proprio", 16),
            "proprio_3d": stack("proprio_3d", 6),
            "proprio_rot": stack("proprio_rot", 6),
            "proprio_hand_finger_tip": stack("proprio_hand_finger_tip", 30),
        }

    # ------------------------------------------------------------------
    def _backbone(self, examples):
        """Run VILA and return (hidden (B,L,H), mask (B,L) bool) for a cross-attn head.

        Used by the flow-matching head, which cross-attends to the whole hidden
        sequence (`vl_embs`) and does not need the action-query latent gather.
        """
        input_ids, attn, labels, images, proprio = self._build_inputs(examples)
        out = self.vlm(input_ids, images, attention_mask=attn, labels=labels)
        return out["hidden_states"], out["attention_mask"].to(torch.bool)

    def _stack_state(self, examples, device, dtype):
        """Stack per-example proprio state into (B, 1, state_dim) for the FM head."""
        if "state" not in examples[0] or examples[0]["state"] is None:
            return None
        return torch.tensor(np.array([np.asarray(e["state"]) for e in examples]),
                            device=device, dtype=dtype)

    def _run(self, examples):
        """Shared pipeline: backbone -> action-query latent -> traj decoder pred."""
        input_ids, attn, labels, images, proprio = self._build_inputs(examples)
        out = self.vlm(input_ids, images, attention_mask=attn, labels=labels)
        hidden, attn2, labels2 = out["hidden_states"], out["attention_mask"], out["labels"]

        output_mask = (labels2 >= self._q_lo) & (labels2 <= self._q_hi)
        latent = hidden[output_mask]  # (B * n_query, H)
        if self._is_joint_head:
            B = len(examples)
            latent = latent.reshape(B, self._n_query, latent.shape[-1])
            pred = self.action_model.predict_action(latent)  # (B, horizon, action_dim)
            return pred.reshape(B * self.action_horizon, -1)
        memory_mask = torch.where(output_mask, torch.zeros_like(attn2), attn2)
        result = self.action_model(latent, proprio, memory=hidden, memory_mask=memory_mask)
        return result["pred"]  # (B*T, 48)

    def forward(self, examples: List[dict] = None, **kwargs):
        if self._is_fm_head:
            # GR00T-style flow-matching loss: cross-attend the DiT to the full VILA
            # hidden sequence and regress the velocity field over the action chunk.
            hidden, mask = self._backbone(examples)
            dev, dt = hidden.device, hidden.dtype
            actions = torch.tensor(np.array([np.asarray(e["action"]) for e in examples]),
                                   device=dev, dtype=dt)
            actions = actions[:, -self.action_horizon :, :]  # (B, horizon, action_dim)
            state = self._stack_state(examples, dev, dt)     # (B, 1, state_dim) or None
            rep = int(self.config.framework.action_model.get("repeated_diffusion_steps", 4))
            h = hidden.repeat(rep, 1, 1)
            m = mask.repeat(rep, 1)
            a = actions.repeat(rep, 1, 1)
            s = state.repeat(rep, 1, 1) if state is not None else None
            # State-condition dropout (π0.5-style regularisation against proprioceptive
            # copycat / causal confusion): with prob `state_dropout_prob`, zero the state
            # input per (repeated) example during training so the policy must rely on
            # vision. Inference always keeps state. We zero the row (not pass None) because
            # the head's state_encoder is batched — a zeroed row is its "null" state token.
            if s is not None and self.training:
                p = float(self.config.framework.get("state_dropout_prob", 0.8))
                if p > 0:
                    keep = (torch.rand(s.shape[0], 1, 1, device=s.device) >= p).to(s.dtype)
                    # `state_dropout_keep_dims` trailing dims are NEVER dropped: they carry the
                    # phase one-hot (a hard task-stage signal the policy must always see), while
                    # the proprioceptive dims in front are still dropped to prevent copycat.
                    kd = int(self.config.framework.get("state_dropout_keep_dims", 0))
                    if kd > 0:
                        s = torch.cat([s[..., :-kd] * keep, s[..., -kd:]], dim=-1)
                    else:
                        s = s * keep
            with torch.autocast("cuda", dtype=torch.float32):
                loss = self.action_model(h, a, s, encoder_attention_mask=m)
            return {"action_loss": loss}
        pred = self._run(examples)  # (B*T, 48)
        B = len(examples)
        pred = pred.reshape(B, self.action_horizon, -1)
        actions = torch.tensor(np.array([np.asarray(e["action"]) for e in examples]),
                               device=pred.device, dtype=pred.dtype)
        target = actions[:, -self.action_horizon :, :]
        return {"action_loss": self.l1_loss(pred, target)}

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs):
        if not isinstance(examples, list):
            examples = [examples]
        # Deployment (bridge) sends numpy frames -> normalize to PIL before resize.
        for e in examples:
            if isinstance(e.get("image"), list):
                e["image"] = [to_pil_preserve(im) for im in e["image"]]
        train_size = getattr(self.config.datasets.vla_data, "obs_image_size", None)
        # The TRAINING dataloader hardcodes a 224x224 resize
        # (gr00t_lerobot/datasets.py `_pack_sample`) and ignores obs_image_size
        # entirely, so every image the model ever saw was band-limited to 224
        # before SigLIP upsampled it to 384. This path is the only one that
        # resizes at all, so serving at obs_image_size (384) hands the model
        # sharper frames than it has ever seen. EGOVLA_OBS_SIZE=224,224
        # reproduces the training bottleneck for an A/B.
        _ov = os.environ.get("EGOVLA_OBS_SIZE")
        if _ov:
            train_size = [int(x) for x in _ov.split(",")]
        if train_size:
            for e in examples:
                if isinstance(e.get("image"), list):
                    e["image"] = resize_images(e["image"], target_size=train_size)
        if self._is_fm_head:
            hidden, mask = self._backbone(examples)
            state = self._stack_state(examples, hidden.device, hidden.dtype)
            with torch.autocast("cuda", dtype=torch.float32):
                pred = self.action_model.predict_action(hidden, state, encoder_attention_mask=mask)
            return {"normalized_actions": pred.detach().float().cpu().numpy()}
        pred = self._run(examples)
        B = len(examples)
        pred = pred.reshape(B, self.action_horizon, -1)
        return {"normalized_actions": pred.detach().float().cpu().numpy()}
