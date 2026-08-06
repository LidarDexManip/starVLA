# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
#
# Action head for the GR00T-N1.7 framework, ported from Isaac-GR00T (N1.7):
#   gr00t/model/gr00t_n1d7/gr00t_n1d7.py :: Gr00tN1d7ActionHead
#
# The submodule NAMES and SHAPES here are chosen to match the public
# `nvidia/GR00T-N1.7-3B` checkpoint so its ``action_head.*`` tensors load 1:1:
#   action_head.model.*            -> self.model            (AlternateVLDiT, 32 layers)
#   action_head.vl_self_attention.*-> self.vl_self_attention(SelfAttentionTransformer, 4 layers)
#   action_head.vlln.*             -> self.vlln             (LayerNorm over VL features)
#   action_head.state_encoder.*    -> self.state_encoder    (CategorySpecificMLP)
#   action_head.action_encoder.*   -> self.action_encoder   (MultiEmbodimentActionEncoder)
#   action_head.action_decoder.*   -> self.action_decoder   (CategorySpecificMLP)
#   action_head.position_embedding -> self.position_embedding
#
# Unlike starVLA's other GR00T head (GR00T_ActionHeader.py, a single-embodiment
# cross-attention DiT), N1.7 uses an "alternate VL DiT": all DiT blocks are
# self-attention with adaLN; even blocks additionally cross-attend to the VL
# features, alternating between text-only and image-only tokens every
# ``attend_text_every_n_blocks``. There are NO future/query tokens.
from typing import Optional

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.distributions import Beta

from starVLA.model.modules.action_model.flow_matching_head.gr00t_n1d7_dit import (
    AlternateVLDiT,
    DiT,
    SelfAttentionTransformer,
)
from starVLA.model.modules.action_model.flow_matching_head.gr00t_n1d7_embodiment_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


def _to_plain(cfg) -> dict:
    """OmegaConf/DictConfig/dict -> plain python dict (diffusers register_to_config
    rejects OmegaConf nodes)."""
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        return dict(cfg)
    return OmegaConf.to_container(cfg, resolve=True)


class GR00TN1d7ActionHead(nn.Module):
    """Flow-matching action head matching GR00T-N1.7.

    Reads its hyper-parameters from ``full_config.framework.action_model``.
    """

    def __init__(self, full_config):
        super().__init__()
        cfg = full_config.framework.action_model
        self.full_config = full_config
        self.config = cfg

        self.hidden_size = int(cfg.hidden_size)
        self.input_embedding_dim = int(cfg.input_embedding_dim)
        self.backbone_embedding_dim = int(cfg.backbone_embedding_dim)
        self.max_action_dim = int(cfg.max_action_dim)
        self.max_state_dim = int(cfg.max_state_dim)
        self.max_num_embodiments = int(cfg.max_num_embodiments)
        self.state_history_length = int(getattr(cfg, "state_history_length", 1))
        self.action_horizon = int(cfg.action_horizon)
        self.num_inference_timesteps = int(cfg.num_inference_timesteps)
        self.num_timestep_buckets = int(cfg.num_timestep_buckets)
        self.add_pos_embed = bool(cfg.add_pos_embed)
        self.use_alternate_vl_dit = bool(getattr(cfg, "use_alternate_vl_dit", True))
        self.state_dropout_prob = float(getattr(cfg, "state_dropout_prob", 0.0))
        self.noise_s = float(cfg.noise_s)

        # --- Diffusion transformer (the bulk of the pretrained weights) ---
        diffusion_model_cfg = _to_plain(cfg.diffusion_model_cfg)
        if self.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **diffusion_model_cfg,
                cross_attention_dim=self.backbone_embedding_dim,
                attend_text_every_n_blocks=int(getattr(cfg, "attend_text_every_n_blocks", 2)),
            )
        else:
            self.model = DiT(**diffusion_model_cfg, cross_attention_dim=self.backbone_embedding_dim)

        # --- Multi-embodiment projectors (category-specific rows, 32 slots) ---
        self.state_encoder = CategorySpecificMLP(
            num_categories=self.max_num_embodiments,
            input_dim=self.max_state_dim * self.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.max_action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=self.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=self.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.max_action_dim,
        )

        # --- VL feature adapters ---
        self.vlln = nn.LayerNorm(self.backbone_embedding_dim) if bool(getattr(cfg, "use_vlln", True)) else nn.Identity()
        vl_sa_cfg = _to_plain(getattr(cfg, "vl_self_attention_cfg", None))
        if vl_sa_cfg and int(vl_sa_cfg.get("num_layers", 0)) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_sa_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if self.add_pos_embed:
            self.position_embedding = nn.Embedding(int(cfg.max_seq_len), self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(float(cfg.noise_beta_alpha), float(cfg.noise_beta_beta))

    # ------------------------------------------------------------------
    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (1 - sample) * self.noise_s

    def process_backbone_output(self, vl_embeds):
        vl_embeds = self.vlln(vl_embeds)
        vl_embeds = self.vl_self_attention(vl_embeds)
        return vl_embeds

    def _run_dit(self, sa_embs, vl_embeds, t_discretized, image_mask, backbone_attention_mask):
        if self.use_alternate_vl_dit:
            out = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                timestep=t_discretized,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )
        else:
            out = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embeds,
                encoder_attention_mask=backbone_attention_mask,
                timestep=t_discretized,
            )
        # DiT.forward returns a tensor (return_all_hidden_states defaults to False)
        return out[0] if isinstance(out, tuple) else out

    def forward(
        self,
        vl_embeds: torch.Tensor,          # [B, S, backbone_embedding_dim]
        actions: torch.Tensor,            # [B, H, max_action_dim]
        state: torch.Tensor,              # [B, state_history_length, max_state_dim]
        embodiment_id: torch.Tensor,      # [B]
        image_mask: torch.Tensor,         # [B, S] bool
        backbone_attention_mask: torch.Tensor,  # [B, S] bool
        action_mask: torch.Tensor,        # [B, H, max_action_dim] {0,1}
    ):
        vl_embeds = self.process_backbone_output(vl_embeds)
        device = vl_embeds.device

        # state: [B, T, D] -> [B, 1, T*D]
        state = state.reshape(state.shape[0], 1, -1)
        state_features = self.state_encoder(state, embodiment_id)
        if self.training and self.state_dropout_prob > 0:
            keep = (torch.rand(state_features.shape[0], device=device) >= self.state_dropout_prob)
            state_features = state_features * keep[:, None, None].to(state_features.dtype)

        # noise the action trajectory (flow matching)
        noise = torch.randn(actions.shape, device=device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=device, dtype=actions.dtype)[:, None, None]
        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)
        if self.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)

        sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output = self._run_dit(sa_embs, vl_embeds, t_discretized, image_mask, backbone_attention_mask)
        pred = self.action_decoder(model_output, embodiment_id)
        pred_actions = pred[:, -actions.shape[1]:]

        # NOTE: ``action_mask`` is a WEIGHT, not just a {0,1} validity flag — the
        # framework may scale flagged rows (e.g. CosmosGR00T_N1d7._frame_weights
        # up-weights press-phase frames). Normalising by ``action_mask.sum()``
        # therefore yields a weighted mean, which keeps the loss on the same scale
        # whatever the weights are. Do not replace it with a plain ``.mean()``.
        per_elem = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = per_elem.sum() / (action_mask.sum() + 1e-6)
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embeds: torch.Tensor,
        state: torch.Tensor,
        embodiment_id: torch.Tensor,
        image_mask: torch.Tensor,
        backbone_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        vl_embeds = self.process_backbone_output(vl_embeds)
        device = vl_embeds.device
        batch_size = vl_embeds.shape[0]

        state = state.reshape(state.shape[0], 1, -1)
        state_features = self.state_encoder(state, embodiment_id)

        actions = torch.randn(
            (batch_size, self.action_horizon, self.max_action_dim),
            dtype=vl_embeds.dtype, device=device,
        )
        dt = 1.0 / self.num_inference_timesteps
        for i in range(self.num_inference_timesteps):
            t_cont = i / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps = torch.full((batch_size,), t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps, embodiment_id)
            if self.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                action_features = action_features + self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = torch.cat((state_features, action_features), dim=1)
            model_output = self._run_dit(sa_embs, vl_embeds, timesteps, image_mask, backbone_attention_mask)
            pred = self.action_decoder(model_output, embodiment_id)
            pred_velocity = pred[:, -self.action_horizon:]
            actions = actions + dt * pred_velocity
        return actions


def get_action_model_n1d7(config=None):
    """Factory: build GR00TN1d7ActionHead from the global framework config."""
    return GR00TN1d7ActionHead(full_config=config)
