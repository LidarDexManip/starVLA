"""
QwenOFTMem (2026-11-06): QwenOFT with a 1 s GRU memory in front of its action head.

QwenOFT maps each of the 30 action-token embeddings to its action row on its own, from the current frame only. This
framework keeps the action-token embeddings of the last few inferences, mean-pooled to one vector each, runs a GRU
over them (oldest first, the current frame last) and adds a projection of the final state to every current action
token before the MLP head:

    q'_t = q_t + W h_t,   h_t = GRU([pool(q_{t-K+1}), ..., pool(q_t)]),   chunk_t = head(q'_t)

W starts at zero, so an untrained memory is plain QwenOFT.

Training reads the history from the loader (examples' history_images / history_state, oldest first; the pipette
"_mem6x12" data config: 6 steps 12 frames apart, i.e. the robot's 5 Hz), encodes the history frames WITHOUT gradient
(their pooled embeddings still come from the current VLM weights) and the current frame with gradient.

Serving keeps the pooled embeddings of the previous inferences in a buffer (one per predict_action call, the 5 Hz
cadence); reset_memory() clears it -- the policy server calls it on every policy reset (episode boundaries). Before
the buffer fills, missing history repeats the oldest frame available, as the loader clamps at the episode start.
"""
from collections import deque
from typing import List

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from starVLA.model.framework.VLM4A.QwenOFT import Qwenvl_OFT, _per_view_sizes
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenOFTMem")
class Qwenvl_OFT_Mem(Qwenvl_OFT):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)
        mc = self.config.framework.get("memory", None) or {}
        self.mem_steps = int(mc.get("steps", 6))
        dim = int(mc.get("dim", 512))
        hidden = int(self.config.framework.action_model.action_hidden_dim)
        self.memory_in = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, dim), nn.SiLU())
        self.memory_gru = nn.GRU(dim, dim, batch_first=True)
        self.memory_out = nn.Linear(dim, hidden)
        nn.init.zeros_(self.memory_out.weight)
        nn.init.zeros_(self.memory_out.bias)
        self._memory = deque(maxlen=max(0, self.mem_steps - 1))

    # ---- shared ----
    def _prompts(self, instructions, state):
        if state is not None:
            instructions = self.add_discretized_state_to_instruction(instructions, state)
        suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{self.action_token * self.chunk_len}<action>."
        return [x + suffix for x in instructions]

    def _queries(self, batch_images, prompts):
        qin = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=prompts)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**qin, output_attentions=False, output_hidden_states=True, return_dict=True)
        return self._gather_action_token_embeddings(out.hidden_states[-1], qin.get("input_ids", None),
                                                    action_token_id=self.action_token_id)

    def _remember(self, pooled_seq):
        """(B, K, H) pooled embeddings, oldest first -> (B, 1, H) correction for the current action tokens"""
        # in the memory weights' dtype with autocast off: float32 in training; the policy server casts the whole
        # framework to bf16, and cuDNN rejects a bf16 GRU fed float32 under the float32 autocast
        dtype = self.memory_gru.weight_ih_l0.dtype
        with torch.autocast("cuda", enabled=False):
            _, h = self.memory_gru(self.memory_in(pooled_seq.to(dtype)).contiguous())
            return self.memory_out(h[-1]).float().unsqueeze(1)

    # ---- training ----
    def forward(self, examples: List[dict] = None, **kwargs):
        instructions = [e["lang"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None
        q = self._queries([e["image"] for e in examples], self._prompts(instructions, state))     # (B, 30, H)
        k = len(examples[0].get("history_images", []))
        if k:
            imgs = [e["history_images"][t] for e in examples for t in range(k)]
            hs = [e["history_state"][t] for e in examples for t in range(k)] if state is not None else None
            with torch.no_grad():
                hq = self._queries(imgs, self._prompts([x for x in instructions for _ in range(k)], hs))
            past = hq.float().mean(1).view(len(examples), k, -1)
        else:
            past = q.new_zeros(len(examples), 0, q.shape[-1]).float()
        with torch.autocast("cuda", dtype=torch.float32):
            seq = torch.cat([past, q.float().mean(1, keepdim=True)], 1)
            pred = self.action_model.predict_action(q.float() + self._remember(seq))
            actions = torch.tensor(np.array([e["action"] for e in examples]), device=pred.device, dtype=pred.dtype)
            target = actions[:, -self.action_horizon:, :]
            lws = [e.get("loss_weight") for e in examples]
            timestep_w = None
            if any(w is not None for w in lws):
                H = target.shape[1]
                timestep_w = torch.tensor(np.stack([np.ones(H, np.float32) if w is None else
                                                    np.asarray(w, np.float32).reshape(-1)[-H:] for w in lws]),
                                          device=pred.device, dtype=pred.dtype)
            if self.grasp_loss_weight != 1.0 or timestep_w is not None:
                loss = self._weighted_l1(pred, target, timestep_w)
            else:
                loss = self.l1_loss(pred, target)
        return {"action_loss": loss}

    # ---- serving ----
    def reset_memory(self):
        self._memory.clear()

    @torch.inference_mode()
    def predict_action(self, examples: List[dict] = None, **kwargs) -> dict:
        if type(examples) is not list:
            examples = [examples]
        size = getattr(self.config.datasets.vla_data, "obs_image_size", None)

        def images(views_list):
            views_list = [to_pil_preserve(v) for v in views_list]
            if size:
                sizes = _per_view_sizes(size, len(views_list[0]))
                views_list = [[im.resize(sz) for im, sz in zip(views, sizes)] for views in views_list]
            return views_list

        instructions = [e["lang"] for e in examples]
        state = [e["state"] for e in examples] if "state" in examples[0] else None
        q = self._queries(images([e["image"] for e in examples]), self._prompts(instructions, state))
        cur = q.float().mean(1)                                          # (B, H)
        k = len(examples[0].get("history_images", []))
        if k:                                  # loader samples (the trainer's held-out eval): the history comes with them
            hs = [e["history_state"][t] for e in examples for t in range(k)] if state is not None else None
            hq = self._queries(images([e["history_images"][t] for e in examples for t in range(k)]),
                               self._prompts([x for x in instructions for _ in range(k)], hs))
            seq = torch.cat([hq.float().mean(1).view(len(examples), k, -1), cur.unsqueeze(1)], 1)
        else:                                  # serving: one stream, the history is this episode's earlier inferences
            if len(examples) != 1:
                raise ValueError("QwenOFTMem serves one stream at a time (its memory is per episode)")
            past = list(self._memory)
            km = self.mem_steps - 1
            if len(past) < km:                                           # the loader clamps at the episode start
                past = [past[0] if past else cur] * (km - len(past)) + past
            seq = torch.stack(past + [cur], 1)
            self._memory.append(cur)
        with torch.autocast("cuda", dtype=torch.float32):
            pred = self.action_model.predict_action(q.float() + self._remember(seq))
        return {"normalized_actions": pred.detach().cpu().numpy()}
