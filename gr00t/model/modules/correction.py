# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Corrector network for Correction Training (arXiv 2610.14136).

GR00T's flow-matching head uses t=0 for noise and t=1 for data:
    z = (1 - t) * noise + t * a,      v = a - noise.
This matches the reference implementation's ``data_time=1`` / ``prediction="v"`` (SiT)
setting. Following that recipe, the corrector receives the stop-gradient clean-action
proposal ``x_hat = z + (1 - t) * v_hat`` (plus the original time and conditioning), outputs a
velocity, and is combined with the stop-gradient base velocity through a learned gate:
    v_1 = sg(v_0) + g * (C(sg(x_hat), t, cond) - sg(v_0)).
"""

import copy

import torch
from torch import nn


class ActionCorrector(nn.Module):
    """A full copy of the action head's denoising network plus a learned output gate."""

    def __init__(self, head: nn.Module, gate_init: float = 0.0, gate_shape: str = "scalar"):
        super().__init__()
        if gate_shape not in ("scalar", "channel"):
            raise ValueError(
                f"correction_gate_shape must be 'scalar' or 'channel', got {gate_shape}"
            )
        self.add_pos_embed = bool(head.config.add_pos_embed)
        self.gate_init = float(gate_init)
        self.state_encoder = copy.deepcopy(head.state_encoder)
        self.action_encoder = copy.deepcopy(head.action_encoder)
        self.model = copy.deepcopy(head.model)
        self.action_decoder = copy.deepcopy(head.action_decoder)
        if self.add_pos_embed:
            self.position_embedding = copy.deepcopy(head.position_embedding)
        channels = 1 if gate_shape == "scalar" else head.action_dim
        ref = next(head.action_decoder.parameters())
        self.gate = nn.Parameter(
            torch.full((channels,), self.gate_init, device=ref.device, dtype=ref.dtype)
        )

    def _submodule_names(self) -> list[str]:
        names = ["state_encoder", "action_encoder", "model", "action_decoder"]
        if self.add_pos_embed:
            names.append("position_embedding")
        return names

    @torch.no_grad()
    def initialize_from(self, head: nn.Module) -> None:
        """Copy the (pretrained) denoiser weights into the corrector and reset the gate.

        Call after the base weights are loaded, never after loading a trained pair.
        """
        for name in self._submodule_names():
            getattr(self, name).load_state_dict(getattr(head, name).state_dict(), strict=True)
        self.gate.fill_(self.gate_init)

    def combine(self, output: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Gated mix ``skip + g * (output - skip)`` (``gate_mix="difference"``)."""
        g = self.gate.to(output.dtype).view(1, 1, -1)
        return skip + g * (output - skip)
