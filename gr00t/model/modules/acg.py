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

"""Action Coherence Guidance (ACG), Park & Kim et al., ICRA 2026 (arXiv:2510.22201).

ACG builds the "bad" prediction of classifier-free-style guidance from the model itself:
in a few self-attention blocks of the DiT the attention map is replaced by the identity,
so every action token only sees itself and temporal coherence between action tokens is
lost. Sampling then extrapolates away from that incoherent prediction,
``v = v_bad + w * (v_main - v_bad)``.
"""

from contextlib import contextmanager
from typing import Iterable, Optional

from diffusers.models.attention import Attention
import torch


# Self-attention blocks perturbed by default. The ACG reference perturbs blocks 7, 9, 11 of the
# 16-block GR00T-N1 DiT (44-69% depth); these are the same relative depths in the 32-block DiT.
DEFAULT_ACG_SKIP_BLOCKS = (15, 19, 23)


class IdentityAttnProcessor:
    """Attention with an identity attention map: ``out = to_out(to_v(x))``.

    Equivalent to softmax(QK^T) = I (the "incoherent variant" of ACG), so queries and keys
    are never computed.
    """

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        if encoder_hidden_states is not None:
            raise ValueError("Identity attention is only defined for self-attention blocks")
        residual = hidden_states
        hidden_states = attn.to_v(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def validate_acg_blocks(blocks: Iterable[int], num_layers: int) -> list[int]:
    """Check that the blocks exist and are self-attention blocks (odd index in the DiT)."""
    blocks = sorted({int(b) for b in blocks})
    if not blocks:
        raise ValueError("ACG needs at least one block to perturb")
    for b in blocks:
        if not 0 <= b < num_layers:
            raise ValueError(f"ACG block {b} is out of range for a DiT with {num_layers} blocks")
        if b % 2 != 1:
            raise ValueError(
                f"ACG block {b} is a cross-attention block; only self-attention (odd) "
                "blocks can be perturbed"
            )
    return blocks


@contextmanager
def acg_perturbation(dit: torch.nn.Module, blocks: Iterable[int]):
    """Temporarily replace the self-attention of ``dit.transformer_blocks[blocks]`` with identity."""
    saved = {}
    try:
        for b in blocks:
            attn = dit.transformer_blocks[b].attn1
            saved[b] = attn.processor
            attn.set_processor(IdentityAttnProcessor())
        yield
    finally:
        for b, processor in saved.items():
            dit.transformer_blocks[b].attn1.set_processor(processor)
