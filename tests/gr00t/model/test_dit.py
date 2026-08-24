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

"""
Tests for DiT / AlternateVLDiT's direction conditioning (round-trip
consistency's direction-embedding injection into `temb` - see the
integration plan). Covers both DiT variants since AlternateVLDiT.forward is
a full override (not a super() call), so the temb patch had to be applied
to both independently.
"""

from gr00t.model.modules.dit import AlternateVLDiT, DiT
import torch


_DIFFUSION_CFG = dict(
    positional_embeddings=None,
    num_layers=2,
    num_attention_heads=2,
    attention_head_dim=32,
    norm_type="ada_norm",
    dropout=0.0,
    final_dropout=False,
    output_dim=64,
    interleave_self_attention=True,
)
_CROSS_ATTENTION_DIM = 64
_INNER_DIM = _DIFFUSION_CFG["num_attention_heads"] * _DIFFUSION_CFG["attention_head_dim"]


def _make_dit() -> DiT:
    return DiT(**_DIFFUSION_CFG, cross_attention_dim=_CROSS_ATTENTION_DIM)


def _make_alternate_vl_dit() -> AlternateVLDiT:
    return AlternateVLDiT(
        **_DIFFUSION_CFG, cross_attention_dim=_CROSS_ATTENTION_DIM, attend_text_every_n_blocks=2
    )


def _inputs(batch_size=2, seq_len=5, encoder_seq_len=6):
    hidden_states = torch.randn(batch_size, seq_len, _INNER_DIM)
    encoder_hidden_states = torch.randn(batch_size, encoder_seq_len, _CROSS_ATTENTION_DIM)
    timestep = torch.zeros(batch_size, dtype=torch.long)
    return hidden_states, encoder_hidden_states, timestep


def _image_masks(batch_size, encoder_seq_len):
    image_mask = torch.zeros(batch_size, encoder_seq_len, dtype=torch.bool)
    image_mask[:, : encoder_seq_len // 2] = True
    backbone_attention_mask = torch.ones(batch_size, encoder_seq_len, dtype=torch.bool)
    return image_mask, backbone_attention_mask


class TestDirectionEmbeddingInit:
    def test_zero_init_dit(self):
        assert torch.all(_make_dit().direction_embedding.weight == 0)

    def test_zero_init_alternate_vl_dit(self):
        assert torch.all(_make_alternate_vl_dit().direction_embedding.weight == 0)


class TestDirectionConditioningNoOpAtInit:
    """Zero-initialized direction_embedding => an exact no-op until trained -
    forward (direction=0) and backward (direction=1) outputs are bit-identical."""

    def test_dit_direction_none_matches_zeros(self):
        dit = _make_dit()
        dit.eval()
        hidden_states, encoder_hidden_states, timestep = _inputs()
        batch_size = hidden_states.shape[0]
        with torch.no_grad():
            out_none = dit(hidden_states, encoder_hidden_states, timestep=timestep, direction=None)
            out_zeros = dit(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                direction=torch.zeros(batch_size, dtype=torch.long),
            )
        assert torch.equal(out_none, out_zeros)

    def test_dit_forward_backward_identical_at_init(self):
        dit = _make_dit()
        dit.eval()
        hidden_states, encoder_hidden_states, timestep = _inputs()
        batch_size = hidden_states.shape[0]
        with torch.no_grad():
            out_fwd = dit(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                direction=torch.zeros(batch_size, dtype=torch.long),
            )
            out_bwd = dit(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                direction=torch.ones(batch_size, dtype=torch.long),
            )
        assert torch.equal(out_fwd, out_bwd)

    def test_alternate_vl_dit_forward_backward_identical_at_init(self):
        dit = _make_alternate_vl_dit()
        dit.eval()
        hidden_states, encoder_hidden_states, timestep = _inputs()
        batch_size = hidden_states.shape[0]
        image_mask, backbone_attention_mask = _image_masks(
            batch_size, encoder_hidden_states.shape[1]
        )
        with torch.no_grad():
            out_fwd = dit(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                direction=torch.zeros(batch_size, dtype=torch.long),
            )
            out_bwd = dit(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                direction=torch.ones(batch_size, dtype=torch.long),
            )
        assert torch.equal(out_fwd, out_bwd)


class TestDirectionConditioningDivergesAfterTraining:
    def test_dit_diverges_after_perturbation(self):
        """Once direction_embedding's backward row is nonzero (as it would be
        after some training), direction=0 and direction=1 must diverge -
        confirms the summation into temb actually reaches every block."""
        dit = _make_dit()
        dit.eval()
        with torch.no_grad():
            dit.direction_embedding.weight[1].normal_(mean=0.0, std=1.0)
        hidden_states, encoder_hidden_states, timestep = _inputs()
        batch_size = hidden_states.shape[0]
        with torch.no_grad():
            out_fwd = dit(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                direction=torch.zeros(batch_size, dtype=torch.long),
            )
            out_bwd = dit(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                direction=torch.ones(batch_size, dtype=torch.long),
            )
        assert not torch.allclose(out_fwd, out_bwd)

    def test_alternate_vl_dit_diverges_after_perturbation(self):
        dit = _make_alternate_vl_dit()
        dit.eval()
        with torch.no_grad():
            dit.direction_embedding.weight[1].normal_(mean=0.0, std=1.0)
        hidden_states, encoder_hidden_states, timestep = _inputs()
        batch_size = hidden_states.shape[0]
        image_mask, backbone_attention_mask = _image_masks(
            batch_size, encoder_hidden_states.shape[1]
        )
        with torch.no_grad():
            out_fwd = dit(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                direction=torch.zeros(batch_size, dtype=torch.long),
            )
            out_bwd = dit(
                hidden_states,
                encoder_hidden_states,
                timestep=timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
                direction=torch.ones(batch_size, dtype=torch.long),
            )
        assert not torch.allclose(out_fwd, out_bwd)
