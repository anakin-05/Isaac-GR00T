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
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

import math

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
from gr00t.model.modules.acg import IdentityAttnProcessor, acg_perturbation, validate_acg_blocks
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8):
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
            "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
            "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }
    )


def _make_action_input(config, batch_size=2):
    return BatchFeature(
        data={
            "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
            "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
        }
    )


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_forward_with_state_dropout(self):
        config = _small_config(state_dropout_prob=0.5)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert torch.isfinite(out["loss"])


class TestActionHeadGetAction:
    """Test inference (denoising loop)."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]  # get_action doesn't need ground-truth action
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1


def _clone_batch(batch: BatchFeature) -> BatchFeature:
    # get_action applies vlln to backbone_output in place, so each call needs its own copy.
    return BatchFeature(data={k: v.clone() for k, v in batch.items()})


def _make_head(seed: int, config) -> Gr00tN1d7ActionHead:
    torch.manual_seed(seed)
    head = Gr00tN1d7ActionHead(config)
    # A freshly initialized LayerNorm is (nearly) idempotent, which would hide vlln being
    # applied twice to the shared backbone output; randomize its affine parameters.
    with torch.no_grad():
        head.vlln.weight.uniform_(0.5, 2.0)
        head.vlln.bias.normal_()
    head.eval()
    return head


class TestActionHeadAutoguidance:
    """Test autoguidance: v = v_guide + w * (v_main - v_guide) in the denoising loop."""

    def _sample(self, head, backbone_output, action_input, guide=None, guidance_scale=1.0):
        torch.manual_seed(1234)
        return head.get_action(
            _clone_batch(backbone_output),
            action_input,
            guide=guide,
            guidance_scale=guidance_scale,
        )["action_pred"]

    def _inputs(self, config):
        action_input = _make_action_input(config)
        del action_input["action"]
        return _make_backbone_output(config), action_input

    def test_scale_one_matches_unguided(self):
        config = _small_config()
        head, guide = _make_head(0, config), _make_head(1, config)
        backbone_output, action_input = self._inputs(config)
        unguided = self._sample(head, backbone_output, action_input)
        guided = self._sample(head, backbone_output, action_input, guide, guidance_scale=1.0)
        assert torch.equal(unguided, guided)

    def test_identical_guide_matches_unguided(self):
        config = _small_config()
        head = _make_head(0, config)
        guide = _make_head(0, config)
        backbone_output, action_input = self._inputs(config)
        unguided = self._sample(head, backbone_output, action_input)
        guided = self._sample(head, backbone_output, action_input, guide, guidance_scale=3.0)
        torch.testing.assert_close(guided, unguided, rtol=1e-5, atol=1e-5)

    def test_guided_matches_manual_rollout(self):
        config = _small_config()
        head, guide = _make_head(0, config), _make_head(1, config)
        backbone_output, action_input = self._inputs(config)
        w = 2.0
        guided = self._sample(head, backbone_output, action_input, guide, guidance_scale=w)

        main_feats = head._encode_features(_clone_batch(backbone_output), action_input)
        guide_feats = guide._encode_features(_clone_batch(backbone_output), action_input)
        batch_size = action_input["state"].shape[0]
        torch.manual_seed(1234)
        actions = torch.randn(batch_size, config.action_horizon, config.max_action_dim)
        n = config.num_inference_timesteps
        for step in range(n):
            t = torch.full((batch_size,), int(step / n * config.num_timestep_buckets))
            embodiment_id = action_input["embodiment_id"]
            v_main = head._predict_velocity(
                actions,
                t,
                main_feats.state_features,
                main_feats.backbone_features,
                embodiment_id,
                backbone_output,
            )
            v_guide = guide._predict_velocity(
                actions,
                t,
                guide_feats.state_features,
                guide_feats.backbone_features,
                embodiment_id,
                backbone_output,
            )
            actions = actions + (v_guide + w * (v_main - v_guide)) / n

        torch.testing.assert_close(guided, actions, rtol=1e-5, atol=1e-5)
        unguided = self._sample(head, backbone_output, action_input)
        assert not torch.allclose(guided, unguided, atol=1e-3)

    def test_guided_with_alternate_vl_dit(self):
        dit_cfg = dict(_small_config().diffusion_model_cfg, num_layers=4)
        config = _small_config(use_alternate_vl_dit=True, diffusion_model_cfg=dit_cfg)
        head, guide = _make_head(0, config), _make_head(1, config)
        backbone_output, action_input = self._inputs(config)
        backbone_output["image_mask"][:, ::2] = False  # half text tokens, half image tokens
        # The real backbone returns a bool mask (attention_mask == 1).
        backbone_output["backbone_attention_mask"] = backbone_output[
            "backbone_attention_mask"
        ].bool()
        guided = self._sample(head, backbone_output, action_input, guide, guidance_scale=2.0)
        assert guided.shape == (2, config.action_horizon, config.max_action_dim)
        assert torch.isfinite(guided).all()


class TestActionHeadTrajectory:
    """return_trajectory exposes the denoising path and per-step guidance shift."""

    def test_trajectory_and_shift(self):
        config = _small_config()
        head, guide = _make_head(0, config), _make_head(1, config)
        action_input = _make_action_input(config)
        del action_input["action"]
        backbone_output = _make_backbone_output(config)

        def sample(**kw):
            torch.manual_seed(7)
            return head.get_action(_clone_batch(backbone_output), action_input, **kw)

        plain = sample()
        guided = sample(guide=guide, guidance_scale=2.0, return_trajectory=True)
        n = config.num_inference_timesteps
        traj, shift = guided["action_trajectory"], guided["guidance_shift"]
        assert traj.shape == (n + 1, 2, config.action_horizon, config.max_action_dim)
        assert shift.shape == (n, 2, config.action_horizon, config.max_action_dim)
        assert torch.equal(traj[-1], guided["action_pred"])
        assert shift.abs().max() > 0
        # Tracking the path does not change the sample, and unguided shifts are zero.
        again = sample(guide=guide, guidance_scale=2.0)
        assert torch.equal(again["action_pred"], guided["action_pred"])
        unguided = sample(return_trajectory=True)
        assert torch.equal(unguided["action_pred"], plain["action_pred"])
        assert unguided["guidance_shift"].abs().max() == 0


class TestActionHeadACG:
    """Test Action Coherence Guidance: identity self-attention gives the bad prediction."""

    BLOCKS = [1, 3]

    def _config(self):
        dit_cfg = dict(_small_config().diffusion_model_cfg, num_layers=4)
        return _small_config(diffusion_model_cfg=dit_cfg)

    def _sample(self, head, backbone_output, action_input, acg_blocks=None, guidance_scale=1.0):
        torch.manual_seed(1234)
        return head.get_action(
            _clone_batch(backbone_output),
            action_input,
            guidance_scale=guidance_scale,
            acg_blocks=acg_blocks,
        )["action_pred"]

    def _inputs(self, config):
        action_input = _make_action_input(config)
        del action_input["action"]
        return _make_backbone_output(config), action_input

    def test_identity_processor_ignores_attention_map(self):
        head = _make_head(0, self._config())
        attn = head.model.transformer_blocks[1].attn1
        x = torch.randn(2, 5, attn.query_dim)
        with acg_perturbation(head.model, [1]):
            out = attn(x)
            # Each token only sees itself: permuting the other tokens changes nothing.
            permuted = attn(torch.cat([x[:, :1], x[:, 1:].flip(1)], dim=1))
        torch.testing.assert_close(out[:, 0], permuted[:, 0])
        torch.testing.assert_close(out, attn.to_out[0](attn.to_v(x)))

    def test_processors_restored(self):
        head = _make_head(0, self._config())
        before = [type(b.attn1.processor) for b in head.model.transformer_blocks]
        with pytest.raises(RuntimeError):
            with acg_perturbation(head.model, self.BLOCKS):
                assert isinstance(
                    head.model.transformer_blocks[1].attn1.processor, IdentityAttnProcessor
                )
                raise RuntimeError
        assert [type(b.attn1.processor) for b in head.model.transformer_blocks] == before

    def test_scale_one_matches_unguided(self):
        config = self._config()
        head = _make_head(0, config)
        backbone_output, action_input = self._inputs(config)
        unguided = self._sample(head, backbone_output, action_input)
        guided = self._sample(head, backbone_output, action_input, self.BLOCKS, 1.0)
        assert torch.equal(unguided, guided)

    def test_guided_matches_manual_rollout(self):
        config = self._config()
        head = _make_head(0, config)
        backbone_output, action_input = self._inputs(config)
        w = 2.0
        guided = self._sample(head, backbone_output, action_input, self.BLOCKS, w)

        feats = head._encode_features(_clone_batch(backbone_output), action_input)
        batch_size = action_input["state"].shape[0]
        torch.manual_seed(1234)
        actions = torch.randn(batch_size, config.action_horizon, config.max_action_dim)
        n = config.num_inference_timesteps
        for step in range(n):
            t = torch.full((batch_size,), int(step / n * config.num_timestep_buckets))
            args = (
                actions,
                t,
                feats.state_features,
                feats.backbone_features,
                action_input["embodiment_id"],
                backbone_output,
            )
            v_main = head._predict_velocity(*args)
            v_bad = head._predict_velocity(*args, acg_blocks=self.BLOCKS)
            assert not torch.allclose(v_main, v_bad, atol=1e-4)
            actions = actions + (v_bad + w * (v_main - v_bad)) / n

        torch.testing.assert_close(guided, actions, rtol=1e-5, atol=1e-5)

    def test_alternate_vl_dit(self):
        dit_cfg = dict(_small_config().diffusion_model_cfg, num_layers=4)
        config = _small_config(use_alternate_vl_dit=True, diffusion_model_cfg=dit_cfg)
        head = _make_head(0, config)
        backbone_output, action_input = self._inputs(config)
        backbone_output["image_mask"][:, ::2] = False
        backbone_output["backbone_attention_mask"] = backbone_output[
            "backbone_attention_mask"
        ].bool()
        guided = self._sample(head, backbone_output, action_input, self.BLOCKS, 2.0)
        assert torch.isfinite(guided).all()

    def test_validate_blocks(self):
        assert validate_acg_blocks([19, 15, 15], 32) == [15, 19]
        for bad in ([], [14], [33], [-1]):
            with pytest.raises(ValueError):
                validate_acg_blocks(bad, 32)

    def test_set_acg_conflicts_with_autoguidance(self):
        from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7

        model = Gr00tN1d7.__new__(Gr00tN1d7)
        torch.nn.Module.__init__(model)
        model.autoguidance_head = object()
        model.acg_blocks = None
        with pytest.raises(ValueError):
            model.set_acg([1], 2.0)


class TestActionHeadAFIL:
    """Test Adaptive Failure-Informed Learning: v = v_s + lam (v_s - v_f), lam = a (1 - cos)."""

    REGION = (3, 5)  # (horizon, dim) of the real action chunk inside the padded output

    def _sample(self, head, backbone_output, action_input, fail=None, alpha=1.0, region=REGION):
        torch.manual_seed(1234)
        return head.get_action(
            _clone_batch(backbone_output),
            action_input,
            guide=fail,
            guidance_scale=alpha,
            adaptive_region=region,
        )["action_pred"]

    def _inputs(self, config):
        action_input = _make_action_input(config)
        del action_input["action"]
        return _make_backbone_output(config), action_input

    def test_alpha_zero_matches_unguided(self):
        config = _small_config()
        head, fail = _make_head(0, config), _make_head(1, config)
        backbone_output, action_input = self._inputs(config)
        unguided = self._sample(head, backbone_output, action_input)
        guided = self._sample(head, backbone_output, action_input, fail, alpha=0.0)
        assert torch.equal(unguided, guided)

    def test_identical_failure_head_matches_unguided(self):
        # Same velocities -> cos = 1 -> lambda = 0 for any alpha.
        config = _small_config()
        head, fail = _make_head(0, config), _make_head(0, config)
        backbone_output, action_input = self._inputs(config)
        unguided = self._sample(head, backbone_output, action_input)
        guided = self._sample(head, backbone_output, action_input, fail, alpha=5.0)
        torch.testing.assert_close(guided, unguided, rtol=1e-5, atol=1e-5)

    def test_matches_manual_rollout(self):
        config = _small_config()
        head, fail = _make_head(0, config), _make_head(1, config)
        backbone_output, action_input = self._inputs(config)
        alpha = 1.5
        guided = self._sample(head, backbone_output, action_input, fail, alpha)

        main_feats = head._encode_features(_clone_batch(backbone_output), action_input)
        fail_feats = fail._encode_features(_clone_batch(backbone_output), action_input)
        batch_size = action_input["state"].shape[0]
        torch.manual_seed(1234)
        actions = torch.randn(batch_size, config.action_horizon, config.max_action_dim)
        n = config.num_inference_timesteps
        horizon, dim = self.REGION
        lams = []
        for step in range(n):
            t = torch.full((batch_size,), int(step / n * config.num_timestep_buckets))
            emb = action_input["embodiment_id"]
            v_s = head._predict_velocity(
                actions,
                t,
                main_feats.state_features,
                main_feats.backbone_features,
                emb,
                backbone_output,
            )
            v_f = fail._predict_velocity(
                actions,
                t,
                fail_feats.state_features,
                fail_feats.backbone_features,
                emb,
                backbone_output,
            )
            a = v_s[:, :horizon, :dim].reshape(batch_size, -1)
            b = v_f[:, :horizon, :dim].reshape(batch_size, -1)
            cos = (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))
            lam = alpha * (1 - cos)
            lams.append(lam)
            actions = actions + (v_s + lam[:, None, None] * (v_s - v_f)) / n

        torch.testing.assert_close(guided, actions, rtol=1e-4, atol=1e-4)
        # The adaptive scale differs per sample, otherwise this would just be autoguidance.
        assert not torch.allclose(lams[0][0], lams[0][1], atol=1e-6)
        unguided = self._sample(head, backbone_output, action_input)
        assert not torch.allclose(guided, unguided, atol=1e-3)

    def test_cosine_ignores_padded_region(self):
        config = _small_config()
        head, fail = _make_head(0, config), _make_head(1, config)
        backbone_output, action_input = self._inputs(config)
        full = (config.action_horizon, config.max_action_dim)
        a = self._sample(head, backbone_output, action_input, fail, 1.0, region=self.REGION)
        b = self._sample(head, backbone_output, action_input, fail, 1.0, region=full)
        assert not torch.allclose(a, b, atol=1e-6)


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


def _beta_time_moments(alpha: float, beta: float, noise_s: float) -> tuple[float, float]:
    """Closed-form mean/variance of sample_time's output, derived from config.

    sample_time draws ``b ~ Beta(alpha, beta)`` and returns ``s = (1 - b) * noise_s``.
    The moments are re-derived here from the config values (not hand-picked
    constants and not a fingerprint of whatever the code currently emits), so
    the assertions are an independent oracle for the sampler rather than an
    author-vs-author restatement.
    """
    mean_b = alpha / (alpha + beta)
    var_b = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
    mean_s = (1.0 - mean_b) * noise_s
    var_s = (noise_s**2) * var_b
    return mean_s, var_s


class TestActionHeadTimeSamplingMetaSafe:
    """Oracle tests for sample_time under meta / no_init_weights construction.

    Regression guard: when the action head is built while the default device is
    meta (as happens inside a nested from_pretrained), sample_time must still
    produce a valid, correctly-distributed noise schedule on the requested
    device — not a crash or uninitialized values from a meta-backed Beta.
    """

    @pytest.mark.parametrize(
        "alpha,beta,noise_s",
        [(1.5, 1.0, 0.999), (2.0, 3.0, 0.95)],
    )
    def test_sample_time_under_meta_construction(self, alpha, beta, noise_s):
        config = _small_config(noise_beta_alpha=alpha, noise_beta_beta=beta, noise_s=noise_s)
        # Reproduce the production failure condition: the whole action head is
        # instantiated while the default device is meta.
        with torch.device("meta"):
            head = Gr00tN1d7ActionHead(config)

        torch.manual_seed(0)
        n = 200_000
        sample = head.sample_time(n, device="cpu", dtype=torch.float32)

        # Requested device/dtype honored even though the module was built on meta.
        assert sample.device.type == "cpu"
        assert sample.dtype == torch.float32
        # No uninitialized / nan / inf leakage from a meta-backed distribution.
        assert torch.isfinite(sample).all()
        # Transformed-Beta support: s = (1 - b) * noise_s with b in [0, 1].
        assert (sample >= 0).all()
        assert (sample <= noise_s + 1e-6).all()

        # Distribution matches Beta(alpha, beta) transformed; RHS from config.
        mean_s, var_s = _beta_time_moments(alpha, beta, noise_s)
        assert math.isclose(sample.mean().item(), mean_s, abs_tol=5e-3)
        assert math.isclose(sample.var(unbiased=False).item(), var_s, rel_tol=0.1)

    def test_sample_time_construction_device_invariant(self):
        """Construction device must not change the sampled noise stream."""
        config = _small_config()
        head_real = Gr00tN1d7ActionHead(config)
        with torch.device("meta"):
            head_meta = Gr00tN1d7ActionHead(config)

        n = 200_000
        torch.manual_seed(1234)
        s_real = head_real.sample_time(n, device="cpu", dtype=torch.float32)
        torch.manual_seed(1234)
        s_meta = head_meta.sample_time(n, device="cpu", dtype=torch.float32)

        # Same seed + config-derived sampler ⇒ byte-identical stream regardless
        # of whether the head was constructed on a real device or on meta.
        assert torch.equal(s_real, s_meta)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad
