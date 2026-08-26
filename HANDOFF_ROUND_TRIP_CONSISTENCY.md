# Round-Trip Consistency Integration — Handoff

Status as of 2026-08-26 (GPU fine-tune/eval section updated after actually running Steps
0-2 on a 40GB MIG slice — originally written 2026-08-24, GPU-blocked at the time). Full
design rationale lives in the plan file
`/home/duydt6/.claude/plans/integrate-round-trip-consistency-steady-shore.md` (context on
*why* this integration is built this way — read that first if picking this up cold). This
doc is the *what's done / what's left* addendum, written after actually running things.

## Done — code integration (complete, verified)

All file-by-file changes from the plan are implemented:

- `gr00t/data/state_action/action_chunking.py` — `ActionChunk.to_reversed_chunking`
- `gr00t/data/state_action/state_action_processor.py` — `apply_reversed`, `_reverse_relative_action`, `_reverse_absolute_action`, `_normalize_action_dict`
- `gr00t/model/modules/dit.py` — zero-init `direction_embedding`, `direction` param on `DiT.forward` and `AlternateVLDiT.forward`
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py` — vision-dropout masking, direction-weighted loss + forward/backward breakdown, `direction`/`drop_vision` threaded through `get_action`/`get_action_with_features`
- `gr00t/model/gr00t_n1d7/setup.py` — new config kwargs threaded to `AutoModel`/`AutoProcessor`, `direction_embedding` missing-key allowlist
- `gr00t/model/gr00t_n1d7/processing_gr00t_n1d7.py` — backward-example sampling in `__call__`, persistence in `save_pretrained`/`from_pretrained`
- `gr00t/configs/model/gr00t_n1d7.py`, `gr00t/configs/finetune_config.py`, `gr00t/experiment/launch_finetune.py` — new config fields threaded end-to-end
- `gr00t/experiment/trainer.py` — `train_forward_action_loss` / `train_backward_action_loss` logging
- `gr00t/policy/gr00t_policy.py` — fixed a real pre-existing bug: `options` was accepted but never forwarded to `model.get_action`
- `gr00t/eval/round_trip_consistency.py` (new) — `compute_round_trip_consistency`, `plot_round_trip_consistency`
- `gr00t/eval/calibration.py` (new) — ported `fit_heteroscedastic_gaussian`/`select_degrees`/etc from the round-trip-consistency reference repo
- `gr00t/eval/open_loop_eval.py` — `--compute-round-trip-consistency`, `--rtc-depths`, `--rtc-save-plot-path`
- **New, beyond the original plan:** `gradient_checkpointing: bool = False` added to `FinetuneConfig`/`launch_finetune.py` (mirrors existing `TrainingConfig.gradient_checkpointing`, which wasn't exposed via the finetune CLI at all before). Turned out **not to reduce memory** in testing here — see Known Issues below — but it's a real, generically useful gap-fix independent of that.

Four real bugs fixed along the way (not in the original plan, found by actually running things):
- `gr00t/eval/calibration.py`: `np.trapezoid` doesn't exist in this repo's pinned numpy 1.26 — fixed to fall back to `np.trapz`.
- `gr00t/model/gr00t_n1d7/gr00t_n1d7.py`: `get_action`'s `drop_vision` option crashed on `.any()` when passed a plain Python bool (e.g. `options={"drop_vision": True}`) instead of a tensor — fixed to broadcast scalars the same way `direction` already did.
- `gr00t/data/state_action/state_action_processor.py` (found running the real fine-tune
  on LIBERO, not just DROID): `apply_reversed`'s ABSOLUTE-rep path assumed an action
  key's reference state key always has the same dimension as the action itself. True
  for DROID's `gripper_position` (both 1-dim, which is why 735 CPU tests never caught
  this) but false for `LIBERO_PANDA`'s `gripper` — a 2-dim two-finger state vs. a 1-dim
  actuation command that happen to share a name. Crashed with a `np.concatenate`
  `ValueError` on the very first training batch. The test suite's own fixture comment
  (`test_state_action_processor.py`'s `_self_consistent_absolute_setup`) had already
  flagged this exact assumption and quietly worked around it rather than fixing it — the
  gap just was never connected to the real `LIBERO_PANDA` embodiment's actual dataset.
  Fixed by skipping reversal for a dimension-mismatched key (one-time warning, key's
  forward-direction action/state left unchanged in the backward example) instead of
  crashing — consistent with the existing `ActionRepresentation.DELTA`-unsupported
  precedent and with Caveat 2 below (gripper reversal was already a regression-consistency
  approximation, not a physical-undo claim). New tests in
  `tests/gr00t/data/state_action/test_state_action_processor.py`
  (`TestApplyReversedDimMismatch`).
- `gr00t/eval/round_trip_consistency.py` (found running Step 2 against the real trained
  checkpoint, not the mocked-policy unit tests): the per-window loop indexed
  `backward_action[key][-1]` assuming a bare `(H, D)` chunk — the shape both of this
  module's own test stubs return. The real `Gr00tPolicy.get_action` contract is batched,
  `(B, H, D)` (see `Gr00tPolicy.check_action`), so `[-1]` grabbed the wrong axis entirely
  and every per-key shape check silently failed, producing zero valid round-trip windows
  at every depth for every trajectory (no crash, no error — just an empty result, which
  is what made this one easy to miss). Fixed by collapsing all leading dims before
  indexing the last row (`reshape(-1, D)[-1]`), which is correct for both the 2D test
  convention and the real 3D one. New regression test
  `TestComputeRoundTripConsistencyShapes::test_batched_policy_output_matches_unbatched`
  in `tests/gr00t/eval/test_round_trip_consistency.py`, asserting a batched and unbatched
  policy stub produce identical C_i.

## Done — tests (all passing)

```bash
uv sync --python 3.12 --extra dev
uv run python -m pytest tests/ -m "not gpu" -q --timeout=300 --ignore=tests/gr00t/eval/sim
# 689 passed, 9 skipped, 35 deselected (before this integration's own new tests)
```

New/extended test files, all passing:
- `tests/gr00t/data/state_action/test_action_chunking.py` (new)
- `tests/gr00t/data/state_action/test_state_action_processor.py` (extended — `apply_reversed` for both RELATIVE/EEF and ABSOLUTE reps)
- `tests/gr00t/model/test_dit.py` (new — direction-embedding no-op-at-init + divergence-after-training)
- `tests/gr00t/model/test_action_head.py` (extended — includes the critical
  `test_disabled_by_default_loss_formula_unchanged` regression guard)
- `tests/gr00t/eval/test_round_trip_consistency.py` (new — **runs against the real
  `demo_data/droid_sample` dataset**; the exact-recovery test confirms the full
  compose/decode pipeline gives C_i ≈ 0 at machine precision when fed the mathematically
  exact inverse)
- `tests/gr00t/eval/test_calibration.py` (new)
- `tests/fixtures/processor_config/processor_config.json` — updated with the 3 new processor fields (a pre-existing fixture-completeness test caught this)

Also fixed two of my own test bugs found by actually running pytest: a `libero_sim`
fixture quirk (state/action stats have independently-chosen, mismatched per-key dims —
harmless for existing tests since `apply_action` never combines them for ABSOLUTE-rep
keys, but `apply_reversed` genuinely needs them to agree) worked around with a
self-consistent statistics override; and pre-existing CPU floating-point non-determinism
at the ~1e-7 level in multi-step model calls (unrelated to this change) meant a couple of
"no-op at init" assertions needed `torch.allclose` instead of `torch.equal`.

**This is the load-bearing verification** for the integration's correctness — it doesn't
need a GPU or the gated checkpoint.

## Done — GPU fine-tune + eval run (2026-08-26, on a 40GB H100 MIG slice)

The 2x RTX 5090 blocker below (31.34 GiB usable each, insufficient for this model's fixed
FP32 overhead) was resolved by moving to a machine with `CUDA_VISIBLE_DEVICES` MIG slices
(`3g.40gb`, 39.5 GiB usable per `torch.cuda.get_device_properties`) — confirming the
handoff's own "≥40 GB" recommendation. The RTX 5090 findings below are kept for reference
(gradient-checkpointing no-op finding especially — still true, still relevant on any GPU).

**Step 0 (smoke test, `demo_data/droid_sample`, 10 steps, single GPU, batch 8): passed.**
`train_forward_action_loss`/`train_backward_action_loss` finite from step 0, checkpoint
saved cleanly, `direction_embedding not in checkpoint - zero-initialized (no-op)` logged
exactly once. Peak usage comfortably inside 39.5 GiB (no OOM at batch 8, unlike every
batch tried on the 31.34 GiB RTX 5090s).

**Step 1 (real fine-tune, 1000 steps, single GPU — 2 GPUs weren't available this time so
this used 1 instead of the plan's `torchrun --nproc_per_node=2`, everything else
identical): passed.** `/tmp/droid_rtc_finetune/checkpoint-1000`. Wall time ~9.3 minutes at
batch 8 (~2.5 it/s). Loss trended down cleanly and stayed finite throughout: forward/backward
losses started around 0.13-0.40 (step 0) and were down to the 0.03-0.10 range by step 1000.
`Training completed!` / `Model saved to /tmp/droid_rtc_finetune`.

**Step 2 (open-loop eval + round-trip diagnostic on `checkpoint-1000`, traj-ids 1 2): passed,
after fixing the `round_trip_consistency.py` batch-shape bug above.** Real numbers:

| Traj | Open-loop MSE | Open-loop MAE | C_1 (mean±std, n) | C_2 (mean±std, n) | C_4 (mean±std, n) |
|---|---|---|---|---|---|
| 1 | 0.00172 | 0.0278 | 0.0099 ± 0.0051 (4) | 0.0190 ± 0.0071 (3) | 0.0699 ± 0 (1) |
| 2 | 0.00816 | 0.0604 | 0.0515 ± 0.0294 (4) | 0.3899 ± 0.3840 (3) | 0.7006 ± 0 (1) |

Two encouraging (but, per the plan's Caveat 5, not statistically meaningful at n=1-4 per
depth / 3 total episodes) qualitative patterns: C_i rises monotonically with depth on both
trajectories, matching the mechanism's own prediction (a deeper round trip should be less
self-consistent); and trajectory 2 — which has ~5x higher open-loop MSE than trajectory 1 —
also has higher C_i at every depth, the exact correlation this diagnostic is meant to
surface. Plots saved to `/tmp/open_loop_eval/traj_rtc_traj{1,2}_rtc.png`.

**Phase B/C/D verification criteria from the plan are now all satisfied**, not just
Phase A (CPU tests). This was blocked purely on GPU memory, never on the code itself — the
two bugs found and fixed above were both latent (never exercised until running against
LIBERO data and a real trained checkpoint, respectively, neither of which the original
Phase A CPU-only verification touched).

Exact commands (Step 1 shown; Step 0 is identical with `--max-steps 10 --save-steps 10`,
Step 2 is the `open_loop_eval.py` invocation already shown in the plan):

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=<mig-or-gpu-uuid> uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B --dataset-path demo_data/droid_sample \
    --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT --num-gpus 1 \
    --output-dir /tmp/droid_rtc_finetune --max-steps 1000 --save-steps 250 \
    --global-batch-size 8 --dataloader-num-workers 4 \
    --enable-direction-conditioning --p-backward 0.5 \
    --vision-dropout-prob-backward 1.0 --backward-loss-weight 1.0
```

CPU suite re-run after both fixes: `738 passed, 9 skipped, 35 deselected` (up from 735 —
the 3 new regression tests above), same 2 pre-existing environmental failures as before
(unrelated to this integration — see Known Issues below), no regressions.

### In progress: real LIBERO+RTC fine-tune (separate from the droid_sample validation above)

Beyond the plan's own droid_sample-based Steps 0-2, also launched a full-scale fine-tune
on `examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot` with RTC enabled, matching the
hyperparameters of this repo's own pre-existing (non-RTC) LIBERO baseline recipe
(`examples/LIBERO/logs/libero-base_line/libero_goal`: `max_steps=40658`,
`global_batch_size=32`, `warmup_ratio=0.05`, `learning_rate=1e-4`, `weight_decay=1e-5`):

```bash
HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
CUDA_VISIBLE_DEVICES=<mig-or-gpu-uuid> uv run python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot/ \
    --embodiment-tag LIBERO_PANDA --num-gpus 1 \
    --output-dir examples/LIBERO/logs/libero_rtc \
    --max-steps 40658 --save-steps 1000 --save-total-limit 5 \
    --warmup-ratio 0.05 --weight-decay 1e-5 --learning-rate 1e-4 \
    --global-batch-size 32 --dataloader-num-workers 4 \
    --shard-size 1024 --num-shards-per-epoch 100000 --episode-sampling-rate 0.1 \
    --enable-direction-conditioning --p-backward 0.5 \
    --vision-dropout-prob-backward 1.0 --backward-loss-weight 1.0
```

A first attempt at this (before the `apply_reversed` fix above existed) got as far as
saving its config/processor but crashed on the very first training batch with the
dimension-mismatch bug — the batch size (32) was not the cause, despite looking like an
OOM candidate at first glance. After the fix, a 20-step smoke test at batch 8 passed
cleanly, then the real 40658-step run at batch 32 was (re)launched into the same
`examples/LIBERO/logs/libero_rtc` output dir and confirmed past checkpoint-1000 with no
OOM and finite forward/backward losses (~0.2 range, still in the 5%-warmup phase) — **batch
32 is confirmed safe on a 40GB MIG slice for this model+resolution combination**. ETA from
the observed rate (~2.5 it/s at batch 32) is roughly 5.5 hours for the full run; check
`examples/LIBERO/logs/libero_rtc/checkpoint-*` for the latest progress if picking this up
later. This run's purpose is closer to "does the RTC feature survive a real production-scale
LIBERO fine-tune" than the plan's own 3-episode droid_sample plumbing check — a useful
independent validation, but its results (once complete) should be read as auxiliary to,
not a replacement for, the droid_sample Steps 0-2 above.

### Historical: the original 2x RTX 5090 blocker (superseded above, kept for reference)

Both local GPUs are RTX 5090, **31.34 GiB usable each** per `nvidia-smi`/CUDA. GR00T
N1.7's `launch_finetune.py` hardcodes **full FP32** training (`config.model.load_bf16 =
False`) — there's no CLI flag to change this. For `nvidia/GR00T-N1.7-3B` with
`tune_diffusion_model=True` (needed — `direction_embedding` lives inside the DiT) the
fixed overhead alone is at the edge of or over 31.34 GiB, **independent of batch size or
gradient checkpointing**:

| Config | Trainable params | Peak GPU mem (batch 8 unless noted) | Result |
|---|---|---|---|
| Default (`tune_projector=True`) | 1.62B (51.5%) | ~30.3–30.4 GiB | OOM at first `optimizer.step()`, ~16–64 MiB short |
| Same, batch=2 | 1.62B | ~30.3 GiB | OOM — **batch size barely moves memory** |
| `--no-tune-projector` | 1.29B (41.1%) | ~30.0 GiB, batch=4 | OOM, ~16–20 MiB short |
| `--gradient-checkpointing` | 1.62B | ~30.0 GiB | OOM, same as without — **no measurable effect** |
| Both combined | 1.29B | ~30.0 GiB, batch=4 | OOM, identical numbers to `--no-tune-projector` alone |

Two findings worth carrying forward:
1. **Gradient checkpointing appears to be a no-op in this repo version.**
   `gr00t/model/modules/dit.py` declares `DiT._supports_gradient_checkpointing = True`
   and initializes `self.gradient_checkpointing = False`, but `DiT.forward`'s
   transformer-block loop never checks that flag or wraps blocks in
   `torch.utils.checkpoint.checkpoint(...)`. Worth a real fix/upstream report separately;
   out of scope for this handoff.
2. The bottleneck is **fixed weight+gradient+Adam-optimizer-state overhead**, not
   activations: ~12.6 GiB (3.14B params × 4 bytes, all FP32) + gradients (trainable ×
   4 bytes) + Adam state (trainable × 8 bytes) already totals 28–32 GiB before any batch
   is processed.

At the time, the next thing to try on this class of hardware would have been GPU 1
specifically (idle, ~195 MiB margin) combined with `--no-tune-projector` — moot now that
a ≥40 GB MIG slice was available; **superseded by "Done" above.**

### Two more mitigations discovered (real, keep using these regardless of GPU)

- **`HF_HUB_DISABLE_XET=1`** — the newer HF Xet download backend hung silently for
  over an hour on the first checkpoint download attempt on this network (confirmed via
  `/proc/<pid>/io` showing zero byte movement, and the Xet log going stale) before I
  killed it and disabled Xet. Plain HTTP downloads worked reliably afterward. Keep this
  set for any HF download in this environment.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — didn't fully close the gap
  here but reduces fragmentation-driven near-miss OOMs generically; cheap to keep.

The exact commands actually run for Steps 0-2 (all three passed) are in the "Done" section
above. `--no-tune-projector` was never actually needed once on a ≥40 GB GPU — batch 8
(DROID, 1000-step real fine-tune) and batch 32 (LIBERO, real 40658-step fine-tune, see
below) both fit comfortably with the default `tune_projector=True`.

## Not started — closed-loop evaluation (LIBERO, RoboCasa, RoboLab)

Everything above is **offline/open-loop only** (`open_loop_eval.py` replays a recorded
trajectory, no live env). LIBERO, RoboCasa (+ RoboCasa GR1 Tabletop), and RoboLab all
evaluate **closed-loop** (policy-in-the-loop via the server-client architecture), which
none of the current code reaches. Full design is now in the plan file's new "Extension:
Closed-Loop Evaluation" section — this is a summary; **no code has been written for this
yet**.

### The key finding: one mechanism can't cover all three

- **LIBERO / RoboCasa / RoboCasa-GR1-Tabletop** all route through this repo's own
  `gr00t/eval/rollout_policy.py` against `gr00t/eval/run_gr00t_server.py` — same client
  shape, only `--model-path`/`--embodiment-tag`/`--env-name` differ per benchmark. We
  control this client.
- **RoboLab** is a *separate GitHub repo* (`NVlabs/RoboLab`, not part of Isaac-GR00T, not
  checked out here) with its own Isaac Sim/Isaac Lab rollout loop
  (`policies/gr00t/run.py`) that talks to the *same* `run_gr00t_server.py` over the same
  `get_action` protocol, using `--open-loop-horizon 8` as its execution-horizon
  equivalent. We do **not** control that client's source.

So reaching RoboLab requires a **server-side** diagnostic (transparent to whatever client
is calling it) — a client-side hook in `rollout_policy.py` alone would cover LIBERO/RoboCasa
but structurally cannot reach RoboLab.

### Two complementary mechanisms designed (neither implemented yet)

1. **Server-side `RoundTripMonitor`** (new `gr00t/policy/round_trip_monitor.py`, wired into
   `gr00t/eval/run_gr00t_server.py` via new flags) — wraps any policy, and every N
   `get_action` calls (default 8, matching every benchmark's `--n-action-steps 8` /
   `--open-loop-horizon 8` convention) also issues one internal backward-direction query,
   logging $\mathcal{C}_i$ to a JSONL sidecar. Works for **all four** benchmarks with zero
   client changes, since `options` already forwards transparently through
   `PolicyServer`/`Gr00tPolicy`/`Gr00tSimPolicyWrapper` today (verified, no code change
   needed there).
2. **Client-side hook in `gr00t/eval/rollout_policy.py`** (`_collect_rollout_episodes`) —
   richer for LIBERO/RoboCasa/RoboCasa-GR1 specifically, since it ties $\mathcal{C}_i$
   directly to each episode's task/success via the same `episode_infos` sink
   `task_progress`/`q_score` already use. Cannot reach RoboLab (external client).

Recommendation: build the server-side one first (it's the one that actually reaches
RoboLab); add the client-side one on top for richer LIBERO/RoboCasa reporting.

### What's needed to actually implement and verify this

- The code changes themselves (see the plan file for exact file list/signatures) —
  straightforward given the existing `options`-forwarding plumbing, but untouched so far.
- **A sim environment for at least one of LIBERO/RoboCasa** to verify Mechanism 2 for
  real (`gr00t/eval/sim/LIBERO/setup_libero.sh` or `gr00t/eval/sim/robocasa/setup_RoboCasa.sh`)
  — not installed in this environment. CPU-safe unit tests (mocked policy/env, no sim) can
  and should still be written and pass without this.
- **RoboLab itself** (a separate `git clone https://github.com/NVlabs/RoboLab`, its own
  Isaac Sim/Isaac Lab install) to verify Mechanism 1 against a real RoboLab client — also
  not available here. Mechanism 1's unit tests don't need it either (mocked policy).
- Note this is a **separate environment dependency from the GPU fine-tune blocker above** —
  a bigger GPU alone doesn't unblock this; LIBERO/RoboCasa need their sim setup scripts run,
  RoboLab needs its own separate repo/Isaac Sim install.

## Caveats (unchanged from the original plan — still apply)

3 demo episodes is a plumbing validation, not a calibrated signal; gripper "inversion" is
temporal reversal, not a physical undo; backward examples train vision-dropped only;
`ActionRepresentation.DELTA` isn't supported. See the plan file's Caveats section for full
detail — none of that changed during implementation.
