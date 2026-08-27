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

### Done: real LIBERO+RTC fine-tune, full 40658 steps (separate from the droid_sample validation above)

Beyond the plan's own droid_sample-based Steps 0-2, also ran a full-scale fine-tune on
`examples/LIBERO/libero_goal_no_noops_1.0.0_lerobot` with RTC enabled, matching the
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
    --global-batch-size 32 --dataloader-num-workers 2 \
    --shard-size 1024 --num-shards-per-epoch 100000 --episode-sampling-rate 0.1 \
    --enable-direction-conditioning --p-backward 0.5 \
    --vision-dropout-prob-backward 1.0 --backward-loss-weight 1.0 \
    --resume-from-checkpoint   # add once an output-dir already has checkpoints
```

A first attempt at this (before the `apply_reversed` fix above existed) got as far as
saving its config/processor but crashed on the very first training batch with the
dimension-mismatch bug — the batch size (32) was not the cause, despite looking like an
OOM candidate at first glance. After the fix, a 20-step smoke test at batch 8 passed
cleanly, then the real run was launched at batch 32 and **completed all 40658 steps** into
`examples/LIBERO/logs/libero_rtc/checkpoint-40658` — `train_loss` and per-step
forward/backward losses finite throughout, settling in the ~0.06-0.09 range by the end,
learning rate fully annealed per the cosine schedule.

**It took 5 crash/resume cycles to get there, from two distinct causes** — worth knowing if
running further LIBERO/RTC fine-tunes on a similarly shared machine:

1. **Corrupt checkpoint on resume (2 occurrences).** `--resume-from-checkpoint` always picks
   the highest-numbered `checkpoint-*` in `output_dir`, but a kill mid-save leaves a partial
   directory (missing `trainer_state.json`, sometimes missing model shards/`optimizer.pt`
   too) that still *looks* like a valid checkpoint by name. Resuming into it fails with
   `FileNotFoundError: .../trainer_state.json`. Fix each time: delete the corrupt
   `checkpoint-N` directory (check `ls -la` for a suspiciously small/incomplete file list
   before trusting a checkpoint after any crash) so resume falls back to the next-latest
   complete one.
2. **Shared SLURM cgroup memory pressure (3 occurrences), diagnosed via `/proc/self/cgroup`.**
   This environment runs inside a SLURM job under Kubernetes
   (`kubepods.../slurmstepd.scope/job_7488/...`) with a **256 GiB memory cgroup limit**
   (`/sys/fs/cgroup/.../job_7488/memory.max`) *shared* across every process in that
   allocation — `free -h`'s host-wide 2 TiB total is not the real constraint and looked
   fine throughout, which is why this took a while to diagnose. `memory.current` for
   `job_7488` was observed at 94-99.9996% of the 256 GiB limit while multiple concurrent
   LIBERO fine-tunes (this one plus 3 others not part of this work, apparently launched
   independently by someone else sharing this allocation) were all running. The kernel's
   cgroup-OOM-killer intermittently picked a victim process when the shared total tipped
   over — sometimes this job's main process (silent `SIGKILL`, exit 137, no Python
   traceback), once a caught `RuntimeError: DataLoader worker ... is killed by signal:
   Killed` (exit 1, with a full traceback — this is what actually surfaced the diagnosis).
   Mitigation: cut `--dataloader-num-workers` from 4 to 2 (fewer worker processes each
   independently caching dataset shards in memory) — did not eliminate the pressure (one
   more kill happened after this change) but the run finished without further incident
   afterward. If this recurs on a similarly shared machine, check
   `/proc/self/cgroup` for a SLURM job path, then
   `cat /sys/fs/cgroup/<that-path>/memory.current` vs `memory.max` — don't trust `free -h`
   alone.

**Eval against the final checkpoint** (`open_loop_eval.py`, same invocation shape as the
droid_sample Step 2, `--execution-horizon 8` matching LIBERO's `action_horizon=16`/2,
3 trajectories from `libero_goal_no_noops_1.0.0_lerobot`):

| Traj | Open-loop MSE | Open-loop MAE | C_1 (mean±std, n) | C_2 (mean±std, n) | C_4 (mean±std, n) |
|---|---|---|---|---|---|
| 0 | 0.0224 | 0.0703 | 1.387 ± 0.054 (6) | 1.396 ± 0.054 (5) | 1.426 ± 0.023 (3) |
| 1 | 0.0150 | 0.0540 | 0.860 ± 0.290 (9) | 0.848 ± 0.305 (8) | 0.906 ± 0.323 (6) |
| 2 | 0.0227 | 0.0709 | 0.987 ± 0.242 (9) | 0.971 ± 0.252 (8) | 1.027 ± 0.256 (6) |

Plots: `/tmp/open_loop_eval/libero_traj_rtc_traj{0,1,2}_rtc.png`. Two things to note reading
these against the droid_sample table above: (1) **C_i's absolute scale is not comparable
across datasets** — it's an unnormalized state-space MSE, and LIBERO's `x/y/z/roll/pitch/
yaw/gripper` state representation has a different physical scale than DROID's `eef_9d`/
`gripper_position`/`joint_position`, so LIBERO's C_i sitting an order of magnitude higher
than DROID's is not evidence of a worse model — only within-dataset, cross-depth/cross-
trajectory comparisons are meaningful. (2) Unlike DROID's clean monotonic-with-depth
pattern, LIBERO's C_i is close to flat across depths 1/2/4 for all three trajectories (e.g.
traj 0: 1.387→1.396→1.426) — plausibly because LIBERO's much larger, more diverse training
set (428 episodes vs. DROID's 3) already fits the backward direction well enough at this
depth range that additional rollout depth doesn't compound error as visibly, but this is
speculative; per the plan's Caveat 5, don't over-read either dataset's pattern as a
validated finding without more trajectories and a proper calibration pass
(`gr00t/eval/calibration.py`).

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

## Done (Mechanism 1) — closed-loop evaluation extension, LIBERO benchmarked

Everything in the "Done" sections above is **offline/open-loop only** (`open_loop_eval.py`
replays a recorded trajectory, no live env). LIBERO, RoboCasa (+ RoboCasa GR1 Tabletop), and
RoboLab all evaluate **closed-loop** (policy-in-the-loop via the server-client
architecture). Mechanism 1 (below) is now implemented, tested, and verified against a real
LIBERO closed-loop benchmark; Mechanism 2 (the richer, LIBERO/RoboCasa-only client-side
hook) is not implemented — Mechanism 1 alone was sufficient for the benchmark requested,
and reaches RoboLab too, which Mechanism 2 structurally cannot.

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

### Mechanism 1 — implemented, tested, verified live

**New `gr00t/policy/round_trip_monitor.py`** — `RoundTripMonitor(PolicyWrapper)` wraps any
`BasePolicy` (works with either `Gr00tPolicy`'s nested observation format or
`Gr00tSimPolicyWrapper`'s flat one, auto-detected). Every `round_trip_check_interval`
`get_action` calls, for each env-batch slot, issues one internal backward-direction query
(`options={"direction": 1, "drop_vision": True}`) anchored on the *current* observation,
decodes the reconstructed past state (reusing the same "collapse leading dims, take the
last row" fix from `round_trip_consistency.py` above — this module hits the exact same
`(H, D)` vs. real-policy `(B, H, D)` shape ambiguity), and diffs it against a cached true
state from `depth` calls ago, logging to a JSONL sidecar. Skips the backward query
entirely when no requested depth has a cached target yet (early in a rollout) — a
real-cost-aware optimization beyond what the plan specified. Wired into
`gr00t/eval/run_gr00t_server.py` via `--enable-round-trip-monitor`,
`--round-trip-check-interval`, `--round-trip-depths`, `--round-trip-log-path`; wrapping
happens *after* `--use-sim-policy-wrapper` so it sits outermost, transparent to whatever
client is talking to the server (verified: `options` already forwards end-to-end through
`PolicyServer` → `Gr00tPolicy`/`Gr00tSimPolicyWrapper`, no server/RPC changes needed). New
tests: `tests/gr00t/policy/test_round_trip_monitor.py` (6 tests, mocked `BasePolicy`,
CPU-safe — interval/depth bookkeeping, multi-env-slot independence, exact-recovery at
depth 1, reset-clears-history, and failure isolation: a round-trip check that raises
never breaks real closed-loop control). New README subsection:
`examples/LIBERO/README.md`'s "Round-trip consistency monitoring".

**Mechanism 2** (client-side hook in `gr00t/eval/rollout_policy.py`, richer per-episode
$\mathcal{C}_i$ for LIBERO/RoboCasa/RoboCasa-GR1 specifically) — **not implemented.**
Mechanism 1 alone was sufficient for the benchmark below and reaches RoboLab too, which
Mechanism 2 structurally cannot; revisit if per-episode/per-task $\mathcal{C}_i$
correlation becomes the priority (see Caveat below on why this benchmark didn't exercise
that correlation).

### LIBERO Goal closed-loop benchmark: RTC vs. baseline (real sim, real rollouts)

Ran the full LIBERO Goal suite (10 tasks) closed-loop, 10 episodes/task, 5 parallel envs,
720 max steps, 8 action steps — for both `examples/LIBERO/logs/libero-rtc/libero_goal-rtc`
(this integration's RTC-enabled fine-tune, checkpoint-40658) and a same-recipe non-RTC
baseline (`Isaac-GR00T-base_line/examples/LIBERO/logs/libero-base_line/libero_goal`, a
separate checkout on this machine), via:

```bash
# Server (repeat per checkpoint; add --enable-round-trip-monitor only for the RTC one)
uv run python gr00t/eval/run_gr00t_server.py --model-path <checkpoint> \
    --embodiment-tag LIBERO_PANDA --use-sim-policy-wrapper \
    --enable-round-trip-monitor --round-trip-log-path /tmp/round_trip_consistency/libero_goal_rtc_benchmark.jsonl
# Client (repeat per task/env-name, MUJOCO_EGL_DEVICE_ID required - see Caveat)
MUJOCO_EGL_DEVICE_ID=0 gr00t/eval/sim/LIBERO/libero_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n-episodes 10 --n-envs 5 --max-episode-steps 720 --n-action-steps 8 \
    --policy-client-host 127.0.0.1 --policy-client-port <5555-or-5556> \
    --env-name libero_sim/<task> --video-dir /tmp/<your-own-writable-dir>/<label>
```

| Task | RTC success | Baseline success |
|---|---|---|
| open_the_middle_drawer_of_the_cabinet | 1.0 | 1.0 |
| put_the_bowl_on_the_stove | 1.0 | 1.0 |
| put_the_wine_bottle_on_top_of_the_cabinet | 1.0 | 1.0 |
| open_the_top_drawer_and_put_the_bowl_inside | 1.0 | 1.0 |
| put_the_bowl_on_top_of_the_cabinet | 1.0 | 1.0 |
| push_the_plate_to_the_front_of_the_stove | 1.0 | 1.0 |
| put_the_cream_cheese_in_the_bowl | 1.0 | 1.0 |
| turn_on_the_stove | 1.0 | 1.0 |
| put_the_bowl_on_the_plate | 1.0 | 1.0 |
| put_the_wine_bottle_on_the_rack | 1.0 | 1.0 |
| **Overall (100 episodes each)** | **1.0** | **1.0** |

**Headline finding: identical, perfect closed-loop success (100/100 episodes) for both
checkpoints.** This directly answers the question this benchmark was built to answer —
**adding round-trip-consistency training (direction conditioning + backward examples) does
not measurably hurt forward-task closed-loop performance** on LIBERO Goal, at least at this
scale. It's a ceiling result, though: LIBERO Goal is close to solved by this model class
already (the repo's own reported baseline is 97.5%), so this comparison couldn't have shown
RTC *helping* task success either — a harder/noisier benchmark would be needed to see
whether RTC's C_i-guided trust signal (its actual purpose) ever changes an executed
behavior, since nothing here ever needed to distrust a chunk.

**Round-trip diagnostic data collected during the RTC run**: 600 rows total across the
benchmark (`/tmp/round_trip_consistency/libero_goal_rtc_benchmark.jsonl`).
$\mathcal{C}_i$ was flat across depths (mean 1.265 / 1.264 / 1.277 for depths 1/2/4 —
similar to the open-loop LIBERO eval's flat-with-depth pattern above, unlike DROID's
monotonic rise). Per-key breakdown: `roll` dominates by roughly an order of magnitude
(mean MSE 6.65) over every other channel (`pitch` 0.76, `yaw` 0.16, `z` 0.04, `x`/`y`
~0.005) — worth a closer look before trusting an aggregate $\mathcal{C}_i$ for this
embodiment, but out of scope to chase further here. **Because both checkpoints hit 100%
success everywhere, this run has zero failure cases to correlate $\mathcal{C}_i$ against** —
real diagnostic data was collected end-to-end (validating the mechanism works), but this
particular benchmark cannot speak to whether $\mathcal{C}_i$ predicts failure, since nothing
failed. A benchmark with a genuine success/failure mix (a harder task suite, fewer training
steps, or an out-of-distribution scene) is needed for that question.

### LIBERO 10 (Long) closed-loop benchmark: RTC vs. baseline (the harder suite, real failures)

Ran the same comparison against **LIBERO 10 (Long)** — the long-horizon, multi-step suite,
harder than Goal — using `examples/LIBERO/logs/libero-rtc/libero_10-rtc` (checkpoint-79273)
vs. `Isaac-GR00T-base_line/examples/LIBERO/logs/libero-base_line/libero_10`, same command
shape as above. Per user request, videos and the round-trip log went under
`examples/LIBERO/logs/sim_eval_videos/libero_10/` (in-repo) instead of `/tmp`.

| Task | RTC | Baseline |
|---|---|---|
| LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket | 9/10 | 9/10 |
| LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket | 10/10 | 8/10 |
| KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it | 10/10 | 10/10 |
| KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 10/10 | 9/10 |
| LIVING_ROOM_SCENE5_...mug_on_the_left_plate_and...mug_on_the_right_plate | 10/10 | 10/10 |
| STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy | 11/11 | 11/11 |
| LIVING_ROOM_SCENE6_...mug_on_the_plate_and...chocolate_pudding... | 8/10 | 9/10 |
| LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket | 10/10 | 10/10 |
| KITCHEN_SCENE8_put_both_moka_pots_on_the_stove | 7/10 | 8/11 |
| KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it | 7/10 | 10/10 |
| **Overall (sum successes / sum episodes)** | **92/101 = 91.1%** | **94/102 = 92.2%** |

(Episode counts of 10 or 11 rather than a flat 10 come from the vectorized collector: with
5 parallel envs and `--n-episodes 10`, individual envs finish at slightly different times,
so the batch occasionally collects one extra episode before stopping — not a bug, just how
`rollout_policy.py`'s vectorized loop terminates.)

**This is a genuinely mixed, informative result** — unlike Goal's clean sweep. RTC and
baseline are statistically close overall (91.1% vs. 92.2%, a 2-episode difference out of
~100), each winning some tasks and losing others: RTC clearly ahead on
`put_both_the_cream_cheese_box_and_the_butter` (10/10 vs. 8/10) and
`put_the_black_bowl_in_the_bottom_drawer` (10/10 vs. 9/10), clearly behind on
`put_the_yellow_and_white_mug_in_the_microwave` (7/10 vs. 10/10) and roughly tied
(within 1 episode) elsewhere. **No consistent direction** — this reads as within-noise
parity between RTC and non-RTC training on this suite, not evidence that RTC training
helps or hurts long-horizon task performance either way at this scale.

**Round-trip diagnostic**: 1755 rows logged (`libero_10_rtc_benchmark.jsonl`), same flat
pattern across depths as Goal (means 1.603 / 1.603 / 1.604 for depths 1/2/4) and the same
`roll`-dominates-per-key pattern (mean MSE 7.67, next highest `pitch` at 1.81, everything
else <0.1). **Attempted to correlate per-task $\mathcal{C}_i$ against the two weak RTC
tasks (KITCHEN_SCENE8 at 7/10, KITCHEN_SCENE6 at 7/10) and could not** — the JSONL schema
has no task/episode field, and the only available proxy (segmenting rows by detecting
`call_index` resets per `env_slot`, on the theory that a new `rollout_policy.py` client
connecting triggers exactly one `RoundTripMonitor.reset()` per task) produced far more
segments (46) than tasks (10), because **individual env slots' `call_index` also appears to
reset mid-task**, most likely because `RoundTripMonitor` identifies an env purely by its
*positional* index in the batched observation (`range(batch_size)`), and the vectorized
collector's batch can shrink as individual sub-envs finish their share of the episode quota
before others — reassigning what looks like "slot 2" to a different underlying environment
mid-rollout. **This is a real, previously-undocumented limitation of Mechanism 1's current
identity tracking**, not a data-quality issue with this run: a future improvement would tag
each JSONL row with a stable per-episode/task identifier (passed through `options`, if the
client is willing to supply one) rather than relying on positional batch index across calls.

### Caveats (closed-loop-specific, found running this for real)

1. **`MUJOCO_EGL_DEVICE_ID` must be set to a plain integer for any `rollout_policy.py`
   client on this class of machine.** MuJoCo's offscreen EGL renderer parses
   `CUDA_VISIBLE_DEVICES` as a comma-separated list of integer device indices; on a
   MIG-enabled host where that variable holds MIG UUID strings (e.g.
   `MIG-9bd72ea7-...`), `int(x)` raises `ValueError` and the client crashes before ever
   reaching the server. `robosuite`'s EGL context code checks `MUJOCO_EGL_DEVICE_ID` first
   — set it to `0` (or whichever physical GPU index has a working display pipeline; MIG's
   compute partitioning doesn't extend to EGL/OpenGL rendering, so a working index is
   independent of which MIG slice your compute allocation actually is).
2. **Shared `/tmp` on a multi-tenant machine breaks two things `robosuite`/
   `rollout_policy.py` hardcode.** `robosuite.utils.log_utils` unconditionally opens
   `/tmp/robosuite.log` at import time; if another user already owns that file, import
   crashes with `PermissionError`. Fixed here by patching the venv's own
   `robosuite/macros.py` (`FILE_LOGGING_LEVEL = "DEBUG"` → `None`) — a per-venv fix, not
   a repo change, since `robosuite`'s own `macros_private` override mechanism doesn't
   actually reach this code path (a known upstream inconsistency: `macros_private`
   overrides names imported directly into `robosuite`'s namespace, but `log_utils.py`
   reads `robosuite.macros.FILE_LOGGING_LEVEL` via `import robosuite.macros as macros`,
   which private-macro overrides never touch). Separately, `rollout_policy.py` defaults
   `--video-dir` to `/tmp/sim_eval_videos_<env>_<uuid>` — if a directory one level up
   (`/tmp/sim_eval_videos_<env_prefix>`) already exists and is owned by another user
   without world-write, `mkdir` fails with the same `PermissionError`. Always pass an
   explicit `--video-dir` under a path only you own on a shared machine.
3. **This machine runs a real SLURM job (see the GPU section's cgroup discussion above) at
   very high load** (observed `uptime` load average ~100-118 on 64 cores while running
   this benchmark) — server/client startup and rendering were all noticeably slower than
   the numbers would suggest on a quiet machine; budget accordingly, and don't assume a
   slow-but-still-progressing process (check `ps -o pid,etimes,time,%cpu` — CPU time
   should keep advancing) is hung.
4. **Per-slot `reset()` granularity, flagged as unconfirmed in the plan, is now
   confirmed**: `gr00t/eval/rollout_policy.py` calls `policy.reset()` exactly once, with no
   arguments, before the rollout loop begins — it never detects or forwards individual
   sub-env auto-resets within the vectorized batch. `RoundTripMonitor` clears its entire
   ring buffer on any `reset()` call, per the plan's documented fallback; a round-trip
   window spanning an undetected mid-rollout episode boundary is a real but rare
   (window-length-vs-episode-length) source of noise in the JSONL log, not a crash.

## Caveats (unchanged from the original plan — still apply)

3 demo episodes is a plumbing validation, not a calibrated signal; gripper "inversion" is
temporal reversal, not a physical undo; backward examples train vision-dropped only;
`ActionRepresentation.DELTA` isn't supported. See the plan file's Caveats section for full
detail — none of that changed during implementation.
